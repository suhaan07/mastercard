"""Hyperdimensional binding of attribute tags into one identity fingerprint.

Following Kanerva's sparse distributed memory and the vector-symbolic
architecture line of work, we combine an identity's several similarity tags --
face, document template, address, device -- into a single hypervector, so that
ring similarity falls out of one comparison instead of four.

Two operations, both closed over :class:`SparseTag`:

* **bind(tag, role)** -- moves a tag into a role-specific region of the space via
  a bijective index permutation. Binding the same face under ``face`` and under
  ``document`` yields uncorrelated tags, so a face that happens to collide with
  someone else's document template cannot create a phantom link.
* **bundle(tags)** -- superposes tags into one, thinned back down to the target
  sparsity by vote count. Similarity in *any* bound component lifts similarity
  of the bundle, which is the behaviour ring detection wants.

Both are deterministic, seed-scoped, and preserve the property that a tag is a
set -- so bundles remain compatible with the Bloom-filter exchange in
``privacy/psi.py``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence

from contracts.schemas import SparseTag


def _role_params(role: str, dim: int) -> tuple[int, int]:
    """Derive a bijective affine permutation ``i -> (a*i + b) mod dim`` for a role.

    ``a`` is forced odd, which guarantees it is coprime to any power-of-two
    ``dim``; for other ``dim`` we step until gcd is 1. Bijectivity matters --
    a non-injective map would collapse distinct cells together and quietly
    destroy the similarity structure.
    """
    digest = hashlib.blake2b(role.encode("utf-8"), digest_size=16).digest()
    a = int.from_bytes(digest[:8], "big") % dim
    b = int.from_bytes(digest[8:], "big") % dim
    a |= 1
    from math import gcd

    while gcd(a, dim) != 1:
        a = (a + 2) % dim or 1
    return a, b


def bind(tag: SparseTag, role: str) -> SparseTag:
    """Move ``tag`` into the subspace for ``role``. Bijective, so lossless."""
    a, b = _role_params(role, tag.dim)
    moved = tuple(sorted(((a * i + b) % tag.dim) for i in tag.indices))
    return SparseTag(dim=tag.dim, indices=moved, seed_id=tag.seed_id)


def unbind(tag: SparseTag, role: str) -> SparseTag:
    """Inverse of :func:`bind`. Used only in tests -- the pipeline never unbinds."""
    a, b = _role_params(role, tag.dim)
    a_inv = pow(a, -1, tag.dim)
    moved = tuple(sorted((a_inv * (i - b)) % tag.dim for i in tag.indices))
    return SparseTag(dim=tag.dim, indices=moved, seed_id=tag.seed_id)


def bundle(tags: Sequence[SparseTag], hash_length: int | None = None) -> SparseTag:
    """Superpose tags, thinned to ``hash_length`` cells by vote count.

    Ties are broken by a deterministic hash of the index rather than by sort
    order, so that a bundle does not systematically favour low indices.
    """
    tags = [t for t in tags if t is not None]
    if not tags:
        raise ValueError("cannot bundle an empty tag sequence")
    dim = tags[0].dim
    seed_id = tags[0].seed_id
    if any(t.dim != dim for t in tags):
        raise ValueError("cannot bundle tags of differing dim")
    if any(t.seed_id != seed_id for t in tags):
        raise ValueError("cannot bundle tags from different institution seeds")

    if hash_length is None:
        hash_length = max(len(t.indices) for t in tags)

    votes: Counter[int] = Counter()
    for t in tags:
        votes.update(t.indices)

    def _rank(item: tuple[int, int]) -> tuple[int, int]:
        idx, count = item
        jitter = int.from_bytes(
            hashlib.blake2b(f"{seed_id}|{idx}".encode(), digest_size=4).digest(), "big"
        )
        return (-count, jitter)

    ranked = sorted(votes.items(), key=_rank)
    kept = tuple(sorted(idx for idx, _ in ranked[:hash_length]))
    return SparseTag(dim=dim, indices=kept, seed_id=seed_id)


def identity_hypervector(
    *,
    face_tag: SparseTag,
    doc_template_tag: SparseTag,
    address_tag: SparseTag | None = None,
    device_tag: SparseTag | None = None,
    hash_length: int | None = None,
) -> SparseTag:
    """Bind an identity's attribute tags into a single comparable fingerprint.

    Ring members share infrastructure, so their hypervectors overlap even when
    no single attribute matches exactly -- which is precisely the case a
    per-attribute exact-match rule misses.
    """
    parts = [bind(face_tag, "face"), bind(doc_template_tag, "doc_template")]
    if address_tag is not None:
        parts.append(bind(address_tag, "address"))
    if device_tag is not None:
        parts.append(bind(device_tag, "device"))
    return bundle(parts, hash_length=hash_length)


def ring_fingerprint(hypervectors: Iterable[SparseTag], hash_length: int | None = None) -> SparseTag:
    """Bundle a set of identity hypervectors into one ring-level fingerprint.

    Cells that survive are those shared across many ring members -- the
    operator's reused infrastructure, isolated from each member's incidentals.
    """
    return bundle(list(hypervectors), hash_length=hash_length)


def categorical_tag(value: str, dim: int, hash_length: int, seed_id: str) -> SparseTag:
    """Deterministic random tag for a categorical value (an address, a device id).

    Exact-equal values give identical tags; different values give tags at chance
    overlap. This lets categorical attributes take part in bundling alongside
    the genuinely continuous face and document tags.
    """
    indices: set[int] = set()
    counter = 0
    while len(indices) < hash_length:
        digest = hashlib.blake2b(
            f"{seed_id}|{value}|{counter}".encode(), digest_size=32
        ).digest()
        for i in range(0, 32, 4):
            if len(indices) >= hash_length:
                break
            indices.add(int.from_bytes(digest[i : i + 4], "big") % dim)
        counter += 1
    return SparseTag(dim=dim, indices=tuple(sorted(indices)), seed_id=seed_id)
