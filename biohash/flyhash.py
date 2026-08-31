"""FlyHash: locality-sensitive hashing after the fruit-fly olfactory circuit.

Reference: Dasgupta, Stevens & Navlakha, "A neural algorithm for a fundamental
computing problem", *Science* 358 (2017).

The fly inverts conventional LSH. Conventional schemes project *down* to a short
dense code; the fly projects *up* -- roughly 50 odorant receptor types fan out
into ~2,000 Kenyon cells through sparse random connections -- and then a single
inhibitory neuron (APL) applies winner-take-all, leaving only the top ~5% of
cells active. The output is a high-dimensional *sparse binary tag*, and similar
inputs produce overlapping tags.

Why this and not a face embedding
---------------------------------
Embeddings are the obvious choice and the wrong one here:

* They are **invertible**. Template-inversion attacks reconstruct a recognisable
  face from an embedding, so a database of embeddings is a database of faces
  with extra steps.
* They are **linkable**. The same embedding at two institutions links a person
  across them, which fails GDPR/DPDP handling of special-category data and
  makes cross-institution sharing legally impossible.

A FlyHash tag gives us three properties an embedding cannot:

1. **A tag is a set, not a vector.** Similarity is set intersection, so
   near-duplicate detection composes directly with the Bloom-filter exchange in
   ``privacy/psi.py``. Near-duplicate face detection and the cross-institution
   privacy layer become one mechanism instead of two.
2. **Revocable and unlinkable.** Each institution holds its own secret
   projection seed, so the same face yields non-comparable tags at different
   institutions, and a compromised tag is revoked by reseeding. That is a
   cancelable biometric in the sense of ISO/IEC 24745.
3. **Lossy by construction.** Winner-take-all keeps only *which* cells won and
   discards all magnitude information -- a many-to-one map, so there is
   materially less to invert.

Claim discipline
----------------
We claim revocability, unlinkability and lossy construction under a stated
threat model. We do **not** claim cryptographic non-invertibility: that is not
proven for FlyHash, and overclaiming it does not survive a technical question.
See ``THREAT_MODEL`` below.
"""

from __future__ import annotations

import hashlib

import numpy as np

from contracts.schemas import SparseTag

#: Shared tag-space width. Every hasher in the project emits tags of this width
#: so that faces, document layouts and categorical attributes are mutually
#: bundleable and exchangeable through one Bloom filter.
TAG_DIM = 4096

THREAT_MODEL = """
Assumed adversary: an attacker who obtains the stored tag database, but not the
institution's projection seed.

Holds:
  * Tags from different institution seeds are not comparable, so a stolen tag
    set cannot be joined against another institution's database.
  * Reseeding invalidates every stolen tag (revocability).
  * Winner-take-all discards magnitude; the map from input to tag is
    many-to-one, so exact reconstruction is not possible.

Does NOT hold:
  * If the projection seed leaks alongside the tags, an attacker can mount a
    template-reconstruction search and recover a coarse approximation of the
    input. Seeds must be held in the same class of storage as signing keys.
  * We make no cryptographic non-invertibility claim. This is a privacy
    engineering control, not an encryption scheme.
"""


def _seed_from_id(seed_id: str, salt: str = "") -> int:
    """Derive a stable 64-bit seed from an institution id.

    Deterministic across processes and platforms -- ``hash()`` is not, which is
    why we use blake2b here.
    """
    digest = hashlib.blake2b(f"{salt}|{seed_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class FlyHash:
    """Sparse random expansion followed by winner-take-all.

    Parameters
    ----------
    input_dim:
        Width of the input feature vector (the "odorant receptor" layer).
    expansion_factor:
        Kenyon-cell layer is ``expansion_factor * input_dim`` wide. The fly uses
        roughly 40x; 20-40x works well here.
    sampling_rate:
        Fraction of input dimensions each Kenyon cell samples. The fly wires
        about 6 of 50 receptors per cell, so ~0.12.
    sparsity:
        Fraction of Kenyon cells that survive winner-take-all. The fly keeps
        about 5%.
    seed_id:
        Institution identifier. Two institutions with different ``seed_id``
        produce non-comparable tags for the same input -- this is the
        unlinkability property, and it is load-bearing.
    salt:
        Optional secret. Rotating the salt revokes every tag issued under it.
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 32,
        sampling_rate: float = 0.12,
        sparsity: float = 0.05,
        seed_id: str = "default",
        salt: str = "",
        dim: int | None = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1]")
        if not 0.0 < sparsity < 1.0:
            raise ValueError("sparsity must be in (0, 1)")

        self.input_dim = input_dim
        # ``dim`` pins the output width directly, which matters because tags
        # from different feature spaces -- a 99-dim face descriptor and a
        # 160-dim document layout -- must land in the *same* space to be
        # bundled into an identity hypervector or exchanged through a shared
        # Bloom filter. Deriving width from an expansion factor alone silently
        # puts them in incompatible spaces.
        self.dim = int(dim) if dim is not None else int(input_dim * expansion_factor)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.sparsity = sparsity
        self.seed_id = seed_id
        self.hash_length = max(1, int(round(self.dim * sparsity)))

        rng = np.random.default_rng(_seed_from_id(seed_id, salt))
        # Sparse binary connectivity: each Kenyon cell samples a random subset
        # of the receptor layer. Boolean, not Gaussian -- the fly's connections
        # are present or absent, and this is cheaper besides.
        n_sample = max(1, int(round(input_dim * sampling_rate)))
        self._proj = np.zeros((self.dim, input_dim), dtype=np.float32)
        for row in range(self.dim):
            picks = rng.choice(input_dim, size=n_sample, replace=False)
            self._proj[row, picks] = 1.0

        # Tie-break ordering, so that equal activations resolve deterministically
        # rather than by numpy's argpartition implementation detail.
        self._tiebreak = rng.permutation(self.dim).astype(np.int64)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        """Divisive normalisation, mimicking the fly's concentration invariance.

        The circuit responds to the *pattern* of receptor activation, not its
        overall intensity. Without this a brighter photo of the same face would
        tag differently, which would defeat the whole purpose.
        """
        x = np.asarray(x, dtype=np.float32)
        mean = np.abs(x).mean(axis=-1, keepdims=True)
        x = x / np.where(mean > 1e-12, mean, 1.0)
        return x - x.mean(axis=-1, keepdims=True)

    def _winners(self, activations: np.ndarray) -> np.ndarray:
        """Winner-take-all. Returns the indices of the top-k cells, sorted."""
        k = self.hash_length
        # Rank primarily by activation, break ties by the fixed permutation so
        # results are reproducible across platforms.
        order = np.lexsort((self._tiebreak, -activations))
        return np.sort(order[:k])

    # -- public API --------------------------------------------------------

    def tag(self, x: np.ndarray) -> SparseTag:
        """Hash one feature vector into a sparse tag."""
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.shape[0] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[0]}")
        activations = self._proj @ self._normalise(x)
        idx = self._winners(activations)
        return SparseTag(dim=self.dim, indices=tuple(int(i) for i in idx), seed_id=self.seed_id)

    def tag_batch(self, X: np.ndarray) -> list[SparseTag]:
        """Hash a stack of feature vectors. ``X`` is ``(n, input_dim)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"expected (n, {self.input_dim}), got {X.shape}")
        acts = self._normalise(X) @ self._proj.T
        return [
            SparseTag(
                dim=self.dim,
                indices=tuple(int(i) for i in self._winners(acts[i])),
                seed_id=self.seed_id,
            )
            for i in range(X.shape[0])
        ]


def tag_overlap(a: SparseTag, b: SparseTag) -> float:
    """Jaccard overlap between two tags. Zero across different seeds."""
    return a.overlap(b)


def chance_overlap(dim: int, hash_length: int) -> float:
    """Expected Jaccard overlap of two independent random tags.

    The baseline that a "near-duplicate" threshold has to clear. With 5%
    sparsity this sits near 0.026, so a threshold of 0.30 is far above noise.
    """
    expected_intersection = hash_length * hash_length / dim
    return expected_intersection / (2 * hash_length - expected_intersection)


def minhash(tag: SparseTag, n_hashes: int = 64, seed: str = "") -> list[int]:
    """MinHash signature of a tag's index set.

    Two sets agree on any given minhash with probability equal to their Jaccard
    similarity -- that is the defining property, and it is what makes an
    *approximate* similarity question answerable with *exact*-match machinery
    such as a Bloom filter or a hash join.

    Uses universal hashing ``(a*x + b) mod p`` over the index universe rather
    than a cryptographic digest per element, so a signature costs a handful of
    vectorised operations.
    """
    import numpy as np

    idx = np.asarray(tag.indices, dtype=np.int64)
    if idx.size == 0:
        return [0] * n_hashes
    rng = np.random.default_rng(_seed_from_id(f"minhash|{seed}", "") % (2**32))
    p = 2_147_483_647  # Mersenne prime > any tag dim we use
    a = rng.integers(1, p, size=n_hashes, dtype=np.int64)
    b = rng.integers(0, p, size=n_hashes, dtype=np.int64)
    # (n_hashes, n_indices) -> min along the index axis
    hashed = (a[:, None] * idx[None, :] + b[:, None]) % p
    return [int(v) for v in hashed.min(axis=1)]


def minhash_bands(
    tag: SparseTag, n_bands: int = 32, band_size: int = 2, seed: str = ""
) -> list[str]:
    """Banded MinHash keys: two tags share a band with probability ``J**band_size``.

    Banding sharpens the similarity threshold. At ``band_size=2``, a
    near-duplicate pair (J~0.6) matches ~36% of bands while an unrelated pair
    (J~0.026) matches ~0.07% -- a separation of roughly 500x, which is what
    turns a noisy overlap into a usable exact-match indicator.

    This replaces an earlier ``cluster_key`` that hashed consecutive slices of
    the sorted index set. That required ~25 indices to match *exactly* and
    therefore almost never fired: measured same-ring recall was 0.5%. Similarity
    means overlapping sets, not identical runs, and only MinHash captures that.
    """
    sig = minhash(tag, n_hashes=n_bands * band_size, seed=seed)
    out = []
    for b in range(n_bands):
        band = sig[b * band_size : (b + 1) * band_size]
        payload = f"{b}|" + ",".join(map(str, band))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
        out.append(digest)
    return out
