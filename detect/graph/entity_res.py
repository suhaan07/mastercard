"""Entity resolution: deciding when two applicants are the same operation.

Three linking mechanisms, in increasing order of how much they cost to compute:

1. **Exact shared tokens** -- the same device, address, phone or DOB token under
   two different names. Cheap, and the strongest single signal, because a
   legitimate pair almost never shares one.
2. **Email handle shape** -- batch-generated identities share a construction
   pattern (``aaaa9999``) even when no two addresses match. Weak alone, useful
   in combination.
3. **Near-duplicate tags** -- faces from one generator run, documents from one
   template. This is the expensive one, and it is why ``biohash.flyhash``
   provides LSH banding: comparing every applicant to every other is O(n^2) and
   at 4,000 applicants that is eight million comparisons. Banding proposes
   candidates in one pass and we compute exact overlap only for those.

The near-duplicate threshold is 0.30, taken from the measured ROC in
``tests/test_biohash.py``: ~90% same-ring recall at a ~0.6% false-link rate.
It is a tunable, not a constant of nature, and the metrics report quotes it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from contracts.schemas import OnboardingEvent, SparseTag

#: Per-attribute near-duplicate thresholds, each chosen by measuring precision
#: and recall against planted rings at full population scale (4,279 identities,
#: 9.15M candidate pairs). A single shared threshold is wrong here because the
#: three tag spaces have very different baseline overlap:
#:
#:   attribute              thresh  precision  same-ring pair recall
#:   face_tag                 0.50      0.912                  0.295
#:   doc_template_tag         0.70      0.977                  0.419
#:   identity_hypervector     0.30      1.000                  0.351
#:
#: Two lessons are baked into these numbers. First, precision matters far more
#: than pairwise recall: community detection connects a ring transitively, so
#: one good link per member is enough, whereas false links chain unrelated
#: people into large phantom clusters. At face threshold 0.30 the linker emitted
#: 19,404 false links and legitimate communities became 93% bound by tag
#: similarity alone. Second, the *hypervector* is the strongest linker by a
#: wide margin -- perfect precision at the loosest threshold -- because binding
#: face, document, address and device means a coincidental match on any single
#: attribute cannot produce a match on the bundle. That is the case for
#: hyperdimensional binding, measured rather than asserted.
#:
#: doc_template_tag needs the highest threshold because every card layout is a
#: portrait box beside a stack of fields, so unrelated templates share a large
#: structural baseline -- the same effect that forced population whitening on
#: the face descriptor.
NEAR_DUPLICATE_THRESHOLDS: dict[str, float] = {
    "face_tag": 0.50,
    "doc_template_tag": 0.70,
    "identity_hypervector": 0.30,
}
NEAR_DUPLICATE_THRESHOLD = 0.50

#: Token fields whose exact reuse across identities is suspicious. A shared
#: name token is not here -- two people genuinely share a name.
SHARED_TOKEN_FIELDS = ("device_id", "address_token", "phone_token", "dob_token", "ip_id")


@dataclass(frozen=True)
class Link:
    """A proposed relationship between two identities, with its evidence."""

    left: str
    right: str
    kind: str
    weight: float
    detail: str


def _pairs(members: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]


def shared_token_links(events: list[OnboardingEvent], max_group: int = 60) -> list[Link]:
    """Link identities that share an exact token under different names.

    ``max_group`` guards against a shared value so common it is meaningless --
    a corporate NAT gateway seen by 4,000 customers is not a ring, and expanding
    it to pairs would produce eight million useless edges. Genuine ring sharing
    is concentrated, so a large group is evidence of infrastructure, not fraud.
    """
    links: list[Link] = []
    for field in SHARED_TOKEN_FIELDS:
        groups: dict[str, list[str]] = defaultdict(list)
        for ev in events:
            groups[getattr(ev, field)].append(ev.identity_id)
        for value, members in groups.items():
            if not 2 <= len(members) <= max_group:
                continue
            # Sharing is more surprising in a small group than a large one.
            weight = 1.0 / (len(members) - 1) ** 0.5
            for a, b in _pairs(members):
                links.append(Link(a, b, f"shared_{field}", weight, value))
    return links


def pii_recombination_links(events: list[OnboardingEvent]) -> list[Link]:
    """The classic synthetic-identity tell: one person's details under another's name.

    A DOB or phone token appearing beside two different name tokens is exactly
    the recombination pattern that manufactured identities leave behind.
    """
    links: list[Link] = []
    for field in ("dob_token", "phone_token"):
        by_value: dict[str, list[OnboardingEvent]] = defaultdict(list)
        for ev in events:
            by_value[getattr(ev, field)].append(ev)
        for value, evs in by_value.items():
            if len(evs) < 2 or len(evs) > 20:
                continue
            if len({e.name_token for e in evs}) < 2:
                continue  # same person, same name -- not recombination
            for a, b in _pairs([e.identity_id for e in evs]):
                links.append(Link(a, b, f"pii_recombination_{field}", 1.0, value))
    return links


#: An email-handle shape is only evidence when it is *rare*. A shape shared by
#: a fifth of the customer base describes the population, not a ring.
EMAIL_SHAPE_MAX_SHARE = 0.02

#: And rare is not enough on its own: a shape held by two people out of 5,000
#: is rare by construction, so the group also has to be small in absolute terms
#: before it looks like one operator's generator rather than a coincidence.
EMAIL_SHAPE_MAX_GROUP = 40


def email_shape_links(
    events: list[OnboardingEvent],
    max_group: int = EMAIL_SHAPE_MAX_GROUP,
    max_share: float = EMAIL_SHAPE_MAX_SHARE,
) -> list[Link]:
    """Identities whose email handles share a *rare* construction pattern.

    The rarity gate is the whole feature. Linking on shape alone measured
    **0 same-ring precision across 1,078 links** on the sloppy scenario -- every
    other link kind runs between 0.26 and 1.00 -- because the common shapes
    ("five letters then four digits") are what most people's email looks like,
    and both populations draw from the same generator. Those links were then the
    largest single input to retro-propagation, which is how one confirmation
    came to flag 73% of a population.

    Gating on population share keeps the mechanism honest and keeps it useful on
    data where a ring's handles really are distinctively constructed: there, the
    shape is rare precisely because one script produced it.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        groups[ev.email_handle_shape].append(ev.identity_id)

    total = max(1, len(events))
    links: list[Link] = []
    for shape, members in groups.items():
        if not 2 <= len(members) <= max_group:
            continue
        share = len(members) / total
        if share > max_share:
            continue
        # Rarer shapes are stronger evidence; the size term keeps a large group
        # from contributing more total weight than a small, tighter one.
        rarity = 1.0 - (share / max_share)
        weight = 0.35 * rarity / (len(members) - 1) ** 0.5
        for a, b in _pairs(members):
            links.append(Link(a, b, "email_shape", weight, shape))
    return links


def near_duplicate_tag_links(
    events: list[OnboardingEvent],
    attribute: str = "face_tag",
    threshold: float | None = None,
    chunk: int = 512,
) -> list[Link]:
    """Link identities whose tags are near-duplicates.

    Computes exact pairwise Jaccard overlap via a sparse boolean incidence
    matrix. Tags are sets of ~200 indices in a 4096-wide space, so the
    intersection of every pair is a single sparse matrix product, chunked to
    keep the dense result bounded.

    This replaced an LSH-banding pass that was quietly broken. Banding hashed
    consecutive slices of the sorted index set and required ~25 indices to match
    *exactly* -- which for two genuinely similar tags essentially never happens,
    because similarity means overlapping sets, not identical runs. Same-ring
    recall measured 0.5%. Proper banding needs minhash signatures; at this
    population size the exact computation is cheaper than getting that right,
    and it is exact.

    Scale note: this is O(n^2) in memory-bounded chunks and is comfortable to
    the low tens of thousands of identities. Beyond that, minhash-banded
    blocking becomes necessary -- and must be validated for recall, not assumed.
    """
    import numpy as np
    from scipy import sparse

    if threshold is None:
        threshold = NEAR_DUPLICATE_THRESHOLDS.get(attribute, NEAR_DUPLICATE_THRESHOLD)
    n = len(events)
    if n < 2:
        return []

    tags: list[SparseTag] = [getattr(ev, attribute) for ev in events]
    dim = tags[0].dim
    sizes = np.array([len(t.indices) for t in tags], dtype=np.int32)

    rows = np.repeat(np.arange(n, dtype=np.int32), sizes)
    cols = np.fromiter((i for t in tags for i in t.indices), dtype=np.int32, count=int(sizes.sum()))
    mat = sparse.csr_matrix(
        (np.ones(cols.shape[0], dtype=np.float32), (rows, cols)), shape=(n, dim)
    )

    links: list[Link] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        inter = (mat[start:stop] @ mat.T).toarray()
        union = sizes[start:stop, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        # Upper triangle only, so each pair is emitted once.
        local_i, j = np.nonzero(jac >= threshold)
        for li, jj in zip(local_i, j):
            i = start + int(li)
            jj = int(jj)
            if jj <= i:
                continue
            links.append(
                Link(
                    events[i].identity_id,
                    events[jj].identity_id,
                    f"near_duplicate_{attribute}",
                    float(jac[li, jj]),
                    f"{jac[li, jj]:.3f}",
                )
            )
    return links


def resolve(events: list[OnboardingEvent]) -> list[Link]:
    """All entity-resolution links for a set of onboarding events."""
    return [
        *shared_token_links(events),
        *pii_recombination_links(events),
        *email_shape_links(events),
        *near_duplicate_tag_links(events, "face_tag"),
        *near_duplicate_tag_links(events, "doc_template_tag"),
        *near_duplicate_tag_links(events, "identity_hypervector"),
    ]
