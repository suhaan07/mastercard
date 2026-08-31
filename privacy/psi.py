"""Cross-institution indicator exchange without sharing PII or biometrics.

The question two institutions need to answer is narrow: *do we both see this
suspicious thing?* Not "send me your customers", not "here is a face" -- just
whether an indicator is common to both books.

This layer exists because of the representation choice made in
``biohash/flyhash.py``. A FlyHash tag is a **set of indices**, not a vector, so"are these two faces similar" and "do our two institutions share an indicator"
are the same operation -- set intersection. That collapse is why the privacy
layer costs almost nothing to build here, and it is why the design doc's
instinct to cut it no longer applies.

Two mechanisms:

* :class:`IndicatorFilter` -- a Bloom filter of salted indicator digests. One
  institution publishes it; another tests membership locally. Nothing that
  crosses the wire can be enumerated back into a customer list, and false
  positives are bounded and quantified.
* :func:`private_set_intersection` -- a two-party exchange over a shared
  epoch salt, returning only the indicators genuinely held by both.

**Threat model, stated plainly.** This gives unlinkability and revocability
against an adversary who sees the exchanged artifacts. It is not encryption. A
party who already holds a candidate indicator can test whether the other side
has it -- that is precisely the intended function, and it means the exchange
must be limited to parties with a legitimate basis, rate-limited, and audited.
Against an adversary holding the epoch salt *and* a dictionary of candidate
indicators, membership is confirmable. Salts rotate per epoch to bound that
window."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from contracts.schemas import SparseTag


def _digest(value: str, salt: str, index: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{salt}|{index}|{value}".encode("utf-8"), digest_size=8).digest(),
        "big",
    )


@dataclass
class IndicatorFilter:
    """A Bloom filter of salted indicator digests.

    Sized from the expected item count and a target false-positive rate, both
    recorded so the receiving side can reason about what a hit is worth. An
    unsized filter is a filter whose error rate nobody can quote, which is not
    something to put in a regulatory conversation.
    """

    n_bits: int
    n_hashes: int
    salt: str
    bits: bytearray = field(default_factory=bytearray)
    n_items: int = 0
    institution_id: str = ""

    @classmethod
    def sized_for(
        cls, expected_items: int, target_fp: float = 0.001, salt: str = "", institution_id: str = ""
    ) -> "IndicatorFilter":
        expected_items = max(1, expected_items)
        n_bits = max(64, int(math.ceil(-expected_items * math.log(target_fp) / (math.log(2) ** 2))))
        n_hashes = max(1, int(round((n_bits / expected_items) * math.log(2))))
        return cls(
            n_bits=n_bits,
            n_hashes=n_hashes,
            salt=salt,
            bits=bytearray((n_bits + 7) // 8),
            institution_id=institution_id,
        )

    def _positions(self, value: str) -> list[int]:
        return [_digest(value, self.salt, i) % self.n_bits for i in range(self.n_hashes)]

    def add(self, value: str) -> None:
        for p in self._positions(value):
            self.bits[p >> 3] |= 1 << (p & 7)
        self.n_items += 1

    def __contains__(self, value: str) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(value))

    @property
    def false_positive_rate(self) -> float:
        """Realised false-positive probability at the current load."""
        if self.n_items == 0:
            return 0.0
        k, m, n = self.n_hashes, self.n_bits, self.n_items
        return float((1 - math.exp(-k * n / m)) ** k)

    @property
    def size_bytes(self) -> int:
        return len(self.bits)


def tag_indicators(tag: SparseTag, n_bands: int = 32, band_size: int = 4) -> list[str]:
    """Turn a sparse tag into exact-matchable indicators via banded MinHash.

    A whole tag is too specific to match across institutions -- two views of the
    same face differ in a fraction of their winning cells, so exact tag equality
    almost never holds. MinHash restores approximate matching through an
    exact-match primitive: two tags agree on each hash with probability equal to
    their Jaccard similarity, and banding sharpens that into a threshold.
    ``band_size`` is 4 by design. A band matches with probability ``J**band_size``,
    so a wider band suppresses coincidence far faster than it suppresses signal.
    At band_size 2 the exchange did not work at population scale: matched
    identities hit 29 of 32 bands but *unrelated* ones still hit 7.9, because
    with hundreds of candidate tags on the far side, coincidental band
    collisions accumulate. At band_size 4 the same comparison gives 22.3 hits
    for a genuine match against 0.06 for an unrelated identity -- a separation
    of ~98 standard deviations, and a decision threshold with room to spare.

    An earlier version used shingles of consecutive indices, which required six
    indices to align at the same window position and barely separated the
    classes at all (19 shared indicators against a control of 9).

    This is a *within-seed* operation. Tags from different institution seeds are
    not comparable, so a genuine exchange uses a shared consortium seed for the
    indicators it intends to share, while each institution's internal tags stay
    under its own secret seed.
    """
    from biohash.flyhash import minhash_bands

    return [f"mh:{k}" for k in minhash_bands(tag, n_bands=n_bands, band_size=band_size)]


def publish_filter(
    indicators: list[str],
    institution_id: str,
    epoch_salt: str,
    target_fp: float = 0.001,
) -> IndicatorFilter:
    """Build the artifact an institution publishes to the network."""
    f = IndicatorFilter.sized_for(
        len(indicators), target_fp=target_fp, salt=epoch_salt, institution_id=institution_id
    )
    for ind in indicators:
        f.add(ind)
    return f


def query_filter(f: IndicatorFilter, candidates: list[str]) -> list[str]:
    """Which of our indicators the other institution also holds.

    Results are *probable* members: the filter's ``false_positive_rate`` is the
    chance any single hit is spurious, and a caller acting on a hit should treat
    it as one signal among several rather than as confirmation.
    """
    return [c for c in candidates if c in f]


#: Bands out of ``n_bands`` that must hit before two institutions are told they
#: share an indicator. Measured background is 0.06 hits with a maximum of 1, and
#: a genuine match scores at least 21, so this sits far from both.
MATCH_BAND_THRESHOLD = 6


def match_identity(
    f: IndicatorFilter, tag: SparseTag, n_bands: int = 32, band_size: int = 4
) -> tuple[int, bool]:
    """How many of this identity's bands the other institution also holds.

    This is the question the exchange actually answers -- *is this specific
    suspicious identity known to you* -- rather than "how many indicators do our
    books share in total", which saturates with population size and separates
    nothing.
    """
    bands = tag_indicators(tag, n_bands=n_bands, band_size=band_size)
    hits = sum(1 for b in bands if b in f)
    return hits, hits >= MATCH_BAND_THRESHOLD


def private_set_intersection(
    left: list[str], right: list[str], epoch_salt: str, target_fp: float = 0.001
) -> dict:
    """Two-party exchange returning only the shared indicators.

    Each side contributes salted digests; neither learns anything about the
    other's non-shared items beyond the filter's bounded error rate.
    """
    f = publish_filter(right, "right", epoch_salt, target_fp=target_fp)
    hits = query_filter(f, left)
    return {
        "n_left": len(left),
        "n_right": len(right),
        "n_shared": len(hits),
        "shared": hits,
        "filter_bytes": f.size_bytes,
        "filter_fp_rate": f.false_positive_rate,
        "bytes_per_item": f.size_bytes / max(1, len(right)),
    }


def _demo() -> None:
    """Two institutions discover a shared operator without exchanging customers."""
    import numpy as np

    from biohash.flyhash import TAG_DIM, FlyHash

    print("--- cross-institution indicator exchange ---")
    print()

    epoch = "2026-W35"
    # A consortium seed for shared indicators. Each institution's *internal*
    # tags stay under its own secret seed and are never comparable to anyone's.
    consortium = FlyHash(input_dim=99, dim=TAG_DIM, seed_id="consortium")
    rng = np.random.default_rng(3)

    # One operator's face, used to open accounts at both institutions under
    # different names. Neither bank can see this on its own.
    operator_face = rng.normal(size=99)
    a_ring = [consortium.tag(operator_face + rng.normal(0, 0.08, 99)) for _ in range(3)]
    a_rest = [consortium.tag(rng.normal(size=99)) for _ in range(400)]
    b_ring = [consortium.tag(operator_face + rng.normal(0, 0.08, 99)) for _ in range(2)]
    b_rest = [consortium.tag(rng.normal(size=99)) for _ in range(400)]

    b_indicators = [i for t in b_ring + b_rest for i in tag_indicators(t)]
    f = publish_filter(b_indicators, "bank_b", epoch)

    print(f"  bank_b publishes    {len(b_ring)+len(b_rest):,} identities as "
          f"{len(b_indicators):,} indicators")
    print(f"  filter size         {f.size_bytes:,} bytes "
          f"({f.size_bytes/max(1,len(b_ring)+len(b_rest)):.0f} bytes per identity)")
    print(f"  filter fp rate      {f.false_positive_rate:.5f}")
    print()

    ring_hits = [match_identity(f, t)[0] for t in a_ring]
    rest_hits = [match_identity(f, t)[0] for t in a_rest]
    flagged = sum(1 for t in a_rest if match_identity(f, t)[1])

    print(f"  bank_a queries its {len(a_ring)+len(a_rest):,} identities against that filter:")
    print(f"    ring identities     {ring_hits}  bands hit (of 32)")
    print(f"    everyone else       mean {np.mean(rest_hits):.2f}, max {max(rest_hits)}")
    print(f"    threshold           {MATCH_BAND_THRESHOLD}")
    print(f"    -> matched          {sum(1 for h in ring_hits if h >= MATCH_BAND_THRESHOLD)}"
          f"/{len(a_ring)} ring identities, {flagged} false positives "
          f"of {len(a_rest):,}")

    print("  what crossed the wire: a bit array. No names, no tokens, no tags,")
    print("  no images, and no way to enumerate bank_b's customers from it.")
    print()

    # Unlinkability still holds for each institution's internal tags.
    a_secret = FlyHash(input_dim=99, dim=TAG_DIM, seed_id="bank_a", salt="internal")
    b_secret = FlyHash(input_dim=99, dim=TAG_DIM, seed_id="bank_b", salt="internal")
    ta, tb = a_secret.tag(operator_face), b_secret.tag(operator_face)
    raw = len(set(ta.indices) & set(tb.indices)) / len(set(ta.indices) | set(tb.indices))
    print(f"  same face under each bank's SECRET seed: raw overlap {raw:.4f} "
          f"(chance ~0.026)")
    print("  -> internal tags stay mutually unlinkable. Only consortium-seeded")
    print("     indicators are shareable, and only as filter bits.")


if __name__ == "__main__":
    _demo()
