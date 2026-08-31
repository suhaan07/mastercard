"""Property tests for the biometric hashing layer.

These lock in the claims the pitch makes. If one of these fails, a slide is
wrong, not just a test.

Run: ``.venv/Scripts/python.exe -m pytest tests/test_biohash.py -v``
     or ``.venv/Scripts/python.exe tests/test_biohash.py`` for a plain report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from biohash.doctemplate import DocTemplate, DocTemplateHasher
from biohash.flyhash import FlyHash, chance_overlap, minhash_bands
from biohash.hdc import bind, bundle, categorical_tag, identity_hypervector, unbind
from biohash.images import (
    FEATURE_DIM,
    DescriptorNormalizer,
    ProceduralFaceSource,
    image_features,
)

#: Overlap above which two tags are treated as near-duplicates. Chosen from the
#: measured ROC: ~95% same-ring recall at a ~0.7% false-link rate.
NEAR_DUPLICATE_THRESHOLD = 0.30


def _fitted_pipeline(seed: int = 42):
    src = ProceduralFaceSource(seed=seed)
    fly = FlyHash(input_dim=FEATURE_DIM, seed_id="bank_a")
    ref = np.array([image_features(src.sample(synthetic=bool(i % 2))) for i in range(300)])
    return src, fly, DescriptorNormalizer().fit(ref)


# --------------------------------------------------------------------------
# FlyHash core properties
# --------------------------------------------------------------------------


def test_similar_inputs_overlap_far_above_chance():
    fly = FlyHash(input_dim=256, seed_id="bank_a")
    rng = np.random.default_rng(0)
    base = rng.normal(size=256)
    t0 = fly.tag(base)
    near = fly.tag(base + rng.normal(scale=0.15, size=256))
    assert near.overlap(t0) > 0.5


def test_unrelated_inputs_overlap_at_chance():
    fly = FlyHash(input_dim=256, seed_id="bank_a")
    rng = np.random.default_rng(1)
    t0 = fly.tag(rng.normal(size=256))
    overlaps = [fly.tag(rng.normal(size=256)).overlap(t0) for _ in range(200)]
    expected = chance_overlap(fly.dim, fly.hash_length)
    assert abs(float(np.mean(overlaps)) - expected) < 0.01


def test_tags_are_unlinkable_across_institution_seeds():
    """The load-bearing privacy claim: the same face at two institutions does
    not produce comparable tags -- and not merely because a guard clause says
    so. The raw index sets themselves overlap only at chance."""
    rng = np.random.default_rng(2)
    face = rng.normal(size=256)
    ta = FlyHash(input_dim=256, seed_id="bank_a").tag(face)
    tb = FlyHash(input_dim=256, seed_id="bank_b").tag(face)

    assert ta.overlap(tb) == 0.0  # enforced by the contract

    raw = len(set(ta.indices) & set(tb.indices)) / len(set(ta.indices) | set(tb.indices))
    assert raw < 2 * chance_overlap(ta.dim, len(ta.indices))


def test_reseeding_revokes_tags():
    """Rotating the salt invalidates every tag issued under the old one."""
    rng = np.random.default_rng(3)
    face = rng.normal(size=256)
    old = FlyHash(input_dim=256, seed_id="bank_a", salt="2026-Q1").tag(face)
    new = FlyHash(input_dim=256, seed_id="bank_a", salt="2026-Q2").tag(face)
    raw = len(set(old.indices) & set(new.indices)) / len(set(old.indices) | set(new.indices))
    assert raw < 2 * chance_overlap(old.dim, len(old.indices))


def test_hashing_is_deterministic_across_instances():
    rng = np.random.default_rng(4)
    x = rng.normal(size=256)
    a = FlyHash(input_dim=256, seed_id="bank_a").tag(x)
    b = FlyHash(input_dim=256, seed_id="bank_a").tag(x)
    assert a.indices == b.indices


def test_batch_matches_single():
    fly = FlyHash(input_dim=128, seed_id="bank_a")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(8, 128))
    assert [t.indices for t in fly.tag_batch(X)] == [fly.tag(x).indices for x in X]


def test_tag_is_invariant_to_input_scaling():
    """Divisive normalisation: a brighter photo of the same face must hash alike."""
    fly = FlyHash(input_dim=256, seed_id="bank_a")
    x = np.random.default_rng(6).normal(size=256)
    assert fly.tag(x).indices == fly.tag(x * 3.7).indices


def test_minhash_bands_collide_for_near_duplicates():
    """LSH blocking must actually propose near-duplicates as candidates.

    Banded MinHash, not the earlier ``cluster_key``: that one required a run of
    sorted indices to match exactly, so it fired on 0.5% of same-ring pairs.
    A band match is probabilistic in the right direction -- likely for an
    overlapping pair, vanishingly unlikely for an unrelated one.
    """
    fly = FlyHash(input_dim=256, seed_id="bank_a")
    rng = np.random.default_rng(7)
    base = rng.normal(size=256)
    a, b = fly.tag(base), fly.tag(base + rng.normal(scale=0.05, size=256))
    far = fly.tag(rng.normal(size=256))
    assert set(minhash_bands(a)) & set(minhash_bands(b))
    assert not (set(minhash_bands(a)) & set(minhash_bands(far)))


# --------------------------------------------------------------------------
# Hyperdimensional binding
# --------------------------------------------------------------------------


def test_bind_is_bijective():
    fly = FlyHash(input_dim=128, seed_id="bank_a")
    t = fly.tag(np.random.default_rng(8).normal(size=128))
    assert unbind(bind(t, "face"), "face").indices == t.indices


def test_roles_are_separated():
    """A face must not be able to collide with someone's document template."""
    fly = FlyHash(input_dim=128, seed_id="bank_a")
    t = fly.tag(np.random.default_rng(9).normal(size=128))
    assert bind(t, "face").overlap(bind(t, "doc_template")) < 0.10


def test_bundle_rejects_mixed_seeds():
    fly_a = FlyHash(input_dim=128, seed_id="bank_a")
    fly_b = FlyHash(input_dim=128, seed_id="bank_b")
    x = np.random.default_rng(10).normal(size=128)
    try:
        bundle([fly_a.tag(x), fly_b.tag(x)])
    except ValueError:
        return
    raise AssertionError("bundling across institution seeds must be refused")


def test_ring_members_separate_from_legitimate_population():
    """Shared infrastructure lifts hypervector overlap well clear of the
    legitimate population -- this is what makes ring detection possible."""
    fly = FlyHash(input_dim=128, seed_id="bank_a")
    rng = np.random.default_rng(11)
    dim, hl = fly.dim, fly.hash_length

    base_face = rng.normal(size=128)
    shared_doc = fly.tag(rng.normal(size=128))
    shared_addr = categorical_tag("12 Mill Lane Apt 4", dim, hl, "bank_a")

    ring = [
        identity_hypervector(
            face_tag=fly.tag(base_face + rng.normal(scale=0.35, size=128)),
            doc_template_tag=shared_doc,
            address_tag=shared_addr,
        )
        for _ in range(30)
    ]
    legit = [
        identity_hypervector(
            face_tag=fly.tag(rng.normal(size=128)),
            doc_template_tag=fly.tag(rng.normal(size=128)),
            address_tag=categorical_tag(f"addr_{i}", dim, hl, "bank_a"),
        )
        for i in range(120)
    ]

    within = [ring[i].overlap(ring[j]) for i in range(30) for j in range(i + 1, 30)]
    between = [legit[i].overlap(legit[j]) for i in range(0, 120, 5) for j in range(i + 1, 120, 7)]
    assert min(within) > max(between)


# --------------------------------------------------------------------------
# End-to-end face pipeline
# --------------------------------------------------------------------------


def test_face_linker_separates_ring_from_population():
    """The operating point the demo relies on: high same-ring recall at a low
    false-link rate. Numbers here are the ones quoted in the metrics report."""
    src, fly, norm = _fitted_pipeline()

    def tag_of(vec):
        return fly.tag(norm.transform(image_features(src.sample(synthetic=True, identity_vec=vec))))

    rings = []
    for _ in range(10):
        base = src.random_identity_vec()
        rings.append([tag_of(src.near_duplicate_vec(base)) for _ in range(6)])
    others = [tag_of(src.random_identity_vec()) for _ in range(120)]

    same = [
        rings[r][i].overlap(rings[r][j])
        for r in range(10)
        for i in range(6)
        for j in range(i + 1, 6)
    ]
    diff = [others[i].overlap(others[j]) for i in range(0, 120, 3) for j in range(i + 1, 120, 5)]

    recall = float(np.mean([s >= NEAR_DUPLICATE_THRESHOLD for s in same]))
    false_link = float(np.mean([d >= NEAR_DUPLICATE_THRESHOLD for d in diff]))
    assert recall > 0.85, f"same-ring recall {recall:.3f} too low"
    assert false_link < 0.02, f"false-link rate {false_link:.4f} too high"


def test_face_descriptor_ignores_global_colour():
    """Regression guard. A low-frequency DCT descriptor was dominated by skin
    and hair, so unrelated faces overlapped at 0.26 against a 0.026 floor. The
    high-pass step fixed it and must stay."""
    src, fly, norm = _fitted_pipeline(seed=77)
    tags = [
        fly.tag(norm.transform(image_features(src.sample(synthetic=True))))
        for _ in range(60)
    ]
    pairs = [tags[i].overlap(tags[j]) for i in range(60) for j in range(i + 1, 60)]
    assert float(np.mean(pairs)) < 0.12


# --------------------------------------------------------------------------
# Document templates
# --------------------------------------------------------------------------


def test_shared_document_template_is_detectable():
    hasher = DocTemplateHasher(seed_id="bank_a")
    rng = np.random.default_rng(21)
    shared = DocTemplate.generate("tpl_shared", rng)

    same = [hasher.tag_template(shared, rng) for _ in range(12)]
    others = [
        hasher.tag_template(DocTemplate.generate(f"tpl_{i}", rng), rng) for i in range(40)
    ]

    within = [same[i].overlap(same[j]) for i in range(12) for j in range(i + 1, 12)]
    across = [s.overlap(o) for s in same for o in others]
    assert min(within) > max(across)


def test_document_tag_survives_render_jitter():
    hasher = DocTemplateHasher(seed_id="bank_a")
    rng = np.random.default_rng(22)
    tpl = DocTemplate.generate("tpl_x", rng)
    a = hasher.tag_template(tpl, rng, jitter=0.0)
    b = hasher.tag_template(tpl, rng, jitter=0.010)
    assert a.overlap(b) > 0.5


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)
