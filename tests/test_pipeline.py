"""End-to-end tests over the whole detection path, on the ``tiny`` scenario.

The unit tests in ``test_biohash.py`` lock in claims. These lock in *seams* --
the places where one module hands something to another and nobody notices for a
day that the handoff is broken. Every test here corresponds to a failure that
actually happened:

* A model pickled while training ran as ``__main__`` could not be loaded by the
  scorer or the report, so everything downstream of training died at import
  time with an ``AttributeError``.
* ``TrainedModel.predict`` returned ``LGBMClassifier.predict`` -- hard 0/1
  labels -- while every threshold was calibrated on ``predict_proba``. The
  report looked fine because it called ``predict_proba`` directly; the serving
  path silently collapsed to two score values and the graduated bands could
  never fire.
* Retro-propagation pushed evidence paths into a heap without a tiebreaker, so
  the moment two entries tied on score it compared two dicts and raised.

None of these is subtle once seen, and none was visible from any single
module's own tests.

Run: ``.venv/Scripts/python.exe tests/test_pipeline.py``
     or ``.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v``
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from contracts.decisions import Action, Band, ViewScope
from detect import ingest

DATA = Path(__file__).resolve().parents[1] / "data" / "tiny"

_cache: dict = {}


def dataset():
    if "ds" not in _cache:
        _cache["ds"] = ingest.load(DATA, view=ViewScope.NETWORK)
    return _cache["ds"]


def behaviour_model():
    """Train once and reuse: the tiny scenario is ~6k events, so this is cheap."""
    if "beh" not in _cache:
        from detect.models.train import train_behaviour

        ds = dataset()
        _cache["beh"] = train_behaviour(ds.auth, ds.labels, ds.ground_truth)
    return _cache["beh"]


# --------------------------------------------------------------------------
# L1 ingest
# --------------------------------------------------------------------------


def test_scenario_loads_all_four_streams():
    ds = dataset()
    assert ds.onboarding and ds.auth and ds.telemetry and ds.labels
    assert ds.ground_truth


def test_view_scoping_is_a_real_restriction():
    """A scoped view must genuinely hold less, not merely display less."""
    net = dataset()
    inst = ingest.institutions_in(DATA)[0]
    scoped = ingest.load(DATA, view=ViewScope.MERCHANT, institution_id=inst)
    assert 0 < len(scoped.auth) < len(net.auth)
    assert all(e.institution_id == inst for e in scoped.auth)


def test_scoped_view_requires_an_institution():
    try:
        ingest.load(DATA, view=ViewScope.MERCHANT)
    except ValueError:
        return
    raise AssertionError("a merchant view without an institution must not be loadable")


def test_vault_refuses_detokenisation():
    from detect.ingest import PIIVault

    try:
        PIIVault().resolve("tok_whatever")
    except PermissionError:
        return
    raise AssertionError("the detection path must not be able to resolve a token")


# --------------------------------------------------------------------------
# L4 models: fit, persist, reload, serve
# --------------------------------------------------------------------------


def test_model_survives_a_save_and_load_from_another_module():
    """The bug: pickling ``self`` recorded the class as ``__main__.TrainedModel``.

    Training runs as ``python -m detect.models.train``, so the class was named
    in ``__main__``; the scorer and the report resolve that name to *their*
    ``__main__`` and the load failed. Saving a dict payload removes the class
    reference entirely.
    """
    from detect.models.train import TrainedModel

    beh = behaviour_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = beh.save(Path(tmp))
        reloaded = TrainedModel.load(path)

    assert reloaded.feature_names == beh.feature_names
    rows = [{n: 0.5 for n in beh.feature_names}]
    assert np.allclose(reloaded.predict(rows), beh.predict(rows))


def test_predict_returns_probabilities_not_labels():
    """Hard labels would collapse every band to allow-or-block."""
    beh = behaviour_model()
    ds = dataset()
    from detect.models.train import build_behaviour_dataset

    rows, _, _, _ = build_behaviour_dataset(ds.auth, [], cutoffs=[])
    scores = beh.predict(rows)

    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # Rounding would hide the failure this test exists for: a well-separated
    # model puts most legitimate traffic at ~1e-7, and rounding to six places
    # collapses all of it to zero. Compare the raw values.
    assert np.unique(scores).size > 2, "scores are not continuous"
    assert not set(np.unique(scores)).issubset({0.0, 1.0}), "predict returned hard labels"


def test_training_refuses_eval_only_features():
    from detect.models.train import LeakageError, assert_no_eval_fields

    try:
        assert_no_eval_fields(["decline_ratio_1h", "is_synthetic"])
    except LeakageError:
        return
    raise AssertionError("ground-truth fields must never be accepted as features")


def test_training_refuses_labels_from_the_future():
    from datetime import timedelta

    from detect.models.train import LeakageError, assert_labels_available

    ds = dataset()
    labels = sorted(ds.labels, key=lambda lb: lb.label_available_at)
    cutoff = labels[0].label_available_at - timedelta(days=1)
    try:
        assert_labels_available(labels[:5], cutoff)
    except LeakageError:
        return
    raise AssertionError("a label must not be usable before it arrived")


# --------------------------------------------------------------------------
# L2/L5 graph and retro-propagation
# --------------------------------------------------------------------------


def graph_and_communities():
    if "graph" not in _cache:
        from detect.graph.build import build
        from detect.graph.communities import detect

        ds = dataset()
        ig = build(ds.onboarding, ds.auth, ds.telemetry)
        _cache["graph"] = (ig, detect(ig))
    return _cache["graph"]


def test_graph_builds_and_finds_communities():
    ig, communities = graph_and_communities()
    assert ig.g.number_of_nodes() > 0
    assert communities, "no candidate rings proposed"


def test_propagation_does_not_trip_over_tied_paths():
    """The heap bug: equal scores fell through to comparing two dicts."""
    from detect.fusion import mark_dormant, propagate

    ig, _ = graph_and_communities()
    ds = dataset()
    ring_members = [t for t in ds.ground_truth if t.is_synthetic and t.ring_id]
    assert ring_members, "the tiny scenario should contain planted rings"

    seed = ring_members[0].identity_id
    prop = propagate(ig, {seed: 1.0})
    transacted = {e.identity_id for e in ds.auth}
    prop = mark_dormant(prop, transacted)

    assert prop, "one confirmation reached nobody at all"
    assert seed not in prop, "a seed must not be reported as its own propagation"
    for p in prop.values():
        assert 0.0 < p.propagated_score <= 1.0
        assert p.hops >= 1


def test_propagation_reaches_ring_siblings_that_never_transacted():
    """The differentiating claim, on the smallest scenario that can carry it."""
    from collections import defaultdict

    from detect.fusion import mark_dormant, propagate

    ig, _ = graph_and_communities()
    ds = dataset()
    transacted = {e.identity_id for e in ds.auth}

    by_ring: dict[str, list] = defaultdict(list)
    for t in ds.ground_truth:
        if t.is_synthetic and t.ring_id:
            by_ring[t.ring_id].append(t)

    reached_any = False
    for members in by_ring.values():
        acted = [m for m in members if m.first_fraud_ts is not None]
        dormant = {m.identity_id for m in members if m.first_fraud_ts is None}
        if not acted or not dormant:
            continue
        prop = mark_dormant(propagate(ig, {acted[0].identity_id: 1.0}), transacted)
        if dormant & set(prop):
            reached_any = True
    assert reached_any, "no dormant ring member was reached from a confirmed sibling"


def test_propagation_does_not_flood_the_population():
    """Recall without precision is not a result.

    Diffusion that reaches most of the population scores a ring recall of 1.000
    and means nothing. This is the test that would have caught it: linking
    identities on a *common* email-handle shape produced 1,078 links at zero
    same-ring precision on the sloppy scenario, and one confirmation then lit up
    73% of the identities in the tiny one.
    """
    from detect.fusion import propagate

    ig, _ = graph_and_communities()
    ds = dataset()
    truth = {t.identity_id: t for t in ds.ground_truth}
    synthetic = {i for i, t in truth.items() if t.is_synthetic}

    seed = next(i for i in synthetic if truth[i].first_fraud_ts is not None)
    flagged = set(propagate(ig, {seed: 1.0})) - {seed}
    assert flagged, "one confirmation reached nobody"

    share = len(flagged) / len(truth)
    precision = len(flagged & synthetic) / len(flagged)
    assert share < 0.25, f"propagation flagged {share:.1%} of all identities"
    assert precision > 0.5, f"only {precision:.1%} of flagged identities are synthetic"


def test_entity_resolution_links_are_mostly_same_ring():
    """Every link kind is evidence; a kind at chance precision is noise."""
    from collections import defaultdict

    from detect.graph.entity_res import resolve

    ds = dataset()
    truth = {t.identity_id: t for t in ds.ground_truth}
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for link in resolve(ds.onboarding):
        a, b = truth.get(link.left), truth.get(link.right)
        same_ring = bool(
            a and b and a.is_synthetic and b.is_synthetic and a.ring_id and a.ring_id == b.ring_id
        )
        kind = link.kind.split(":")[0]
        by_kind[kind][1] += 1
        by_kind[kind][0] += int(same_ring)

    weak = {
        kind: good / total
        for kind, (good, total) in by_kind.items()
        # A handful of links is not a measurement; judge kinds that carry weight.
        if total >= 20 and good / total < 0.2
    }
    assert not weak, f"link kinds at or near chance precision: {weak}"


def test_forward_fusion_only_ever_raises_a_score():
    """Onboarding conditions behaviour upward; it must never mask bad behaviour."""
    from detect.fusion import fuse_forward

    for behaviour in (0.0, 0.2, 0.9):
        assert fuse_forward(behaviour, 0.0) == behaviour
        assert fuse_forward(behaviour, 1.0) >= behaviour
        assert fuse_forward(behaviour, 1.0) <= 1.0


# --------------------------------------------------------------------------
# L6 decision engine
# --------------------------------------------------------------------------


def test_bands_are_monotonic_and_every_action_is_reachable():
    from detect.decision import DecisionEngine

    engine = DecisionEngine()
    seen: set[Action] = set()
    previous = -1.0
    for score in np.linspace(0.0, 1.0, 101):
        resp = engine.decide(
            event_id=f"ev_{score:.2f}",
            score=float(score),
            features={},
            onboarding_score=0.0,
        )
        assert resp.score >= previous - 1e-9
        previous = resp.score
        seen.add(resp.action)

    assert Action.ALLOW in seen
    assert seen - {Action.ALLOW}, "no score in [0,1] produced an intervention"


def test_every_reason_code_has_human_readable_text():
    from contracts.decisions import REASON_TEXT, ReasonCode

    missing = [c for c in ReasonCode if c not in REASON_TEXT]
    assert not missing, f"reason codes with no explanation: {missing}"


def test_blocks_always_carry_a_reason():
    """A block a human cannot explain is a block a regulator will not accept."""
    from detect.decision import DecisionEngine

    engine = DecisionEngine()
    features = {
        "decline_ratio_1h": 0.95,
        "distinct_pans_1h": 40.0,
        "pan_entropy_1h": 0.05,
        "low_ticket_ratio_1h": 0.98,
        "cvv_mismatch_rate_1h": 0.8,
    }
    resp = engine.decide(
        event_id="ev_block",
        score=0.99,
        features=features,
        onboarding_score=0.9,
    )
    assert resp.band in (Band.HIGH, Band.CONFIRMED, Band.ELEVATED)
    assert resp.reason_codes, "an intervention was returned with no reason codes"
    assert all(isinstance(t, str) and t for t in resp.explain())


def test_honeypot_response_is_plausible_and_uninformative():
    """Same shape for every card, deterministic per card, ~5% approvals.

    An endpoint that declines everything tells the attacker as much as one that
    approves everything; the point is that the answer carries no signal about
    whether the card is live.
    """
    from detect.decision import honeypot_response

    a = honeypot_response("tok_card_a")
    assert set(a) == set(honeypot_response("tok_card_b"))
    assert a == honeypot_response("tok_card_a"), "probing the same card twice must agree"

    codes = [honeypot_response(f"tok_{i}")["response_code"] for i in range(2000)]
    approvals = sum(c == "approved" for c in codes) / len(codes)
    assert 0.01 < approvals < 0.12, f"implausible approval rate {approvals:.3f}"


# --------------------------------------------------------------------------
# Analyst copilot (off the auth path)
# --------------------------------------------------------------------------


def test_narrative_falls_back_to_the_template_without_a_key():
    """The console must not depend on a network call or a configured secret."""
    from detect.copilot import CaseEvidence, narrate

    ev = CaseEvidence(
        ring_id="cmt_0001",
        ring_size=12,
        cohesion=0.8,
        suspicion=0.6,
        institutions=["inst_00", "inst_01"],
        cross_institution=True,
        shared_attributes={"device": 9},
        seed_identity="idn_0000001",
        n_flagged=10,
        n_dormant_flagged=4,
    )
    result = narrate(ev, use_model=False)

    assert result["source"] == "template"
    assert result["model"] is None
    assert "cmt_0001" in result["narrative"]
    # The numbers in the prose come from the evidence, never from the model.
    assert "4" in result["narrative"] and "10" in result["narrative"]


def test_narrative_says_so_when_nothing_has_been_confirmed():
    """Structure alone is not evidence of fraud, and the case must say that."""
    from detect.copilot import CaseEvidence, template_narrative

    text = template_narrative(
        CaseEvidence(
            ring_id="cmt_0002",
            ring_size=6,
            cohesion=0.4,
            suspicion=0.2,
            institutions=["inst_00"],
            cross_institution=False,
        )
    )
    assert "No account in this community has been confirmed" in text
    assert "Recommended next step:" in text


def test_case_evidence_carries_no_pii_fields():
    """Only tokens, counts and scores may reach the model."""
    from dataclasses import fields

    from detect.copilot import CaseEvidence

    forbidden = {"name", "dob", "address", "email", "phone", "pan", "card_number", "ssn"}
    names = {f.name for f in fields(CaseEvidence)}
    assert not (names & forbidden), f"PII-shaped fields on the evidence: {names & forbidden}"


def test_evidence_assembles_from_a_real_community():
    from detect.copilot import evidence_for_ring
    from detect.fusion import mark_dormant, propagate

    ig, communities = graph_and_communities()
    ds = dataset()
    community = communities[0]
    transacted = {e.identity_id for e in ds.auth}
    seed = next((i for i in community.identity_ids if i in transacted), None)
    prop = mark_dormant(propagate(ig, {seed: 1.0}), transacted) if seed else {}

    ev = evidence_for_ring(community, ig, prop)
    assert ev.ring_id == community.community_id
    assert ev.ring_size == community.size
    assert ev.shared_attributes, "a community with no shared attributes is not a community"


# --------------------------------------------------------------------------
# The scoring service, in-process
# --------------------------------------------------------------------------


def test_scorer_serves_a_decision_for_a_real_event():
    """Exercises the serving path the gateway calls, without a network hop."""
    import os

    os.environ.setdefault("FRAUD_DATA", str(DATA))
    from detect.decision import DecisionEngine
    from detect.features.stream import StreamFeatureStore

    ds = dataset()
    beh = behaviour_model()
    store = StreamFeatureStore()
    engine = DecisionEngine()

    scores = []
    for ev in sorted(ds.auth, key=lambda e: e.ts)[:800]:
        features = store.features(ev)
        score = float(beh.predict([features])[0])
        resp = engine.decide(
            event_id=ev.event_id,
            score=score,
            features=features,
            onboarding_score=0.0,
        )
        store.update(ev)
        scores.append(resp.score)
        assert resp.event_id == ev.event_id
        assert 0.0 <= resp.score <= 1.0

    assert np.unique(scores).size > 2, "serving path is not scoring continuously"


# --------------------------------------------------------------------------


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- a plain report is the point
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  PASS  {name}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
