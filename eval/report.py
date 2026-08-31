"""The metrics report. Every headline number at a fixed false-positive rate.

The design doc is explicit that the defender's real cost function is false
declines, so a detection rate quoted without an FPR is not a result. Everything
here is reported at 0.1% FPR by default.

The numbers that differentiate this project are the middle three:

* **Detection rate at fixed FPR** -- the headline.
* **Attempts-to-detection** -- how many card tests get through before the block.
  Lower is the entire point; catching a ring on its four-hundredth attempt is
  not catching it.
* **Ring recall before first transaction** -- of a confirmed ring's accounts,
  what fraction did we flag *before* they did anything at all?
* **False-decline rate on look-alikes** -- the small business, the traveller,
  the customer on a new phone. This is the number that keeps the others honest.
* **Thin-file fairness check** -- thin-file customers are disproportionately
  young, migrant or low-income and are legitimately thin-file. If their decline
  rate diverges from everyone else's, the model has learned a proxy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from contracts.decisions import ViewScope

TARGET_FPR = 0.001


@dataclass
class Report:
    scenario: str
    target_fpr: float
    detection_rate: float = 0.0
    precision: float = 0.0
    auc: float = float("nan")
    realised_fpr: float = 0.0
    attempts_to_detection: dict = field(default_factory=dict)
    ring_recall_before_transacting: dict = field(default_factory=dict)
    propagation_precision: dict = field(default_factory=dict)
    lookalike_false_decline: dict = field(default_factory=dict)
    thin_file_false_decline: dict = field(default_factory=dict)
    view_delta: dict = field(default_factory=dict)
    holdout: dict = field(default_factory=dict)
    onboarding: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _split_ts(model):
    """The train/test boundary the model itself recorded, if it has one."""
    from datetime import datetime

    raw = (model.metrics or {}).get("split_ts")
    if not raw:
        return None
    return raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))


def attempts_to_detection(auth, scores, y_true, threshold) -> dict:
    """How many fraudulent attempts each account lands before we first fire."""
    counts: dict[str, int] = defaultdict(int)
    caught_at: dict[str, int] = {}
    for ev, sc, y in zip(auth, scores, y_true):
        if not y:
            continue
        acct = ev.account_id
        counts[acct] += 1
        if acct not in caught_at and sc >= threshold:
            caught_at[acct] = counts[acct]

    vals = np.array(sorted(caught_at.values())) if caught_at else np.array([])
    return {
        "accounts_with_fraud": len(counts),
        "accounts_caught": len(caught_at),
        "accounts_never_caught": len(counts) - len(caught_at),
        "median_attempts": float(np.median(vals)) if vals.size else None,
        "p90_attempts": float(np.percentile(vals, 90)) if vals.size else None,
        "mean_attempts": float(vals.mean()) if vals.size else None,
    }


def ring_recall_before_transacting(propagated: dict, truth: list) -> dict:
    """The differentiating number: ring members flagged before they ever acted.

    Only accounts with no fraud event at all are counted. An account we caught
    on its first attempt was still caught *after* acting, and folding those in
    would quietly inflate the number this whole project rests on.
    """
    by_ring: dict[str, list] = defaultdict(list)
    for t in truth:
        if t.is_synthetic and t.ring_id:
            by_ring[t.ring_id].append(t)

    out: dict = {}
    total_dormant = total_flagged = 0
    for ring_id, members in sorted(by_ring.items()):
        dormant = [m for m in members if m.first_fraud_ts is None]
        flagged = [m for m in dormant if m.identity_id in propagated]
        total_dormant += len(dormant)
        total_flagged += len(flagged)
        out[ring_id] = {
            "size": len(members),
            "never_transacted": len(dormant),
            "flagged_before_transacting": len(flagged),
            "recall": len(flagged) / max(1, len(dormant)),
        }
    out["_overall"] = {
        "never_transacted": total_dormant,
        "flagged_before_transacting": total_flagged,
        "recall": total_flagged / max(1, total_dormant),
    }
    return out


def view_delta_by(
    auth,
    truth,
    model,
    target_fpr: float,
    net_scores,
    net_y,
    build_dataset,
    ground_truth_fn,
    to_matrix_fn,
    threshold_fn,
    key: str,
    min_events: int = 500,
    max_groups: int = 40,
    split_ts=None,
) -> dict:
    """The "why Mastercard" number: what one player provably cannot see.

    Card testing is sprayed across merchants and issuers on purpose, so each
    player sees a handful of attempts and nothing looks alarming. The delta is
    measured, not asserted: the *same* trained model scores the *same* events
    twice, and the only thing that changes is which events were available when
    the rolling windows were built. A scoped view rebuilds its windows from its
    own traffic alone, so an account that made forty attempts across eight
    merchants presents as five attempts at each.

    **Both sides are thresholded once, on the pooled result.** Giving each
    merchant its own threshold looked more faithful -- a merchant does calibrate
    on its own traffic -- but at 0.1% FPR a 2,500-event merchant has a budget of
    two and a half false positives, so the threshold is fit to two or three
    points and the comparison measures sampling noise. It produced a *negative*
    delta on the drift scenario, i.e. the merchant beating the network, which is
    not a believable result and was the tell. Pooling puts the same number of
    negatives behind both thresholds, leaving feature construction as the only
    difference between them.

    ``key`` selects the unit of blindness -- ``institution_id`` for an issuer's
    or acquirer's books, ``merchant_id`` for one storefront. The second is the
    larger number: ``merchant_spread`` is the knob the attacker turns, while
    every one of an account's attempts still lands on its own institution.
    """
    groups: dict[str, list] = defaultdict(list)
    for ev in auth:
        groups[getattr(ev, key)].append(ev)

    net_by_event = {ev.event_id: (sc, yv) for ev, sc, yv in zip(auth, net_scores, net_y)}
    fraud_ids = {ev.event_id for ev, yv in zip(auth, net_y) if yv == 1}

    # Largest fraud exposure first: a group with no fraud has no detection rate
    # to contribute, and the cap keeps the pass proportional to the point.
    ranked = sorted(
        (g for g, evs in groups.items() if len(evs) >= min_events),
        key=lambda g: -sum(1 for ev in groups[g] if ev.event_id in fraud_ids),
    )[:max_groups]

    pooled_scores: list[float] = []
    pooled_y: list[int] = []
    pooled_net_scores: list[float] = []
    per_group: dict[str, dict] = {}

    for g in ranked:
        # Windows are built over everything the group saw; only the holdout is
        # scored, matching how the network side is measured.
        ordered = sorted(groups[g], key=lambda e: e.ts)
        rows, _, _, _ = build_dataset(ordered, [], cutoffs=[])
        scores = np.asarray(
            model.booster.predict_proba(to_matrix_fn(rows, model.feature_names))[:, 1]
        )
        keep = [
            i
            for i, ev in enumerate(ordered)
            if (split_ts is None or ev.ts > split_ts) and ev.event_id in net_by_event
        ]
        if not keep:
            continue
        kept = [ordered[i] for i in keep]
        y = ground_truth_fn(kept, truth)
        if y.sum() == 0:
            continue

        group_scores = [float(scores[i]) for i in keep]
        group_net = [float(net_by_event[ev.event_id][0]) for ev in kept]
        pooled_scores.extend(group_scores)
        pooled_y.extend(int(v) for v in y)
        pooled_net_scores.extend(group_net)
        per_group[g] = {"auth_events": len(kept), "fraud_events": int(y.sum())}

    if not pooled_y:
        return {"_overall": {"scoped_by": key, "groups_compared": 0}}

    y_arr = np.asarray(pooled_y)
    scoped_arr = np.asarray(pooled_scores)
    net_arr = np.asarray(pooled_net_scores)

    scoped_thr = threshold_fn(y_arr, scoped_arr, target_fpr)
    net_thr = threshold_fn(y_arr, net_arr, target_fpr)
    scoped_tp = int(((scoped_arr >= scoped_thr) & (y_arr == 1)).sum())
    net_tp = int(((net_arr >= net_thr) & (y_arr == 1)).sum())
    n_pos = int((y_arr == 1).sum())

    overall = {
        "scoped_by": key,
        "groups_compared": len(per_group),
        "auth_events": int(y_arr.size),
        "fraud_events": n_pos,
        "scoped_view_detection_rate": scoped_tp / max(1, n_pos),
        "network_view_detection_rate": net_tp / max(1, n_pos),
        "delta": (net_tp - scoped_tp) / max(1, n_pos),
        "scoped_view_realised_fpr": float(
            ((scoped_arr >= scoped_thr) & (y_arr == 0)).sum() / max(1, int((y_arr == 0).sum()))
        ),
        "network_view_realised_fpr": float(
            ((net_arr >= net_thr) & (y_arr == 0)).sum() / max(1, int((y_arr == 0).sum()))
        ),
    }
    return {"_overall": overall, **per_group}


def propagation_precision(propagated: dict, truth: list, seeds: dict) -> dict:
    """What fraction of everything propagation flagged is actually a ring member.

    Ring recall on its own is not a result. Diffusion that reaches most of the
    population scores a recall of 1.000 and means nothing, and a graph with a
    dense shared attribute -- a card token seen by many accounts, a public IP --
    will do exactly that if the conductance is set too loosely. This is the
    number that keeps the recall honest, and if it is low the recall should not
    be quoted at all.
    """
    synthetic = {t.identity_id for t in truth if t.is_synthetic}
    seeded_rings = {
        t.ring_id for t in truth if t.identity_id in seeds and t.ring_id
    }
    in_seeded_ring = {
        t.identity_id for t in truth if t.ring_id in seeded_rings and t.is_synthetic
    }

    flagged = set(propagated) - set(seeds)
    if not flagged:
        return {"flagged": 0}

    tp_synthetic = len(flagged & synthetic)
    tp_same_ring = len(flagged & in_seeded_ring)
    return {
        "identities_total": len(truth),
        "seeds": len(seeds),
        "flagged": len(flagged),
        "flagged_share_of_population": len(flagged) / max(1, len(truth)),
        "precision_synthetic": tp_synthetic / len(flagged),
        "precision_same_ring": tp_same_ring / len(flagged),
        "false_flags": len(flagged) - tp_synthetic,
    }


def ring_visibility(truth, onboarding) -> dict:
    """How much of a ring a single institution can even see.

    Independent of any model: rings that straddle institutions are invisible as
    rings to each one separately, which is the structural reason the delta above
    exists at all.
    """
    inst_of = {e.identity_id: e.institution_id for e in onboarding}
    by_ring: dict[str, list] = defaultdict(list)
    for t in truth:
        if t.is_synthetic and t.ring_id:
            by_ring[t.ring_id].append(t.identity_id)

    shares = []
    out: dict = {}
    for ring_id, members in sorted(by_ring.items()):
        counts: dict[str, int] = defaultdict(int)
        for m in members:
            inst = inst_of.get(m)
            if inst:
                counts[inst] += 1
        if not counts:
            continue
        largest = max(counts.values())
        share = largest / len(members)
        shares.append(share)
        out[ring_id] = {
            "size": len(members),
            "institutions_spanned": len(counts),
            "largest_single_institution_share": share,
        }
    out["_overall"] = {
        "rings": len(shares),
        "median_share_visible_to_one_institution": float(np.median(shares)) if shares else None,
    }
    return out


def main() -> None:
    from detect import ingest
    from detect.models.train import (
        TrainedModel,
        auth_ground_truth,
        build_behaviour_dataset,
        metrics_at_threshold,
        threshold_for_fpr,
        to_matrix,
    )

    ap = argparse.ArgumentParser(description="Metrics at a fixed FPR")
    ap.add_argument("--data", default="data/sloppy")
    ap.add_argument("--models", default="models")
    ap.add_argument("--fpr", type=float, default=TARGET_FPR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-rings", action="store_true", help="skip the graph pass")
    ap.add_argument(
        "--skip-views", action="store_true", help="skip the merchant-vs-network comparison"
    )
    args = ap.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.models) / data_dir.name
    rep = Report(scenario=data_dir.name, target_fpr=args.fpr)

    ds = ingest.load(data_dir, view=ViewScope.NETWORK)
    print(ds.summary())

    beh = TrainedModel.load(model_dir / "behaviour.pkl")
    onb = TrainedModel.load(model_dir / "onboarding.pkl")

    all_events = sorted(ds.auth, key=lambda e: e.ts)
    # Windows must be built over the whole stream -- an event's features depend
    # on everything before it -- but scoring is only honest after the split the
    # model was trained on. Scoring the training period too was inflating every
    # number here, and on the sophisticated scenario it did something worse:
    # in-sample rows saturate the model's probabilities at exactly 1.0, so
    # enough negatives tied at the top that no threshold could meet the FPR
    # budget and the reported detection rate collapsed to zero.
    all_rows, _, _, _ = build_behaviour_dataset(all_events, ds.labels, cutoffs=[])
    all_scores = np.asarray(
        beh.booster.predict_proba(to_matrix(all_rows, beh.feature_names))[:, 1]
    )
    split_ts = _split_ts(beh)
    keep = [i for i, ev in enumerate(all_events) if split_ts is None or ev.ts > split_ts]

    ordered = [all_events[i] for i in keep]
    rows = [all_rows[i] for i in keep]
    scores = all_scores[np.asarray(keep, dtype=int)]
    y = auth_ground_truth(ordered, ds.ground_truth)
    rep.holdout = {
        "split_ts": split_ts.isoformat() if split_ts else None,
        "events_scored": len(ordered),
        "events_total": len(all_events),
        "fraud_events": int(y.sum()),
    }
    print(
        f"scoring the out-of-time holdout: {len(ordered):,} of {len(all_events):,} events "
        f"after {split_ts}"
    )

    thr = threshold_for_fpr(y, scores, args.fpr)
    m = metrics_at_threshold(y, scores, thr)
    rep.detection_rate = m["detection_rate"]
    rep.precision = m["precision"]
    rep.auc = m["auc"]
    rep.realised_fpr = m["false_positive_rate"]
    rep.attempts_to_detection = attempts_to_detection(ordered, scores, y, thr)

    legit = y == 0

    # -- false declines on cohorts that resemble fraud but are not ---------
    lookalikes = {t.account_id for t in ds.ground_truth if t.is_lookalike}
    la_mask = np.array([ev.account_id in lookalikes for ev in ordered])
    rep.lookalike_false_decline = {
        "lookalike_events": int((la_mask & legit).sum()),
        "declined": int(((scores >= thr) & la_mask & legit).sum()),
        "rate": float(((scores >= thr) & la_mask & legit).sum() / max(1, (la_mask & legit).sum())),
        "baseline_legit_rate": float(
            ((scores >= thr) & ~la_mask & legit).sum() / max(1, (~la_mask & legit).sum())
        ),
    }

    # -- fairness: thin-file customers must not be proxied against ---------
    thin = {e.identity_id for e in ds.onboarding if e.credit_file_age_months < 12}
    thin_mask = np.array([ev.identity_id in thin for ev in ordered])
    rep.thin_file_false_decline = {
        "thin_file_legit_events": int((thin_mask & legit).sum()),
        "declined": int(((scores >= thr) & thin_mask & legit).sum()),
        "rate": float(
            ((scores >= thr) & thin_mask & legit).sum() / max(1, (thin_mask & legit).sum())
        ),
        "thick_file_rate": float(
            ((scores >= thr) & ~thin_mask & legit).sum() / max(1, (~thin_mask & legit).sum())
        ),
    }

    rep.onboarding = {
        k: onb.metrics.get(k)
        for k in ("auc", "detection_rate", "false_positive_rate", "precision")
    }

    # -- merchant view vs network view -------------------------------------
    if not args.skip_views:
        common = dict(
            # The full stream, so a scoped view can build its own windows; the
            # holdout filter is applied inside, against the same split.
            auth=all_events,
            truth=ds.ground_truth,
            model=beh,
            target_fpr=args.fpr,
            net_scores=all_scores,
            net_y=auth_ground_truth(all_events, ds.ground_truth),
            build_dataset=build_behaviour_dataset,
            ground_truth_fn=auth_ground_truth,
            to_matrix_fn=to_matrix,
            threshold_fn=threshold_for_fpr,
            split_ts=split_ts,
        )
        rep.view_delta = {
            # Two units of blindness, because they are not the same claim. An
            # institution sees all of its own customers' attempts; a merchant
            # sees a slice of everyone's.
            "by_institution": view_delta_by(**common, key="institution_id", min_events=2000),
            "by_merchant": view_delta_by(**common, key="merchant_id"),
            "ring_visibility": ring_visibility(ds.ground_truth, ds.onboarding),
        }

    # -- ring recall before transacting, via retro-propagation -------------
    if not args.skip_rings:
        from detect.fusion import mark_dormant, propagate
        from detect.graph.build import build
        from detect.graph.communities import detect

        ig = build(ds.onboarding, ds.auth, ds.telemetry)
        communities = detect(ig)
        transacted = {e.identity_id for e in ds.auth}

        # Seed one confirmed account per planted ring: the analyst confirms one
        # case, and we measure how much of the rest of the ring that reaches.
        seeds: dict[str, float] = {}
        by_ring: dict[str, list] = defaultdict(list)
        for t in ds.ground_truth:
            if t.is_synthetic and t.ring_id and t.first_fraud_ts is not None:
                by_ring[t.ring_id].append(t)
        for ring_id, members in by_ring.items():
            first = sorted(members, key=lambda t: t.first_fraud_ts)[0]
            seeds[first.identity_id] = 1.0

        prop = mark_dormant(propagate(ig, seeds), transacted)
        rep.ring_recall_before_transacting = ring_recall_before_transacting(
            prop, ds.ground_truth
        )
        rep.propagation_precision = propagation_precision(prop, ds.ground_truth, seeds)
        rep.notes.append(
            f"retro-propagation seeded with {len(seeds)} confirmed accounts "
            f"(one per ring); {len(communities)} communities detected"
        )

    # -- print --------------------------------------------------------------
    print()
    print(f"=== {rep.scenario} @ fixed FPR {args.fpr:.3%} ===")
    print(f"  behaviour AUC              {rep.auc:.4f}")
    print(f"  detection rate             {rep.detection_rate:.4f}")
    print(f"  precision                  {rep.precision:.4f}")
    print(f"  realised FPR               {rep.realised_fpr:.5f}")
    print(f"  onboarding AUC             {rep.onboarding.get('auc')}")

    a = rep.attempts_to_detection
    print()
    print("  attempts-to-detection")
    print(f"    accounts with fraud      {a['accounts_with_fraud']}")
    print(f"    caught                   {a['accounts_caught']}")
    print(f"    never caught             {a['accounts_never_caught']}")
    print(f"    median attempts first    {a['median_attempts']}")
    print(f"    p90 attempts first       {a['p90_attempts']}")

    print()
    la = rep.lookalike_false_decline
    tf = rep.thin_file_false_decline
    print("  false declines")
    print(f"    look-alike cohort        {la['rate']:.5f}  ({la['declined']}/{la['lookalike_events']})")
    print(f"    other legitimate         {la['baseline_legit_rate']:.5f}")
    print(f"    thin-file legitimate     {tf['rate']:.5f}  vs thick-file {tf['thick_file_rate']:.5f}")

    if rep.view_delta:
        print()
        print("  single-player view vs network view  (the why-Mastercard number)")
        for label, node in (
            ("per institution", rep.view_delta["by_institution"]),
            ("per merchant", rep.view_delta["by_merchant"]),
        ):
            v = node["_overall"]
            print(
                f"    {label:<16} {v.get('groups_compared', 0):>3} compared, "
                f"{v.get('fraud_events', 0):>6} fraud events"
            )
            if not v.get("groups_compared"):
                print("      (no group large enough to compare)")
                continue
            print(f"      that player alone            {v['scoped_view_detection_rate']:.4f}")
            print(f"      network view                 {v['network_view_detection_rate']:.4f}")
            print(f"      delta                        {v['delta']:+.4f}")
        share = rep.view_delta["ring_visibility"]["_overall"][
            "median_share_visible_to_one_institution"
        ]
        if share is not None:
            print(f"    median ring share one bank sees  {share:.3f}")

    if rep.ring_recall_before_transacting:
        o = rep.ring_recall_before_transacting["_overall"]
        print()
        print("  ring recall BEFORE transacting  (the differentiating number)")
        print(f"    accounts that never transacted   {o['never_transacted']}")
        print(f"    flagged anyway                   {o['flagged_before_transacting']}")
        print(f"    recall                           {o['recall']:.4f}")
        pp = rep.propagation_precision
        if pp.get("flagged"):
            print(
                f"    flagged in total                 {pp['flagged']} "
                f"({pp['flagged_share_of_population']:.1%} of all identities)"
            )
            print(f"    of those, actually synthetic     {pp['precision_synthetic']:.4f}")
            print(f"    of those, in a seeded ring       {pp['precision_same_ring']:.4f}")
            print(f"    false flags                      {pp['false_flags']}")
        for ring_id, r in sorted(rep.ring_recall_before_transacting.items()):
            if ring_id.startswith("_"):
                continue
            print(
                f"      {ring_id:<12} size {r['size']:>3}  dormant {r['never_transacted']:>3}"
                f"  flagged {r['flagged_before_transacting']:>3}  recall {r['recall']:.3f}"
            )

    out_path = args.out or (model_dir / "report.json")
    Path(out_path).write_text(json.dumps(asdict(rep), indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
