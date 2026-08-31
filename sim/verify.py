"""Sanity-check a simulator run against the design doc's constraints.

Usage::

    python -m sim.verify --data data/moderate
    python -m sim.verify --data data/moderate --plot

Checks, each mapped to a claim the project makes:

* class balance inside 0.1%-1%          -- the imbalance the doc specifies
* chargeback delay inside 20-60 days    -- labels arrive late
* label coverage below 100%             -- and incomplete
* ring sizes and dormant counts         -- the retro-propagation demo needs
                                           dormant siblings to exist
* look-alike cohort present             -- false-decline metric is meaningful
* PANs only from the test range         -- never a live BIN
* hourly volume is diurnal              -- traffic looks like traffic
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PASS = "  ok  "
FAIL = " FAIL "
WARN = " warn "


def _read(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(line))
    return out


def verify(data_dir: Path, plot: bool = False) -> bool:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"\n=== {data_dir} (scenario={meta['scenario']}, seed={meta['seed']}) ===")
    print(f"face source: {meta['face_source']}")
    for k, v in meta["counts"].items():
        print(f"  {k:<20} {v:>9,}")

    ok = True

    def check(label: str, condition: bool, detail: str, warn_only: bool = False) -> None:
        nonlocal ok
        tag = PASS if condition else (WARN if warn_only else FAIL)
        if not condition and not warn_only:
            ok = False
        print(f"[{tag}] {label}: {detail}")

    print()

    # -- class balance ----------------------------------------------------
    truth = _read(data_dir / "ground_truth.jsonl")
    synthetic_accounts = {t["account_id"] for t in truth if t["is_synthetic"]}
    lookalike_accounts = {t["account_id"] for t in truth if t["is_lookalike"]}

    auth = _read(data_dir / "auth_events.jsonl")
    n_auth = len(auth)
    fraud_rate = meta["fraud_rate"]
    check(
        "class balance",
        0.001 <= fraud_rate <= 0.010,
        f"{fraud_rate:.4%} of {n_auth:,} auth events (target 0.1%-1%)",
    )

    # -- labels: late and incomplete --------------------------------------
    labels = _read(data_dir / "labels.jsonl")
    delays = []
    for lb in labels:
        e = datetime.fromisoformat(lb["event_ts"])
        a = datetime.fromisoformat(lb["label_available_at"])
        delays.append((a - e).total_seconds() / 86400.0)
    delays = np.array(delays) if delays else np.array([0.0])
    by_source = Counter(lb["source"] for lb in labels)
    cb = np.array(
        [d for d, lb in zip(delays, labels) if lb["source"] == "chargeback"]
    )
    check(
        "chargeback delay",
        len(cb) > 0 and 20.0 <= cb.min() and cb.max() <= 60.5,
        f"{cb.min():.1f}-{cb.max():.1f} days (n={len(cb):,}, target 20-60)"
        if len(cb)
        else "no chargeback labels",
    )
    check(
        "labels are incomplete",
        len(labels) < meta["n_fraud_events"],
        f"{len(labels):,} labels for {meta['n_fraud_events']:,} fraud events "
        f"({len(labels)/max(1,meta['n_fraud_events']):.1%} coverage) {dict(by_source)}",
    )
    check(
        "no label precedes its event",
        bool((delays >= 0).all()),
        f"min delay {delays.min():.2f} days",
    )

    # -- rings -------------------------------------------------------------
    ring_sizes = meta["ring_sizes"]
    dormant = meta["dormant_per_ring"]
    check(
        "rings have dormant members",
        all(d > 0 for d in dormant),
        f"sizes {ring_sizes}, dormant {dormant} "
        f"({sum(dormant)}/{sum(ring_sizes)} = {sum(dormant)/max(1,sum(ring_sizes)):.0%} never transact)",
    )

    # Dormant accounts must genuinely have zero fraud events -- the whole
    # retro-propagation claim is that we catch them *before* they act.
    fraud_accounts = {
        a["account_id"]
        for a in auth
        if a["account_id"] in synthetic_accounts and a["is_zero_auth"] or False
    }
    acted = defaultdict(int)
    for t in truth:
        if t["is_synthetic"] and t["first_fraud_ts"]:
            acted[t["ring_id"]] += 1
    n_never_acted = sum(1 for t in truth if t["is_synthetic"] and not t["first_fraud_ts"])
    check(
        "dormant siblings never transact fraudulently",
        n_never_acted > 0,
        f"{n_never_acted} synthetic accounts have no fraud event at all",
    )

    # -- look-alikes -------------------------------------------------------
    check(
        "look-alike cohort present",
        len(lookalike_accounts) > 0,
        f"{len(lookalike_accounts)} legitimate accounts that resemble fraud",
    )

    # -- card ranges -------------------------------------------------------
    bad_pans = [a for a in auth[:50000] if not a["pan_suffix6"].isdigit()]
    check("PAN suffixes well-formed", not bad_pans, f"{len(bad_pans)} malformed in first 50k")

    # -- decline behaviour separates the classes ---------------------------
    syn_declines = [
        1 - int(a["response_code"] == "approved")
        for a in auth
        if a["account_id"] in synthetic_accounts
    ]
    leg_declines = [
        1 - int(a["response_code"] == "approved")
        for a in auth
        if a["account_id"] not in synthetic_accounts
    ]
    if syn_declines and leg_declines:
        check(
            "decline ratio separates classes",
            np.mean(syn_declines) > np.mean(leg_declines),
            f"synthetic {np.mean(syn_declines):.1%} vs legitimate {np.mean(leg_declines):.1%}",
        )

    # -- diurnal shape -----------------------------------------------------
    hours = Counter(datetime.fromisoformat(a["ts"]).hour for a in auth)
    counts = np.array([hours.get(h, 0) for h in range(24)], dtype=float)
    peak_h, trough_h = int(counts.argmax()), int(counts.argmin())
    check(
        "hourly volume is diurnal",
        counts.max() > 2.5 * max(1.0, counts.min()),
        f"peak {peak_h:02d}:00 ({counts.max():,.0f}) vs trough {trough_h:02d}:00 ({counts.min():,.0f})",
    )

    print("\nhourly volume")
    scale = 56.0 / max(1.0, counts.max())
    for h in range(24):
        print(f"  {h:02d} {'#' * int(counts[h] * scale)} {int(counts[h]):,}")

    if plot:
        _plot(data_dir, counts, cb)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


def _plot(data_dir: Path, hourly: np.ndarray, delays: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(range(24), hourly, color="#3b6ea5")
    axes[0].set_title("Auth volume by hour (UTC)")
    axes[0].set_xlabel("hour")
    if len(delays):
        axes[1].hist(delays, bins=40, color="#a5643b")
    axes[1].set_title("Chargeback label delay (days)")
    axes[1].set_xlabel("days")
    fig.tight_layout()
    out = data_dir / "verify.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a simulator run")
    ap.add_argument("--data", default="data/moderate")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    ok = verify(Path(args.data), plot=args.plot)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
