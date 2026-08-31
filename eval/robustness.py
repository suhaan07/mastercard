"""The robustness curve: detection against attacker sophistication.

A single detection number is a claim about one attacker. The design doc's
fourth claim is stronger and harder -- that detection *holds* as the ring gets
better at hiding -- and the only honest way to show it is the same detector,
the same fixed FPR, and a ring whose operator profile has been turned down.

This aggregates the per-scenario reports rather than recomputing them, so the
curve is assembled from exactly the numbers that were reported individually.
Scenarios are ordered by how much the operator is spending on evasion, not
alphabetically:

* ``sloppy`` -- device reuse, tight subnets, short dormancy, loud bursts
* ``moderate`` -- the middle setting on every knob
* ``sophisticated`` -- low reuse, spread subnets, long dormancy, thin bursts
* ``drift`` -- parameters shift mid-run, so what was learned goes stale

Usage::

    python -m eval.robustness                       # all scenarios found
    python -m eval.robustness --scenarios sloppy sophisticated
    python -m eval.robustness --rebuild             # re-run each report first
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: Increasing operator sophistication. Anything not listed sorts last.
SOPHISTICATION_ORDER = ["sloppy", "moderate", "sophisticated", "drift"]


def order_key(name: str) -> tuple[int, str]:
    return (
        SOPHISTICATION_ORDER.index(name) if name in SOPHISTICATION_ORDER else len(
            SOPHISTICATION_ORDER
        ),
        name,
    )


def rebuild(scenario: str, model_root: Path, fpr: float) -> None:
    """Re-run the report for one scenario, in this interpreter's environment."""
    cmd = [
        sys.executable,
        "-m",
        "eval.report",
        "--data",
        f"data/{scenario}",
        "--models",
        str(model_root),
        "--fpr",
        str(fpr),
    ]
    print(f"  running {' '.join(cmd)}")
    # One scenario failing should not cost the other three: the curve is more
    # useful with a gap in it than not produced at all.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  !! {scenario} report failed (exit {result.returncode}); continuing")


def load_reports(scenarios: list[str], model_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in scenarios:
        path = model_root / name / "report.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"  !! no report for {name} ({path}); run: python -m eval.report --data data/{name}")
    return out


def curve(reports: dict[str, dict]) -> list[dict]:
    rows = []
    for name in sorted(reports, key=order_key):
        r = reports[name]
        atd = r.get("attempts_to_detection", {}) or {}
        ring = (r.get("ring_recall_before_transacting", {}) or {}).get("_overall", {})
        la = r.get("lookalike_false_decline", {}) or {}
        views = ((r.get("view_delta") or {}).get("by_merchant") or {}).get("_overall", {})
        rows.append(
            {
                "scenario": name,
                "target_fpr": r.get("target_fpr"),
                "realised_fpr": r.get("realised_fpr"),
                "auc": r.get("auc"),
                "detection_rate": r.get("detection_rate"),
                "median_attempts_to_detection": atd.get("median_attempts"),
                "accounts_never_caught": atd.get("accounts_never_caught"),
                "ring_recall_before_transacting": ring.get("recall"),
                "lookalike_false_decline": la.get("rate"),
                "merchant_view_detection_rate": views.get("scoped_view_detection_rate"),
                "network_view_detection_rate": views.get("network_view_detection_rate"),
            }
        )
    return rows


def fmt(v, spec: str = ".4f", width: int = 9) -> str:
    if v is None:
        return "--".rjust(width)
    return format(v, spec).rjust(width)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detection vs attacker sophistication")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--models", default="models")
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--rebuild", action="store_true", help="re-run each scenario's report")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_root = Path(args.models)
    scenarios = args.scenarios or sorted(
        (p.name for p in model_root.iterdir() if p.is_dir()), key=order_key
    )

    if args.rebuild:
        for name in scenarios:
            if Path(f"data/{name}").exists():
                rebuild(name, model_root, args.fpr)

    reports = load_reports(scenarios, model_root)
    if not reports:
        raise SystemExit("no reports found; run python -m eval.report --data data/<scenario>")

    rows = curve(reports)

    print()
    print("=== robustness curve: detection vs attacker sophistication ===")
    print("    (same detector, same fixed FPR; only the ring's operator profile changes)")
    print()
    header = (
        f"  {'scenario':<15}{'AUC':>9}{'detect':>9}{'ring pre':>9}"
        f"{'med atts':>9}{'missed':>9}{'lookalike':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['scenario']:<15}"
            f"{fmt(r['auc'])}"
            f"{fmt(r['detection_rate'])}"
            f"{fmt(r['ring_recall_before_transacting'])}"
            f"{fmt(r['median_attempts_to_detection'], '.1f')}"
            f"{fmt(r['accounts_never_caught'], 'd')}"
            f"{fmt(r['lookalike_false_decline'], '.5f', 11)}"
        )

    if any(r["merchant_view_detection_rate"] is not None for r in rows):
        print()
        print("  merchant view vs network view")
        for r in rows:
            if r["merchant_view_detection_rate"] is None:
                continue
            m = r["merchant_view_detection_rate"]
            n = r["network_view_detection_rate"]
            print(f"  {r['scenario']:<15}{fmt(m)}{fmt(n)}   delta {n - m:+.4f}")

    detections = [r["detection_rate"] for r in rows if r["detection_rate"] is not None]
    if len(detections) > 1:
        print()
        print(
            f"  spread across sophistication levels: "
            f"{min(detections):.4f} to {max(detections):.4f} "
            f"({max(detections) - min(detections):.4f})"
        )
        print("  A curve that is flat here is the claim; a cliff is the finding.")

    out_path = Path(args.out or (model_root / "robustness.json"))
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
