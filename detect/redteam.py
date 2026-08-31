"""The adversarial loop: an attacker that adapts to whatever we deployed.

A static detector has a half-life. Once the obvious burst pattern is blocked,
the operator slows down, spreads across merchants, buys more devices and jitters
its timing -- and every one of those is a knob on
:class:`~sim.config.RingProfile`. So the red team is a search over that
parameter space with a single objective: **evade the deployed detector at the
fixed FPR**.

The search is hill-climbing with restarts rather than anything fancier. The
space is seven-dimensional, evaluation is expensive (each candidate regenerates
ring traffic and rescores it), and the point is to demonstrate an arms race
converging, not to find a global optimum. A defender who needs a global optimum
to feel worried has misunderstood the problem.

Everything happens inside the simulator. This produces evasion *parameters*, not
tooling: the output is a table of which defensive signals degrade under which
operator behaviour, which is a defensive result.

Usage::

    python -m detect.redteam --data data/moderate --rounds 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.config import RingProfile, ScenarioConfig, load_scenario

#: The knobs the red team may turn, with their bounds. Each corresponds to
#: something an operator can actually change: buy more devices, rent more
#: subnets, wait longer, spray wider, slow down.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "device_reuse_rate": (0.02, 0.95),
    "subnet_concentration": (0.02, 0.95),
    "pii_recombination_rate": (0.0, 0.7),
    "doc_template_reuse": (0.02, 0.95),
    "face_reuse_rate": (0.02, 0.95),
    "face_jitter": (0.02, 0.20),
    "artifact_strength": (0.05, 1.8),
    "burst_intensity": (3.0, 90.0),
    "merchant_spread": (1, 25),
    "inter_arrival_jitter": (0.05, 1.5),
    "dormancy_days_min": (1, 45),
}

#: Knobs that cost the operator something to turn. Evasion is not free: buying
#: a device per account, waiting six weeks before acting, or spreading over
#: twenty merchants all reduce yield. Without a cost term the search simply
#: turns every knob to maximum evasion and reports a useless "undetectable"
#: configuration.
EVASION_COST: dict[str, float] = {
    "device_reuse_rate": -1.0,      # lower reuse = more devices to buy
    "subnet_concentration": -0.6,   # lower concentration = more proxies
    "doc_template_reuse": -0.4,
    "face_reuse_rate": -0.5,
    "artifact_strength": -0.8,      # a better generator costs more
    "burst_intensity": 0.5,         # slower testing = less throughput
    "merchant_spread": -0.2,
    "dormancy_days_min": -0.5,      # waiting costs time
}


@dataclass
class Candidate:
    params: dict[str, float]
    detection_rate: float = 1.0
    operator_cost: float = 0.0
    #: Detection minus cost. The attacker minimises this.
    objective: float = 1.0
    notes: dict = field(default_factory=dict)


def operator_cost(params: dict[str, float]) -> float:
    """How expensive this configuration is to run, normalised to roughly [0, 1]."""
    total = 0.0
    for k, direction in EVASION_COST.items():
        lo, hi = SEARCH_SPACE[k]
        norm = (float(params[k]) - lo) / (hi - lo)
        # direction < 0 means *lower* values cost more
        total += (1.0 - norm) * abs(direction) if direction < 0 else norm * direction
    return float(total / sum(abs(v) for v in EVASION_COST.values()))


def perturb(
    params: dict[str, float], rng: np.random.Generator, scale: float = 0.25
) -> dict[str, float]:
    """Move a few knobs at once. Changing one at a time explores far too slowly."""
    out = dict(params)
    n = int(rng.integers(2, 5))
    for k in rng.choice(list(SEARCH_SPACE), size=n, replace=False):
        lo, hi = SEARCH_SPACE[str(k)]
        span = (hi - lo) * scale
        val = float(out[str(k)]) + float(rng.normal(0, span))
        out[str(k)] = float(np.clip(val, lo, hi))
        if str(k) in ("merchant_spread", "dormancy_days_min"):
            out[str(k)] = float(int(round(out[str(k)])))
    return out


def apply_params(profile: RingProfile, params: dict[str, float]) -> RingProfile:
    updates = {k: v for k, v in params.items() if hasattr(profile, k)}
    if "merchant_spread" in updates:
        updates["merchant_spread"] = int(updates["merchant_spread"])
    if "dormancy_days_min" in updates:
        lo = int(updates["dormancy_days_min"])
        updates["dormancy_days_min"] = lo
        updates["dormancy_days_max"] = max(lo + 3, int(profile.dormancy_days_max))
    return dataclasses.replace(profile, **updates)


def evaluate(
    cfg: ScenarioConfig,
    params: dict[str, float],
    model,
    target_fpr: float,
    seed: int,
) -> Candidate:
    """Regenerate ring traffic under ``params`` and score it with the deployed model.

    Only the ring side is regenerated. The legitimate population is held fixed
    so that the measured change is attributable to the attacker's behaviour and
    not to a different world.
    """
    from detect.features.stream import StreamFeatureStore
    from detect.models.train import metrics_at_threshold, to_matrix
    from sim.run import Simulator

    trial = dataclasses.replace(cfg, seed=seed)
    trial.ring = apply_params(cfg.ring, params)
    # A small population keeps each round affordable; the ring is what matters.
    trial.population = dataclasses.replace(trial.population, n_customers=450)
    trial.days = min(cfg.days, 70)

    sim = Simulator(trial)
    customers, lookalikes, rings = sim.build()

    attempts = []
    for cust in customers + lookalikes:
        attempts.extend(sim.legit_attempts(cust))
    fraud_ids: set[str] = set()
    for ring in rings:
        for member in ring.members:
            warm = sim.cardtest.warmup(
                member,
                until_day=member.activation_day
                if member.activation_day is not None
                else trial.days,
            )
            burst = sim.cardtest.generate(ring, member)
            attempts.extend(warm)
            attempts.extend(burst)
            for a in burst:
                fraud_ids.add(id(a))

    attempts.sort(key=lambda a: a.day)
    store = StreamFeatureStore()
    rows, y = [], []
    for a in attempts:
        ev = sim.auth_event(a)
        rows.append(store.features(ev))
        y.append(int(id(a) in fraud_ids))
        store.update(ev)

    y_arr = np.array(y)
    if y_arr.sum() == 0:
        return Candidate(params=params, detection_rate=0.0, operator_cost=operator_cost(params),
                         objective=-operator_cost(params), notes={"note": "no fraud generated"})

    scores = np.asarray(model.booster.predict_proba(to_matrix(rows, model.feature_names))[:, 1])
    m = metrics_at_threshold(y_arr, scores, model.threshold_at_target_fpr)
    cost = operator_cost(params)
    return Candidate(
        params=params,
        detection_rate=m["detection_rate"],
        operator_cost=cost,
        # The attacker wants low detection and low cost.
        objective=m["detection_rate"] + 0.35 * cost,
        notes={"n_fraud": int(y_arr.sum()), "n_events": len(y), "fpr": m["false_positive_rate"]},
    )


def run(
    data_dir: Path,
    scenario: str,
    rounds: int,
    target_fpr: float,
    seed: int = 11,
) -> dict:
    from detect.models.train import TrainedModel

    cfg = load_scenario(scenario)
    model = TrainedModel.load(Path("models") / data_dir.name / "behaviour.pkl")
    rng = np.random.default_rng(seed)

    current = {k: float(getattr(cfg.ring, k, (lo + hi) / 2)) for k, (lo, hi) in SEARCH_SPACE.items()}
    best = evaluate(cfg, current, model, target_fpr, seed)
    history = [best]

    print(f"round  detection  op-cost  objective   changed")
    print(f"{0:>5}  {best.detection_rate:>9.4f}  {best.operator_cost:>7.3f}  "
          f"{best.objective:>9.4f}   (baseline)")

    for r in range(1, rounds + 1):
        cand_params = perturb(best.params, rng, scale=0.3)
        cand = evaluate(cfg, cand_params, model, target_fpr, seed + r)
        changed = [
            k
            for k in SEARCH_SPACE
            if abs(cand_params[k] - best.params[k]) > 1e-9
        ]
        accepted = cand.objective < best.objective
        print(
            f"{r:>5}  {cand.detection_rate:>9.4f}  {cand.operator_cost:>7.3f}  "
            f"{cand.objective:>9.4f}   {'ACCEPT' if accepted else '  --  '} "
            f"{','.join(changed[:3])}"
        )
        if accepted:
            best = cand
        history.append(cand)

    print()
    print(f"baseline detection {history[0].detection_rate:.4f} "
          f"-> best evasion {best.detection_rate:.4f}")
    print(f"operator cost      {history[0].operator_cost:.3f} -> {best.operator_cost:.3f}")
    print("\nknobs the attacker turned:")
    for k in SEARCH_SPACE:
        a, b = history[0].params[k], best.params[k]
        if abs(a - b) > 1e-6:
            print(f"  {k:<26} {a:>8.3f} -> {b:>8.3f}")

    return {
        "scenario": scenario,
        "rounds": rounds,
        "baseline_detection": history[0].detection_rate,
        "best_evasion_detection": best.detection_rate,
        "baseline_cost": history[0].operator_cost,
        "best_cost": best.operator_cost,
        "best_params": best.params,
        "history": [
            {
                "detection_rate": c.detection_rate,
                "operator_cost": c.operator_cost,
                "objective": c.objective,
                **c.notes,
            }
            for c in history
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Red-team search against the deployed detector")
    ap.add_argument("--data", default="data/moderate")
    ap.add_argument("--scenario", default=None, help="defaults to the data dir name")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data)
    result = run(data_dir, args.scenario or data_dir.name, args.rounds, args.fpr)

    out = args.out or (Path("models") / data_dir.name / "redteam.json")
    Path(out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
