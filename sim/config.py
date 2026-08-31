"""Scenario configuration for the simulator.

Everything that shapes a run lives in a YAML scenario file so that a result is
reproducible from a seed plus a config, and nothing else. Judges ask.

The ring operator profile is the difficulty dial the design doc calls for: turn
the knobs up and you get a lazy operator that shares devices and bursts hard;
turn them down and you get one that spreads across merchants, ages its accounts
and reuses almost nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCENARIO_DIR = Path(__file__).parent / "scenarios"


@dataclass
class RingProfile:
    """How sloppy a fraud ring's operator is.

    Every field is in [0, 1] except the counts and day figures. Higher means
    sloppier and therefore easier to catch -- except ``merchant_spread`` and
    ``dormancy_days``, where higher means *more* sophisticated, since spreading
    attempts and ageing accounts are both evasive behaviours.
    """

    name: str = "moderate"
    n_rings: int = 4
    ring_size_min: int = 20
    ring_size_max: int = 50

    device_reuse_rate: float = 0.55
    subnet_concentration: float = 0.60
    pii_recombination_rate: float = 0.35
    doc_template_reuse: float = 0.60
    face_reuse_rate: float = 0.50
    #: Jitter applied to a reused face vector. Smaller means more obviously
    #: duplicated faces.
    face_jitter: float = 0.06
    #: Generator quality: how loud the image artifacts are.
    artifact_strength: float = 0.80

    dormancy_days_min: int = 7
    dormancy_days_max: int = 21
    #: Card-testing attempts per hour during a burst.
    burst_intensity: float = 45.0
    burst_hours: float = 3.0
    n_bursts_per_account: int = 1
    #: How many merchants the testing is split across.
    merchant_spread: int = 3
    #: Timing jitter. Low means machine-regular, which is itself a tell.
    inter_arrival_jitter: float = 0.25
    #: Share of ring accounts that never act during the run -- the dormant
    #: siblings that retro-propagation is supposed to catch.
    dormant_share: float = 0.65
    #: Fraction of the run over which a ring's accounts are opened.
    #:
    #: Was 0.35, which concentrated every burst into the first third of the
    #: timeline: with 2-7 day dormancy, all sloppy fraud landed in days 2-38 of
    #: a 90-day run, so any time-ordered holdout was structurally fraud-free and
    #: the behaviour model had nothing to be evaluated against. Real operators
    #: open accounts continuously.
    onboard_window_fraction: float = 0.70


@dataclass
class PopulationConfig:
    n_customers: int = 4000
    n_merchants: int = 220
    n_institutions: int = 6
    devices_min: int = 1
    devices_max: int = 3
    #: Log-normal transaction amount parameters.
    amount_mu: float = 3.4
    amount_sigma: float = 1.0
    #: Mean transactions per customer per day.
    daily_rate_mean: float = 0.75
    daily_rate_sigma: float = 0.5
    #: Probability per customer of a genuine life event during the run.
    p_device_change: float = 0.12
    p_travel: float = 0.10
    p_move: float = 0.05
    base_decline_rate: float = 0.055


@dataclass
class LookalikeConfig:
    """Legitimate customers whose behaviour resembles fraud.

    These are the false-decline risk, and reporting a detection rate without
    them is reporting a number that means nothing.
    """

    n_small_business: int = 45
    n_travellers: int = 40
    n_new_device: int = 40
    small_business_daily_rate: float = 18.0
    small_business_amount_mu: float = 2.1


@dataclass
class LabelConfig:
    chargeback_delay_min_days: int = 20
    chargeback_delay_max_days: int = 60
    #: Not every fraud is charged back. Incomplete labels are the real world.
    chargeback_coverage: float = 0.72
    #: Analyst review catches some fraud sooner.
    analyst_review_share: float = 0.10
    analyst_delay_days: int = 3
    #: Legitimate transactions occasionally get charged back too.
    false_chargeback_rate: float = 0.0008


@dataclass
class DriftConfig:
    """Concept drift: the attacker changes tactics mid-run."""

    enabled: bool = False
    #: Fraction of the way through the run at which parameters shift.
    at_fraction: float = 0.55
    #: Multipliers applied to the ring profile after the shift.
    burst_intensity_mult: float = 0.35
    merchant_spread_mult: float = 2.5
    device_reuse_mult: float = 0.4
    inter_arrival_jitter_mult: float = 2.5


@dataclass
class ScenarioConfig:
    name: str = "moderate"
    seed: int = 20260830
    days: int = 90
    start_date: str = "2026-03-01"
    institution_prefix: str = "inst"

    population: PopulationConfig = field(default_factory=PopulationConfig)
    ring: RingProfile = field(default_factory=RingProfile)
    lookalikes: LookalikeConfig = field(default_factory=LookalikeConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)

    #: Target share of auth events that are fraudulent. The simulator warns if
    #: the realised rate falls outside 0.1%-1%, the band the design doc sets.
    target_fraud_rate_min: float = 0.001
    target_fraud_rate_max: float = 0.010

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_scenario(name_or_path: str) -> ScenarioConfig:
    """Load a scenario by name (from ``sim/scenarios``) or by explicit path."""
    path = Path(name_or_path)
    if not path.exists():
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"no scenario {name_or_path!r}; available: {available}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _merge(asdict(ScenarioConfig()), raw)

    return ScenarioConfig(
        name=merged["name"],
        seed=merged["seed"],
        days=merged["days"],
        start_date=merged["start_date"],
        institution_prefix=merged["institution_prefix"],
        population=PopulationConfig(**merged["population"]),
        ring=RingProfile(**merged["ring"]),
        lookalikes=LookalikeConfig(**merged["lookalikes"]),
        labels=LabelConfig(**merged["labels"]),
        drift=DriftConfig(**merged["drift"]),
        target_fraud_rate_min=merged["target_fraud_rate_min"],
        target_fraud_rate_max=merged["target_fraud_rate_max"],
    )
