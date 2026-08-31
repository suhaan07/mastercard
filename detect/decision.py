"""L6: the decision engine. Score plus context in, graduated action out.

Blocking is not the only response, and it is usually the wrong one. The bands
here escalate:

* **Allow** -- the overwhelming majority.
* **Step-up** -- a 3DS challenge. Costs the customer ten seconds, costs the
  attacker the whole attempt.
* **Throttle** -- silently rate-limit. The attacker's script keeps running and
  learns nothing, while the damage rate falls.
* **Honeypot** -- return plausible responses so the attacker cannot tell live
  cards from dead ones. This is the interesting one: a block tells the attacker
  the account is burned and to move to the next of the four hundred. Poisoned
  results make their *entire harvest* untrustworthy, which is far more
  expensive to them than losing one account.
* **Block** -- and freeze the ring, with a case for an analyst.

Every action emits reason codes, because "computer says no" is not a decision a
regulator or an analyst can work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from contracts.decisions import (
    BAND_ACTION,
    Action,
    Band,
    EvidenceHop,
    ReasonCode,
    ScoreResponse,
    ViewScope,
)

RULES_PATH = Path(__file__).parent / "rules.yaml"


@dataclass
class Thresholds:
    medium: float = 0.30
    elevated: float = 0.55
    high: float = 0.75
    confirmed: float = 0.90

    def band(self, score: float) -> Band:
        if score >= self.confirmed:
            return Band.CONFIRMED
        if score >= self.high:
            return Band.HIGH
        if score >= self.elevated:
            return Band.ELEVATED
        if score >= self.medium:
            return Band.MEDIUM
        return Band.LOW


@dataclass
class RuleConfig:
    thresholds: Thresholds = field(default_factory=Thresholds)
    #: Feature triggers that add a reason code when exceeded. Reason codes are
    #: explanatory, not additive -- the score decides the action, these say why.
    triggers: dict[str, dict] = field(default_factory=dict)
    #: How much a poor onboarding score tightens behavioural thresholds. This is
    #: the forward half of the seam.
    onboarding_tightening: float = 0.25

    @classmethod
    def load(cls, path: Path | str = RULES_PATH) -> "RuleConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        t = raw.get("thresholds", {})
        return cls(
            thresholds=Thresholds(**t) if t else Thresholds(),
            triggers=raw.get("triggers", {}),
            onboarding_tightening=float(raw.get("onboarding_tightening", 0.25)),
        )


def _triggered_reasons(features: dict[str, float], triggers: dict[str, dict]) -> list[ReasonCode]:
    """Reason codes whose feature trigger fired, strongest first."""
    fired: list[tuple[float, ReasonCode]] = []
    for name, spec in triggers.items():
        feature = spec.get("feature")
        if feature is None or feature not in features:
            continue
        value = float(features[feature])
        above = spec.get("above")
        below = spec.get("below")
        hit = (above is not None and value >= float(above)) or (
            below is not None and value <= float(below)
        )
        if not hit:
            continue
        try:
            code = ReasonCode(name)
        except ValueError:
            continue
        # Rank by how far past the trigger the value sits, so the most extreme
        # signal is explained first.
        ref = float(above) if above is not None else float(below)
        margin = abs(value - ref) / (abs(ref) + 1e-6)
        fired.append((margin, code))
    fired.sort(key=lambda x: -x[0])
    return [c for _, c in fired]


class DecisionEngine:
    """Turns a fused score plus its context into an action with reasons."""

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig.load()

    def effective_thresholds(self, onboarding_score: float) -> Thresholds:
        """Tighten thresholds for accounts that onboarded badly.

        The forward direction of the fusion: how an account was born changes how
        much behavioural evidence is needed to act on it. A ten-year customer
        gets the benefit of the doubt; an account that barely passed KYC three
        weeks ago does not.
        """
        k = 1.0 - self.config.onboarding_tightening * max(0.0, min(1.0, onboarding_score))
        t = self.config.thresholds
        return Thresholds(
            medium=t.medium * k,
            elevated=t.elevated * k,
            high=t.high * k,
            confirmed=t.confirmed * k,
        )

    def decide(
        self,
        *,
        event_id: str,
        score: float,
        features: dict[str, float] | None = None,
        onboarding_score: float = 0.0,
        ring_id: str | None = None,
        propagated: bool = False,
        evidence_path: list[EvidenceHop] | None = None,
        view: ViewScope = ViewScope.NETWORK,
        latency_ms: float = 0.0,
    ) -> ScoreResponse:
        features = features or {}
        thresholds = self.effective_thresholds(onboarding_score)
        band = thresholds.band(score)
        action = BAND_ACTION[band]

        reasons = _triggered_reasons(features, self.config.triggers)

        if propagated:
            # The account may have done nothing itself. Say so explicitly --
            # this is the reason code an analyst most needs to see, because the
            # account's own history looks clean.
            reasons.insert(0, ReasonCode.RING_SIBLING_CONFIRMED_FRAUD)
        if ring_id and ReasonCode.RING_MEMBERSHIP not in reasons:
            reasons.append(ReasonCode.RING_MEMBERSHIP)
        if view is ViewScope.NETWORK and features.get("cross_institution_hits", 0) > 0:
            reasons.append(ReasonCode.CROSS_INSTITUTION_INDICATOR)

        if not reasons and band is not Band.LOW:
            reasons.append(ReasonCode.VELOCITY_SPIKE)

        # Deduplicate, preserving order.
        seen: set[ReasonCode] = set()
        ordered = [r for r in reasons if not (r in seen or seen.add(r))]

        return ScoreResponse(
            event_id=event_id,
            score=float(max(0.0, min(1.0, score))),
            band=band,
            action=action,
            reason_codes=ordered[:6],
            ring_id=ring_id,
            view=view,
            propagated=propagated,
            evidence_path=evidence_path or [],
            scored_at=datetime.now(timezone.utc),
            latency_ms=latency_ms,
        )


def honeypot_response(card_token: str) -> dict:
    """A plausible-looking authorisation result that carries no information.

    Deterministic in the card token, so the attacker retrying the same card gets
    a consistent answer and cannot detect the honeypot by probing for
    inconsistency. Roughly one in twenty "approves", matching the rate a real
    testing run would see -- an endpoint that declines everything is as
    informative to the attacker as one that approves everything.
    """
    import hashlib

    h = int.from_bytes(hashlib.blake2b(card_token.encode(), digest_size=4).digest(), "big")
    approved = (h % 20) == 0
    return {
        "response_code": "approved" if approved else "declined_do_not_honor",
        "avs_result": "match" if h % 3 else "no_match",
        "cvv_result": "match" if approved else "no_match",
        "honeypot": True,
    }
