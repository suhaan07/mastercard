"""Decision-engine types: actions, bands, reason codes and the scoring API shape.

Every block needs a human-readable reason -- that is a regulatory requirement
and an analyst-workflow one, and it is on the constraints table in the design
doc. So reason codes are an enum with mandated text, not free-form strings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ViewScope(str, Enum):
    """What slice of the world the caller is allowed to see.

    This is the merchant-view/network-view toggle. It is enforced by the
    gateway and the mock services, not merely filtered in the UI: a merchant
    caller genuinely cannot retrieve cross-institution features.
    """

    MERCHANT = "merchant"
    ISSUER = "issuer"
    NETWORK = "network"


class Band(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    ELEVATED = "elevated"
    HIGH = "high"
    CONFIRMED = "confirmed"


class Action(str, Enum):
    ALLOW = "allow"
    STEP_UP = "step_up"
    THROTTLE = "throttle"
    #: Return plausible responses so the attacker cannot distinguish live cards
    #: from dead ones. Poisons their results rather than telling them they were
    #: caught.
    HONEYPOT = "honeypot"
    BLOCK = "block"


BAND_ACTION: dict[Band, Action] = {
    Band.LOW: Action.ALLOW,
    Band.MEDIUM: Action.STEP_UP,
    Band.ELEVATED: Action.THROTTLE,
    Band.HIGH: Action.HONEYPOT,
    Band.CONFIRMED: Action.BLOCK,
}


class ReasonCode(str, Enum):
    # Behavioural / card testing
    HIGH_DECLINE_RATIO = "high_decline_ratio"
    MANY_DISTINCT_PANS = "many_distinct_pans"
    LOW_PAN_ENTROPY = "low_pan_entropy"
    ZERO_AUTH_BURST = "zero_auth_burst"
    LOW_TICKET_CONCENTRATION = "low_ticket_concentration"
    CVV_AVS_MISMATCH_RATE = "cvv_avs_mismatch_rate"
    MACHINE_REGULAR_TIMING = "machine_regular_timing"
    NARROW_MERCHANT_SET = "narrow_merchant_set"
    VELOCITY_SPIKE = "velocity_spike"

    # Identity / onboarding
    SYNTHETIC_IDENTITY_SIGNALS = "synthetic_identity_signals"
    GAN_ARTIFACTS_DETECTED = "gan_artifacts_detected"
    NEAR_DUPLICATE_FACE_TAG = "near_duplicate_face_tag"
    SHARED_DOC_TEMPLATE = "shared_doc_template"
    PII_RECOMBINATION = "pii_recombination"
    THIN_FILE_SUDDEN_ACTIVITY = "thin_file_sudden_activity"
    CREDIT_FILE_AGE_INCONSISTENT = "credit_file_age_inconsistent"
    MAIL_DROP_ADDRESS = "mail_drop_address"

    # Ring / graph
    RING_MEMBERSHIP = "ring_membership"
    RING_SIBLING_CONFIRMED_FRAUD = "ring_sibling_confirmed_fraud"
    SHARED_DEVICE_WITH_FLAGGED = "shared_device_with_flagged"
    SUBNET_CONCENTRATION = "subnet_concentration"

    # Cross-institution
    CROSS_INSTITUTION_INDICATOR = "cross_institution_indicator"
    NETWORK_WIDE_VELOCITY = "network_wide_velocity"


REASON_TEXT: dict[ReasonCode, str] = {
    ReasonCode.HIGH_DECLINE_RATIO: "Unusually high share of declined attempts in a short window.",
    ReasonCode.MANY_DISTINCT_PANS: "Many distinct card numbers attempted from one account.",
    ReasonCode.LOW_PAN_ENTROPY: "Card numbers are close to sequential, indicating enumeration.",
    ReasonCode.ZERO_AUTH_BURST: "Burst of zero-value authorisations, typical of liveness probing.",
    ReasonCode.LOW_TICKET_CONCENTRATION: "Attempts concentrated at very low amounts.",
    ReasonCode.CVV_AVS_MISMATCH_RATE: "Elevated rate of CVV and address verification failures.",
    ReasonCode.MACHINE_REGULAR_TIMING: "Inter-attempt timing is too regular to be human.",
    ReasonCode.NARROW_MERCHANT_SET: "Testing concentrated on a small set of merchants.",
    ReasonCode.VELOCITY_SPIKE: "Attempt rate far above this account's own baseline.",
    ReasonCode.SYNTHETIC_IDENTITY_SIGNALS: "Identity verification signals fit a manufactured profile.",
    ReasonCode.GAN_ARTIFACTS_DETECTED: "Applicant image shows statistical artifacts of AI generation.",
    ReasonCode.NEAR_DUPLICATE_FACE_TAG: "Face similarity tag closely matches other applicants.",
    ReasonCode.SHARED_DOC_TEMPLATE: "Identity document shares a generation template with other applicants.",
    ReasonCode.PII_RECOMBINATION: "Personal details reappear across applicants under different names.",
    ReasonCode.THIN_FILE_SUDDEN_ACTIVITY: "Thin credit file followed by an abrupt jump in activity.",
    ReasonCode.CREDIT_FILE_AGE_INCONSISTENT: "Credit file age is inconsistent with the declared age.",
    ReasonCode.MAIL_DROP_ADDRESS: "Address is shared by many otherwise unrelated applicants.",
    ReasonCode.RING_MEMBERSHIP: "Account belongs to a cluster with strong shared-attribute density.",
    ReasonCode.RING_SIBLING_CONFIRMED_FRAUD: "A confirmed fraudulent account shares infrastructure with this one.",
    ReasonCode.SHARED_DEVICE_WITH_FLAGGED: "Device fingerprint is shared with an already-flagged account.",
    ReasonCode.SUBNET_CONCENTRATION: "Originating addresses are tightly clustered in one subnet.",
    ReasonCode.CROSS_INSTITUTION_INDICATOR: "Indicator matches a suspicious marker reported by another institution.",
    ReasonCode.NETWORK_WIDE_VELOCITY: "Activity is unremarkable at one merchant but severe network-wide.",
}


def reason_text(code: ReasonCode) -> str:
    """Human-readable explanation for a reason code."""
    return REASON_TEXT[code]


class EvidenceHop(BaseModel):
    """One hop of a retro-propagation path, so a propagated score can explain itself."""

    model_config = ConfigDict(frozen=True)

    from_node: str
    to_node: str
    via: str = Field(description="Edge type that carried the evidence")
    contribution: float


class ScoreRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    view: ViewScope = ViewScope.NETWORK
    #: Present for auth scoring; absent for onboarding scoring.
    auth_event: dict | None = None
    onboarding_event: dict | None = None


class ScoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    score: float = Field(ge=0.0, le=1.0)
    band: Band
    action: Action
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    ring_id: str | None = None
    view: ViewScope = ViewScope.NETWORK
    #: Populated when the score was raised by evidence from a ring sibling
    #: rather than by this account's own behaviour. This is the retro-
    #: propagation demo, made auditable.
    propagated: bool = False
    evidence_path: list[EvidenceHop] = Field(default_factory=list)
    scored_at: datetime | None = None
    latency_ms: float = 0.0

    def explain(self) -> list[str]:
        return [reason_text(c) for c in self.reason_codes]
