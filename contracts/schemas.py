"""Event schemas for the four streams, plus the biometric tag type.

Design rules that the rest of the system depends on:

* Every event carries ``event_id``, ``ts`` (UTC) and ``institution_id``. The
  institution is what makes the merchant-view/network-view split physically
  real: a single institution only ever sees its own events.
* No PII crosses this boundary. Names, addresses, emails, phones and DOBs are
  present only as opaque tokens produced by the ingest vault.
* No biometric embeddings, anywhere. Face and document similarity travel as
  :class:`SparseTag` -- the sparse index set of a FlyHash tag. See
  ``biohash/flyhash.py`` for why.
* Labels carry ``label_available_at`` as well as ``event_ts``. Training must
  filter on the former; using the latter is future leakage and
  ``detect/models/train.py`` asserts against it.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# Biometric / structural similarity tags
# --------------------------------------------------------------------------


class SparseTag(BaseModel):
    """A FlyHash tag: the indices of the winning cells, never a dense vector.

    Stored as a sorted index set so that similarity is a set operation and
    therefore composes directly with the Bloom-filter exchange in
    ``privacy/psi.py``. ``seed_id`` records which institution's secret
    projection produced the tag -- tags from different seeds are not
    comparable, which is exactly the unlinkability property we want.
    """

    model_config = ConfigDict(frozen=True)

    dim: int = Field(gt=0, description="Width of the expanded (Kenyon-cell) layer")
    indices: tuple[int, ...] = Field(description="Sorted indices of active cells")
    seed_id: str = Field(description="Institution seed that produced this tag")

    @field_validator("indices")
    @classmethod
    def _sorted_unique(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(v)) != len(v):
            raise ValueError("tag indices must be unique")
        return tuple(sorted(v))

    @property
    def sparsity(self) -> float:
        return len(self.indices) / self.dim

    def overlap(self, other: "SparseTag") -> float:
        """Jaccard overlap. Returns 0.0 across different seeds by construction."""
        if self.seed_id != other.seed_id or self.dim != other.dim:
            return 0.0
        a, b = set(self.indices), set(other.indices)
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    def as_set(self) -> frozenset[int]:
        return frozenset(self.indices)


# --------------------------------------------------------------------------
# Enumerations used across the auth stream
# --------------------------------------------------------------------------


class AuthResponseCode(str, Enum):
    APPROVED = "approved"
    DECLINED_INSUFFICIENT_FUNDS = "declined_insufficient_funds"
    DECLINED_INVALID_CARD = "declined_invalid_card"
    DECLINED_EXPIRED = "declined_expired"
    DECLINED_CVV = "declined_cvv"
    DECLINED_DO_NOT_HONOR = "declined_do_not_honor"
    DECLINED_RISK = "declined_risk"


class AvsResult(str, Enum):
    MATCH = "match"
    PARTIAL = "partial"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"


class CvvResult(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    NOT_PROVIDED = "not_provided"


class LabelSource(str, Enum):
    CHARGEBACK = "chargeback"
    ANALYST = "analyst"
    CONFIRMED_FRAUD = "confirmed_fraud"


# --------------------------------------------------------------------------
# Verification signals (onboarding)
# --------------------------------------------------------------------------


class VerificationSignals(BaseModel):
    """What an identity-verification vendor would emit, plus our own detector.

    The first three are classic vendor outputs. The remainder come from
    ``biohash/artifacts.py`` and are computed from the applicant's selfie at
    onboarding time -- never in the auth path.
    """

    model_config = ConfigDict(frozen=True)

    template_match_score: float = Field(ge=0.0, le=1.0)
    exif_consistency: float = Field(ge=0.0, le=1.0)
    liveness_score: float = Field(ge=0.0, le=1.0)

    # GAN-artifact detector outputs
    spectral_peak_ratio: float = Field(ge=0.0, description="Upsampling grid energy in the FFT")
    residual_kurtosis: float = Field(description="Kurtosis of the high-pass residual")
    color_corr_anomaly: float = Field(ge=0.0, description="Deviation from natural channel correlation")
    saturation_clip_ratio: float = Field(ge=0.0, le=1.0, description="Fraction of clipped pixels")


# --------------------------------------------------------------------------
# The four streams
# --------------------------------------------------------------------------


class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    ts: datetime = Field(description="UTC timestamp of the event")
    institution_id: str = Field(description="Whose books this event lands on")


class OnboardingEvent(_EventBase):
    """An account application. One per identity, at t=0 for that account."""

    application_id: str
    identity_id: str = Field(description="Vault token, not a name")
    account_id: str

    # Tokenised PII -- shared tokens are the ring signal, the values never travel.
    name_token: str
    dob_token: str
    address_token: str
    email_token: str
    phone_token: str
    email_handle_shape: str = Field(description="Normalised handle pattern, e.g. 'aaa.9999'")

    # Observed infrastructure
    device_id: str
    ip_id: str
    asn: int

    # Declared attributes that the fairness review cares about staying coarse
    declared_age: int = Field(ge=18, le=100)
    credit_file_age_months: int = Field(ge=0)
    address_shared_count: int = Field(ge=0, description="Applicants seen at this address")

    # Similarity tags -- never embeddings
    face_tag: SparseTag
    doc_template_tag: SparseTag
    identity_hypervector: SparseTag

    signals: VerificationSignals


class AuthEvent(_EventBase):
    """A single authorisation attempt."""

    merchant_id: str
    merchant_category: str
    account_id: str
    identity_id: str

    card_token: str = Field(description="Tokenised PAN")
    pan_suffix6: str = Field(min_length=6, max_length=6, description="Test-range digits, for entropy features")
    amount: float = Field(ge=0.0)
    currency: str = "USD"
    is_zero_auth: bool = False

    response_code: AuthResponseCode
    avs_result: AvsResult
    cvv_result: CvvResult

    device_id: str
    ip_id: str
    asn: int
    session_id: str

    @property
    def approved(self) -> bool:
        return self.response_code is AuthResponseCode.APPROVED


class SessionTelemetry(_EventBase):
    """Behavioural telemetry for a session. Bots are too regular; that is the tell."""

    session_id: str
    account_id: str
    device_id: str
    ip_id: str
    asn: int
    user_agent_hash: str

    typing_cadence_cv: float = Field(ge=0.0, description="Coeff. of variation of keystroke intervals")
    mouse_entropy: float = Field(ge=0.0)
    page_dwell_ms: float = Field(ge=0.0)
    automation_score: float = Field(ge=0.0, le=1.0)
    tz_offset_min: int
    screen_res: str


class Label(_EventBase):
    """A label that arrives late, as labels do.

    ``ts`` is when the label became available and equals ``label_available_at``;
    it is duplicated so that a naive sort by ``ts`` is still leakage-safe.
    """

    subject_type: str = Field(description="'auth_event' or 'identity'")
    subject_id: str
    is_fraud: bool
    source: LabelSource
    event_ts: datetime = Field(description="When the labelled event actually happened")
    label_available_at: datetime = Field(description="When we learn it -- train on this")

    @field_validator("label_available_at")
    @classmethod
    def _not_before_event(cls, v: datetime, info) -> datetime:
        event_ts = info.data.get("event_ts")
        if event_ts is not None and v < event_ts:
            raise ValueError("label_available_at cannot precede event_ts")
        return v

    @property
    def delay_days(self) -> float:
        return (self.label_available_at - self.event_ts).total_seconds() / 86400.0


class GroundTruth(BaseModel):
    """EVAL ONLY. Never import this into a feature or training path.

    ``detect/models/train.py`` refuses to run if this module's fields appear in
    a feature frame.
    """

    model_config = ConfigDict(frozen=True)

    identity_id: str
    account_id: str
    is_synthetic: bool
    ring_id: str | None = None
    ring_preset: str | None = None
    is_lookalike: bool = False
    first_fraud_ts: datetime | None = None
    onboarded_ts: datetime


#: Field names that must never reach a model. Enforced in training.
EVAL_ONLY_FIELDS: frozenset[str] = frozenset(
    {"is_synthetic", "ring_id", "ring_preset", "is_lookalike", "first_fraud_ts"}
)

#: Stream name -> model, used by the simulator writer and the ingest reader.
STREAM_MODELS: dict[str, type[BaseModel]] = {
    "onboarding_events": OnboardingEvent,
    "auth_events": AuthEvent,
    "session_telemetry": SessionTelemetry,
    "labels": Label,
    "ground_truth": GroundTruth,
}
