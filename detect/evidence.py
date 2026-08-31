"""Why a ring is a ring: the evidence, laid out as the supply chain.

The console was showing the *conclusion* -- here is a community of 36 identities
-- and none of the reasoning. An analyst cannot act on a coloured graph, and a
judge cannot tell it apart from any other clustering demo. The signals that
actually make the case were all being computed and none of them reached a
screen.

This assembles them into the three stages the design doc describes, so the
console can show the chain rather than the cluster:

* **Manufacture** -- the applicant's face and document. GAN-artifact statistics
  from ``biohash/artifacts.py``, vendor verification scores, and how many ring
  members carry near-duplicate face tags or a shared document template.
* **Onboard** -- how the identity got through KYC. Thin credit file against
  declared age, addresses shared with other applicants, and PII recombination:
  the same date of birth or phone appearing under different names.
* **Weaponise** -- the card testing itself. Distinct PANs per account, decline
  ratio, CVV and AVS mismatch rates, low-ticket and zero-auth share, PAN
  numeric entropy, and how many merchants the testing was spread across.

Every figure is paired with the same figure for the legitimate population, so
the screen shows a contrast rather than a number. A decline ratio of 0.82 means
nothing on its own; next to a population ratio of 0.04 it is the whole story.

All of this is off the authorisation path. The index is built once, when the
graph is built, and read from memory afterwards.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

#: Below this, an authorisation is a "low ticket" -- the amount an attacker uses
#: to check a card is live without attracting attention.
LOW_TICKET = 5.0

#: Which way each metric points, and what it means in one line. Without the
#: direction the screen is misleading: a liveness score *below* the population
#: is the suspicious case, and a ratio of 0.86x rendered as a bar looks
#: reassuring when it is the opposite.
METRICS: dict[str, tuple[str, str]] = {
    # manufacture
    "spectral_peak_ratio": ("higher", "Upsampling grid energy in the FFT - a generator fingerprint"),
    "residual_kurtosis": ("higher", "Kurtosis of the high-pass residual"),
    "color_corr_anomaly": ("higher", "Deviation from natural channel correlation"),
    "saturation_clip_ratio": ("higher", "Fraction of clipped pixels"),
    "template_match_score": ("lower", "Vendor confidence the document matches its template"),
    "exif_consistency": ("lower", "Camera metadata consistent with the image"),
    "liveness_score": ("lower", "Vendor liveness check on the selfie"),
    # onboard
    "thin_file_share": ("higher", "Members with under 12 months of credit file"),
    "credit_age_ratio": ("lower", "Credit file age against the maximum plausible for the declared age"),
    "declared_age": ("neutral", "Mean declared age"),
    "address_shared_count": ("higher", "Other applicants seen at the same address"),
    "pii_recombination_share": ("higher", "Members whose DOB, phone or address appears under another name"),
    "shared_device_share": ("higher", "Members on a device another applicant also used"),
    "shared_address_share": ("higher", "Members at an address another applicant also used"),
    "shared_phone_share": ("higher", "Members on a phone another applicant also used"),
    # weaponise
    "accounts_transacting": ("count", "Members that have authorised at all"),
    "attempts": ("count", "Authorisation attempts in total"),
    "distinct_pans_per_account": ("higher", "Distinct card numbers tried per account"),
    "decline_ratio": ("higher", "Share of attempts declined"),
    "cvv_mismatch_rate": ("higher", "Share of attempts failing CVV"),
    "avs_mismatch_rate": ("higher", "Share of attempts failing address verification"),
    "zero_auth_ratio": ("higher", "Share of zero-value authorisations - liveness probing"),
    "low_ticket_ratio": ("higher", f"Share of attempts under {LOW_TICKET:.0f} units"),
    "pan_digit_entropy": ("lower", "Digit entropy of attempted PANs - enumeration is low"),
    # Deliberately neutral: a sloppy ring concentrates on a narrow merchant set,
    # a sophisticated one sprays across many. Both readings are real, so the
    # number is shown as context rather than scored in a direction.
    "merchants_per_account": ("neutral", "Merchants each account spread its testing across"),
    "peak_attempts_per_hour": ("higher", "Attempt rate across the span they were made in"),
}


@dataclass
class IdentityRollup:
    """Everything one identity did, aggregated once."""

    attempts: int = 0
    declines: int = 0
    cvv_mismatch: int = 0
    avs_mismatch: int = 0
    zero_auth: int = 0
    low_ticket: int = 0
    cards: set = field(default_factory=set)
    merchants: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    suffixes: list = field(default_factory=list)
    first_ts: object = None
    last_ts: object = None

    @property
    def peak_hourly_rate(self) -> float:
        """Attempts per hour across the span they were made in.

        A blunt measure, but the one that separates a burst from a customer:
        forty attempts in an afternoon is a different object from forty
        attempts across three months.
        """
        if self.first_ts is None or self.last_ts is None or self.attempts < 2:
            return 0.0
        hours = max((self.last_ts - self.first_ts).total_seconds() / 3600.0, 1 / 60)
        return self.attempts / hours


def digit_entropy(suffixes: list[str]) -> float:
    """Normalised Shannon entropy of the digits in attempted PANs.

    Sequential enumeration -- ...0001, ...0002, ...0003 -- has low entropy in
    its trailing digits. A real customer's cards do not.
    """
    digits = [c for s in suffixes for c in s if c.isdigit()]
    if len(digits) < 4:
        return 1.0
    counts = Counter(digits)
    total = len(digits)
    h = -sum((n / total) * math.log2(n / total) for n in counts.values())
    return h / math.log2(10)


class EvidenceIndex:
    """One pass over the streams, then every ring is a dictionary lookup."""

    def __init__(self, onboarding: list, auth: list, links: list | None = None) -> None:
        self.onboarding = {ev.identity_id: ev for ev in onboarding}
        self.rollup: dict[str, IdentityRollup] = defaultdict(IdentityRollup)

        for ev in auth:
            r = self.rollup[ev.identity_id]
            r.attempts += 1
            r.declines += 0 if ev.approved else 1
            r.cvv_mismatch += 1 if ev.cvv_result.value != "match" else 0
            r.avs_mismatch += 1 if ev.avs_result.value != "match" else 0
            r.zero_auth += 1 if ev.is_zero_auth else 0
            r.low_ticket += 1 if ev.amount < LOW_TICKET else 0
            r.cards.add(ev.card_token)
            r.merchants.add(ev.merchant_id)
            r.devices.add(ev.device_id)
            if len(r.suffixes) < 400:
                r.suffixes.append(ev.pan_suffix6)
            r.first_ts = ev.ts if r.first_ts is None else min(r.first_ts, ev.ts)
            r.last_ts = ev.ts if r.last_ts is None else max(r.last_ts, ev.ts)

        # Entity-resolution links, indexed both ways so a ring can be asked
        # which of its members are linked to each other and by what.
        self.links_by_identity: dict[str, list] = defaultdict(list)
        for link in links or []:
            self.links_by_identity[link.left].append(link)
            self.links_by_identity[link.right].append(link)

        # Shared-token counts across the whole population: an address used by
        # nine applicants is only interesting because most are used by one.
        self.token_counts: dict[str, Counter] = {
            field_name: Counter(getattr(ev, field_name) for ev in onboarding)
            for field_name in ("device_id", "address_token", "phone_token", "dob_token", "ip_id")
        }
        # PII recombination: one person's details under more than one name.
        self.names_per_token: dict[str, dict[str, set]] = {
            f: defaultdict(set) for f in ("dob_token", "phone_token", "address_token")
        }
        for ev in onboarding:
            for f in self.names_per_token:
                self.names_per_token[f][getattr(ev, f)].add(ev.name_token)

        self.population = self._aggregate(list(self.onboarding))

    # ------------------------------------------------------------------
    def _aggregate(self, identity_ids: list[str]) -> dict:
        ids = [i for i in identity_ids if i in self.onboarding]
        if not ids:
            return {}

        events = [self.onboarding[i] for i in ids]
        sig = [e.signals for e in events]

        def mean(values) -> float:
            values = [v for v in values if v is not None]
            return float(sum(values) / len(values)) if values else 0.0

        # -- manufacture ------------------------------------------------
        manufacture = {
            "spectral_peak_ratio": mean(s.spectral_peak_ratio for s in sig),
            "residual_kurtosis": mean(s.residual_kurtosis for s in sig),
            "color_corr_anomaly": mean(s.color_corr_anomaly for s in sig),
            "saturation_clip_ratio": mean(s.saturation_clip_ratio for s in sig),
            "template_match_score": mean(s.template_match_score for s in sig),
            "exif_consistency": mean(s.exif_consistency for s in sig),
            "liveness_score": mean(s.liveness_score for s in sig),
        }

        # -- onboard ----------------------------------------------------
        thin = sum(1 for e in events if e.credit_file_age_months < 12)
        credit_ratio = mean(
            e.credit_file_age_months / max(1, (e.declared_age - 18) * 12) for e in events
        )
        recombination = 0
        for e in events:
            for f in self.names_per_token:
                if len(self.names_per_token[f][getattr(e, f)]) > 1:
                    recombination += 1
                    break

        onboard = {
            "thin_file_share": thin / len(events),
            "credit_age_ratio": credit_ratio,
            "declared_age": mean(e.declared_age for e in events),
            "address_shared_count": mean(e.address_shared_count for e in events),
            "pii_recombination_share": recombination / len(events),
            "shared_device_share": self._shared_share(events, "device_id"),
            "shared_address_share": self._shared_share(events, "address_token"),
            "shared_phone_share": self._shared_share(events, "phone_token"),
        }

        # -- weaponise --------------------------------------------------
        rolls = [self.rollup[i] for i in ids if self.rollup[i].attempts > 0]
        if rolls:
            total_attempts = sum(r.attempts for r in rolls)
            weaponise = {
                "accounts_transacting": len(rolls),
                "attempts": total_attempts,
                "distinct_pans_per_account": mean(len(r.cards) for r in rolls),
                "decline_ratio": sum(r.declines for r in rolls) / total_attempts,
                "cvv_mismatch_rate": sum(r.cvv_mismatch for r in rolls) / total_attempts,
                "avs_mismatch_rate": sum(r.avs_mismatch for r in rolls) / total_attempts,
                "zero_auth_ratio": sum(r.zero_auth for r in rolls) / total_attempts,
                "low_ticket_ratio": sum(r.low_ticket for r in rolls) / total_attempts,
                "pan_digit_entropy": mean(digit_entropy(r.suffixes) for r in rolls),
                "merchants_per_account": mean(len(r.merchants) for r in rolls),
                "peak_attempts_per_hour": mean(r.peak_hourly_rate for r in rolls),
            }
        else:
            weaponise = {"accounts_transacting": 0, "attempts": 0}

        return {"manufacture": manufacture, "onboard": onboard, "weaponise": weaponise}

    def _shared_share(self, events: list, field_name: str) -> float:
        """Fraction of members whose token is used by more than one applicant."""
        counts = self.token_counts[field_name]
        shared = sum(1 for e in events if counts[getattr(e, field_name)] > 1)
        return shared / len(events)

    # ------------------------------------------------------------------
    def link_summary(self, identity_ids: list[str]) -> list[dict]:
        """Entity-resolution links whose *both* ends are inside the ring.

        A link to somewhere outside says nothing about why these accounts
        belong together, so only internal ones are counted.
        """
        members = set(identity_ids)
        seen: set = set()
        counts: Counter = Counter()
        for i in identity_ids:
            for link in self.links_by_identity.get(i, ()):
                if link.left in members and link.right in members:
                    pair = (min(link.left, link.right), max(link.left, link.right), link.kind)
                    if pair not in seen:
                        seen.add(pair)
                        counts[link.kind.split(":")[0]] += 1
        return [{"kind": k, "pairs": n} for k, n in counts.most_common()]

    def for_ring(self, identity_ids: list[str]) -> dict:
        """Ring figures beside everyone-else figures, ready to render.

        The baseline excludes this ring's own members. Comparing a ring against
        a population that contains it pulls the baseline toward the ring and
        understates every contrast -- and on a small population it understates
        it a lot.
        """
        ring = self._aggregate(identity_ids)
        if not ring:
            return {}

        members = set(identity_ids)
        rest = [i for i in self.onboarding if i not in members]
        baseline = self._aggregate(rest) if rest else self.population

        out: dict = {"members": len(identity_ids), "links": self.link_summary(identity_ids)}
        for stage in ("manufacture", "onboard", "weaponise"):
            rows = []
            for metric, value in ring.get(stage, {}).items():
                direction, description = METRICS.get(metric, ("higher", ""))
                base = baseline.get(stage, {}).get(metric)
                # An absolute count has no meaningful ratio against a
                # population of a different size, so it is shown as context.
                ratio = None
                if direction != "count" and base:
                    ratio = value / base
                rows.append(
                    {
                        "metric": metric,
                        "label": metric.replace("_", " "),
                        "description": description,
                        "direction": direction,
                        "ring": value,
                        "population": base,
                        "ratio": ratio,
                        # True when the ring sits on the suspicious side of the
                        # rest of the population for this metric.
                        # Only a directional metric can be "elevated". A count
                        # has no baseline to beat, and a neutral metric is
                        # context -- flagging it red would be asserting a
                        # direction the metric does not have.
                        "elevated": (
                            (value > base if direction == "higher" else value < base)
                            if base is not None and direction in ("higher", "lower")
                            else None
                        ),
                    }
                )
            out[stage] = rows
        return out
