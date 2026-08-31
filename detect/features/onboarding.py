"""L4 inputs: features available at t=0, when an account is opened.

This is the identity half of the seam. Everything here is knowable at the moment
of application -- no behaviour, because there is none yet. That constraint is
what makes the layer useful: it scores an account before it has done anything,
which is the only way to catch a ring before it acts.

Three families:

* **Verification signals** -- what a vendor reports, plus the four measured
  GAN-artifact statistics from ``biohash/artifacts.py``.
* **Classic synthetic-identity tells** -- credit-file age inconsistent with
  declared age, thin file, mail-drop address, generated-looking email handle.
* **Graph features at t=0** -- how many other applicants share this device,
  address, phone; how many near-duplicate tags; whether the shared
  infrastructure crosses institutions.

**Fairness note.** Thin-file customers are disproportionately young, migrant or
low-income and are *legitimately* thin-file, so ``credit_file_age_months`` alone
must never drive a decline. It is included as a ratio against declared age --
the tell is the *inconsistency*, a 45-year-old with a four-month file, not
youth. The metrics report breaks out decline rates on the thin-file cohort for
exactly this reason.
"""

from __future__ import annotations

from collections import defaultdict

from contracts.schemas import OnboardingEvent
from detect.graph.entity_res import NEAR_DUPLICATE_THRESHOLDS


def _handle_shape_stats(events: list[OnboardingEvent]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        counts[ev.email_handle_shape] += 1
    return counts


def _token_counts(events: list[OnboardingEvent]) -> dict[str, dict[str, int]]:
    fields = ("device_id", "address_token", "phone_token", "dob_token", "ip_id")
    out: dict[str, dict[str, int]] = {f: defaultdict(int) for f in fields}
    for ev in events:
        for f in fields:
            out[f][getattr(ev, f)] += 1
    return out


def _name_variety(events: list[OnboardingEvent]) -> dict[str, set[str]]:
    """Names seen alongside each DOB and phone token -- the recombination tell."""
    out: dict[str, set[str]] = defaultdict(set)
    for ev in events:
        out[f"dob:{ev.dob_token}"].add(ev.name_token)
        out[f"phone:{ev.phone_token}"].add(ev.name_token)
    return out


class OnboardingFeatureBuilder:
    """Builds t=0 feature rows for a cohort of applications.

    Population statistics (how many share a device, how common a handle shape
    is) are computed across the cohort passed in. In serving this is the
    institution's existing book; here it is the scenario.
    """

    def __init__(self, events: list[OnboardingEvent]) -> None:
        self.events = events
        self._handle_counts = _handle_shape_stats(events)
        self._tokens = _token_counts(events)
        self._names = _name_variety(events)
        self._tag_neighbours = self._count_tag_neighbours(events)

    def _count_tag_neighbours(self, events: list[OnboardingEvent]) -> dict[str, dict[str, int]]:
        """How many other applicants each identity is tag-near-duplicate with.

        Uses the same exact-overlap path and per-attribute thresholds as entity
        resolution, so the feature and the graph edge cannot disagree.
        """
        from detect.graph.entity_res import near_duplicate_tag_links

        out: dict[str, dict[str, int]] = {
            attr: defaultdict(int) for attr in NEAR_DUPLICATE_THRESHOLDS
        }
        for attr in NEAR_DUPLICATE_THRESHOLDS:
            for link in near_duplicate_tag_links(events, attr):
                out[attr][link.left] += 1
                out[attr][link.right] += 1
        return out

    def row(self, ev: OnboardingEvent) -> dict[str, float]:
        s = ev.signals
        f: dict[str, float] = {
            # Vendor verification signals
            "template_match_score": s.template_match_score,
            "exif_consistency": s.exif_consistency,
            "liveness_score": s.liveness_score,
            # Measured generative-image statistics
            "spectral_peak_ratio": s.spectral_peak_ratio,
            "residual_kurtosis": s.residual_kurtosis,
            "color_corr_anomaly": s.color_corr_anomaly,
            "saturation_clip_ratio": s.saturation_clip_ratio,
        }

        # -- classic synthetic-identity tells ------------------------------
        plausible_months = max(1, (ev.declared_age - 18) * 12)
        f["credit_file_age_months"] = float(ev.credit_file_age_months)
        # The tell is the inconsistency, not youth. A 45-year-old with a
        # four-month credit file is odd; a 19-year-old with one is normal.
        f["credit_age_ratio"] = float(ev.credit_file_age_months) / plausible_months
        f["declared_age"] = float(ev.declared_age)
        f["is_thin_file"] = float(ev.credit_file_age_months < 12)

        f["address_shared_count"] = float(ev.address_shared_count)
        f["email_shape_frequency"] = float(self._handle_counts.get(ev.email_handle_shape, 1))

        for field in ("device_id", "address_token", "phone_token", "dob_token", "ip_id"):
            f[f"shared_{field}_count"] = float(self._tokens[field][getattr(ev, field)])

        # PII recombination: one person's details under more than one name.
        f["dob_name_variety"] = float(len(self._names[f"dob:{ev.dob_token}"]))
        f["phone_name_variety"] = float(len(self._names[f"phone:{ev.phone_token}"]))

        # -- tag neighbourhood --------------------------------------------
        for attr in NEAR_DUPLICATE_THRESHOLDS:
            f[f"near_dup_{attr}_count"] = float(self._tag_neighbours[attr].get(ev.identity_id, 0))

        f["tag_sparsity"] = ev.face_tag.sparsity
        return f

    def build(self) -> tuple[list[str], list[dict[str, float]]]:
        rows = [self.row(ev) for ev in self.events]
        names = sorted(rows[0].keys()) if rows else []
        return names, rows


def feature_names(events: list[OnboardingEvent]) -> list[str]:
    return OnboardingFeatureBuilder(events[:1]).build()[0] if events else []
