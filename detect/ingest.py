"""L1: ingest and normalise, with the PII vault boundary.

One read path for all four streams, and one rule: **values do not cross this
boundary, only tokens and tags do**. The simulator already emits tokenised
fields, which is the point -- the vault is where detokenisation would live in a
real deployment, and nothing downstream of here is capable of asking for it.

The other job of this module is *view scoping*. A merchant caller sees only its
own institution's events; a network caller sees all of them. This is enforced at
read time rather than at display time, so the merchant-vs-network comparison
measures a real difference in available evidence rather than a filtered chart.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from contracts.decisions import ViewScope
from contracts.schemas import (
    AuthEvent,
    GroundTruth,
    Label,
    OnboardingEvent,
    SessionTelemetry,
)

STREAM_FILES = {
    "onboarding_events": OnboardingEvent,
    "auth_events": AuthEvent,
    "session_telemetry": SessionTelemetry,
    "labels": Label,
}


class PIIVault:
    """Where values would live. Deliberately not populated from the event path.

    The simulator tokenises before emitting, so this class exists to make the
    boundary explicit and to fail loudly if any code tries to resolve a token
    back to a value without holding the vault. In a real deployment this is an
    HSM-backed tokenisation service; here it is a locked door with nothing
    behind it, which is the correct amount of PII for a demo to hold.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def resolve(self, token: str) -> str:
        raise PermissionError(
            f"detokenisation of {token!r} is not available to the detection path; "
            "only tokens and tags cross the L1 boundary"
        )

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class Dataset:
    """A loaded scenario, scoped to a view."""

    root: Path
    view: ViewScope
    institution_id: str | None
    onboarding: list[OnboardingEvent]
    auth: list[AuthEvent]
    telemetry: list[SessionTelemetry]
    labels: list[Label]
    ground_truth: list[GroundTruth]

    @property
    def n_events(self) -> int:
        return len(self.onboarding) + len(self.auth) + len(self.telemetry)

    def summary(self) -> str:
        scope = (
            f"{self.view.value}"
            if self.institution_id is None
            else f"{self.view.value}:{self.institution_id}"
        )
        return (
            f"{self.root.name} [{scope}] "
            f"onboarding={len(self.onboarding):,} auth={len(self.auth):,} "
            f"telemetry={len(self.telemetry):,} labels={len(self.labels):,}"
        )


def iter_stream(path: Path, model, limit: int | None = None) -> Iterator:
    """Parse a JSONL stream into contract models, one at a time."""
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            yield model.model_validate_json(line)


def load(
    root: str | Path,
    view: ViewScope = ViewScope.NETWORK,
    institution_id: str | None = None,
    limit: int | None = None,
) -> Dataset:
    """Load a scenario directory under a view scope.

    A ``MERCHANT`` or ``ISSUER`` view requires an ``institution_id`` and returns
    only that institution's slice. That is what makes the network-view delta a
    measurement: the same detector genuinely has less to work with.
    """
    root = Path(root)
    if view is not ViewScope.NETWORK and institution_id is None:
        raise ValueError(f"view {view.value!r} requires an institution_id")

    def scoped(records: list) -> list:
        if view is ViewScope.NETWORK:
            return records
        return [r for r in records if r.institution_id == institution_id]

    onboarding = scoped(list(iter_stream(root / "onboarding_events.jsonl", OnboardingEvent, limit)))
    auth = scoped(list(iter_stream(root / "auth_events.jsonl", AuthEvent, limit)))
    telemetry = scoped(list(iter_stream(root / "session_telemetry.jsonl", SessionTelemetry, limit)))
    labels = scoped(list(iter_stream(root / "labels.jsonl", Label, limit)))

    # Ground truth is never scoped -- it is for evaluation only, and evaluation
    # happens outside the view boundary by definition.
    truth = list(iter_stream(root / "ground_truth.jsonl", GroundTruth))

    return Dataset(
        root=root,
        view=view,
        institution_id=institution_id,
        onboarding=onboarding,
        auth=auth,
        telemetry=telemetry,
        labels=labels,
        ground_truth=truth,
    )


def institutions_in(root: str | Path) -> list[str]:
    """Institution ids present in a scenario, for driving the view toggle."""
    root = Path(root)
    seen: set[str] = set()
    with (root / "onboarding_events.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            seen.add(json.loads(line)["institution_id"])
    return sorted(seen)
