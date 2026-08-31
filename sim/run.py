"""Simulator entry point: builds a world, runs it, writes four JSONL streams.

Usage::

    python -m sim.run --scenario sophisticated
    python -m sim.run --scenario moderate --out data/moderate

Everything is driven by a seed plus a scenario file, so a run reproduces
exactly. ``verify.py`` checks the output against the design doc's constraints.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from contracts.schemas import (
    AuthEvent,
    AuthResponseCode,
    AvsResult,
    CvvResult,
    GroundTruth,
    Label,
    LabelSource,
    OnboardingEvent,
    SessionTelemetry,
)
from sim.cardtest import AuthAttempt, CardTestGenerator
from sim.config import ScenarioConfig, load_scenario
from sim.lookalikes import LookalikeGenerator
from sim.population import Customer, PopulationGenerator, seasonality
from sim.rings import Ring, RingGenerator
from sim.world import Identity, World


class Simulator:
    def __init__(self, cfg: ScenarioConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = datetime.fromisoformat(cfg.start_date).replace(tzinfo=timezone.utc)
        self.world = World(cfg, self.rng)
        self.population_gen = PopulationGenerator(self.world, self.rng)
        self.ring_gen = RingGenerator(self.world, self.rng)
        self.cardtest = CardTestGenerator(self.world, self.rng)
        self._event_seq = 0

    # -- helpers -----------------------------------------------------------

    def _ts(self, day: float) -> datetime:
        return self.start + timedelta(days=float(day))

    def _eid(self, prefix: str) -> str:
        self._event_seq += 1
        return f"{prefix}_{self._event_seq:09d}"

    # -- build -------------------------------------------------------------

    def build(self) -> tuple[list[Customer], list[Customer], list[Ring]]:
        pc = self.cfg.population
        t0 = time.time()
        customers = self.population_gen.build(pc.n_customers, start_index=0)
        idx = pc.n_customers
        lookalikes, idx = LookalikeGenerator(self.world, self.population_gen, self.rng).build(idx)
        rings, idx = self.ring_gen.build(idx)
        n_ring = sum(len(r.members) for r in rings)
        print(
            f"  built {len(customers)} customers, {len(lookalikes)} look-alikes, "
            f"{len(rings)} rings ({n_ring} accounts) in {time.time()-t0:.1f}s"
        )
        return customers, lookalikes, rings

    # -- streams -----------------------------------------------------------

    def onboarding_event(self, identity: Identity) -> OnboardingEvent:
        return OnboardingEvent(
            event_id=self._eid("onb"),
            ts=self._ts(max(identity.onboarded_day, -3650)),
            institution_id=identity.institution_id,
            application_id=f"app_{identity.identity_id[4:]}",
            identity_id=identity.identity_id,
            account_id=identity.account_id,
            name_token=identity.name_token,
            dob_token=identity.dob_token,
            address_token=identity.address_token,
            email_token=identity.email_token,
            phone_token=identity.phone_token,
            email_handle_shape=identity.email_handle_shape,
            device_id=identity.device_ids[0],
            ip_id=identity.ip_id,
            asn=identity.asn,
            declared_age=identity.declared_age,
            credit_file_age_months=identity.credit_file_age_months,
            address_shared_count=self.world.address_shared_count(identity.address_token),
            face_tag=identity.face_tag,
            doc_template_tag=identity.doc_template_tag,
            identity_hypervector=identity.hypervector,
            signals=identity.signals,
        )

    def legit_attempts(self, cust: Customer) -> list[AuthAttempt]:
        """Ordinary transactions for one legitimate customer."""
        rng = self.rng
        out: list[AuthAttempt] = []
        times = self.population_gen.transaction_times(cust, self.cfg.days)
        session_day = -99.0
        session_id = ""
        for t in times:
            device, ip_id, asn = cust.state_at(float(t))
            if t - session_day > 0.05:
                session_day = float(t)
                session_id = f"ses_{int(rng.integers(0, 2**48)):012x}"
            merchant = self.population_gen.pick_merchant(cust)
            card_token, suffix = cust.card_at(float(t), rng)
            approved = rng.random() > merchant.base_decline_rate
            out.append(
                AuthAttempt(
                    day=float(t),
                    account_id=cust.identity.account_id,
                    identity_id=cust.identity.identity_id,
                    institution_id=cust.identity.institution_id,
                    merchant=merchant,
                    card_token=card_token,
                    pan_suffix6=suffix,
                    amount=self.population_gen.amount(cust),
                    is_zero_auth=False,
                    response_code=(
                        AuthResponseCode.APPROVED
                        if approved
                        else AuthResponseCode.DECLINED_INSUFFICIENT_FUNDS
                    ),
                    avs_result=AvsResult.MATCH if rng.random() < 0.94 else AvsResult.PARTIAL,
                    cvv_result=CvvResult.MATCH if rng.random() < 0.97 else CvvResult.NO_MATCH,
                    device_id=device,
                    ip_id=ip_id,
                    asn=asn,
                    session_id=session_id,
                    is_fraud=False,
                )
            )
        return out

    def auth_event(self, a: AuthAttempt) -> AuthEvent:
        return AuthEvent(
            event_id=self._eid("ath"),
            ts=self._ts(a.day),
            institution_id=a.institution_id,
            merchant_id=a.merchant.merchant_id,
            merchant_category=a.merchant.category,
            account_id=a.account_id,
            identity_id=a.identity_id,
            card_token=a.card_token,
            pan_suffix6=a.pan_suffix6,
            amount=round(a.amount, 2),
            is_zero_auth=a.is_zero_auth,
            response_code=a.response_code,
            avs_result=a.avs_result,
            cvv_result=a.cvv_result,
            device_id=a.device_id,
            ip_id=a.ip_id,
            asn=a.asn,
            session_id=a.session_id,
        )

    def telemetry_for(self, a: AuthAttempt) -> SessionTelemetry:
        """One telemetry record for a session.

        Emitted for the first attempt seen in each session during the streaming
        write pass, so telemetry comes out already in time order and no second
        list has to be held in memory.

        Scripted sessions are *too regular*: near-constant keystroke intervals,
        little mouse movement, short dwell. That regularity is the tell, and it
        is why the legitimate distributions here are deliberately wide.
        """
        rng = self.rng
        if a.is_bot:
            cadence_cv = float(np.clip(rng.normal(0.12, 0.06), 0.01, 1.5))
            mouse = float(np.clip(rng.normal(0.6, 0.3), 0.0, 5.0))
            dwell = float(np.clip(rng.normal(900, 400), 50, 60000))
            automation = float(np.clip(rng.beta(5.5, 2.2), 0, 1))
        else:
            cadence_cv = float(np.clip(rng.normal(0.62, 0.18), 0.01, 1.5))
            mouse = float(np.clip(rng.normal(2.6, 0.8), 0.0, 5.0))
            dwell = float(np.clip(rng.normal(9000, 4500), 50, 60000))
            automation = float(np.clip(rng.beta(1.6, 7.0), 0, 1))

        return SessionTelemetry(
            event_id=self._eid("tel"),
            ts=self._ts(a.day),
            institution_id=a.institution_id,
            session_id=a.session_id,
            account_id=a.account_id,
            device_id=a.device_id,
            ip_id=a.ip_id,
            asn=a.asn,
            user_agent_hash=f"ua_{int(rng.integers(0, 2**32)):08x}",
            typing_cadence_cv=cadence_cv,
            mouse_entropy=mouse,
            page_dwell_ms=dwell,
            automation_score=automation,
            tz_offset_min=int(rng.choice([-480, -300, -120, 0, 60, 120, 330, 480])),
            screen_res=str(rng.choice(["1920x1080", "1366x768", "2560x1440", "390x844"])),
        )

    def labels_for(self, attempt: AuthAttempt, event_id: str) -> Label | None:
        """Emit a late, incomplete label -- the only kind that actually exists.

        Chargebacks arrive 20-60 days after the event and cover only part of the
        fraud. A small share of fraud is caught sooner by analyst review, and a
        few legitimate transactions are charged back in error.
        """
        rng = self.rng
        lc = self.cfg.labels
        event_ts = self._ts(attempt.day)

        if attempt.is_fraud:
            if rng.random() < lc.analyst_review_share:
                avail = event_ts + timedelta(days=float(rng.uniform(1, lc.analyst_delay_days)))
                source = LabelSource.ANALYST
            elif rng.random() < lc.chargeback_coverage:
                avail = event_ts + timedelta(
                    days=float(rng.uniform(lc.chargeback_delay_min_days, lc.chargeback_delay_max_days))
                )
                source = LabelSource.CHARGEBACK
            else:
                return None  # never labelled; the model must live with this
            is_fraud = True
        else:
            if rng.random() >= lc.false_chargeback_rate:
                return None
            avail = event_ts + timedelta(
                days=float(rng.uniform(lc.chargeback_delay_min_days, lc.chargeback_delay_max_days))
            )
            source = LabelSource.CHARGEBACK
            is_fraud = True  # disputed, though the transaction was genuine

        return Label(
            event_id=self._eid("lbl"),
            ts=avail,
            institution_id=attempt.institution_id,
            subject_type="auth_event",
            subject_id=event_id,
            is_fraud=is_fraud,
            source=source,
            event_ts=event_ts,
            label_available_at=avail,
        )

    # -- run ---------------------------------------------------------------

    def run(self, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        customers, lookalikes, rings = self.build()

        t0 = time.time()
        onboarding: list[OnboardingEvent] = []
        attempts: list[AuthAttempt] = []
        truth: list[GroundTruth] = []

        # -- legitimate population and look-alikes
        for cust in customers + lookalikes:
            onboarding.append(self.onboarding_event(cust.identity))
            attempts.extend(self.legit_attempts(cust))
            truth.append(
                GroundTruth(
                    identity_id=cust.identity.identity_id,
                    account_id=cust.identity.account_id,
                    is_synthetic=False,
                    ring_id=None,
                    ring_preset=None,
                    is_lookalike=cust.is_lookalike,
                    first_fraud_ts=None,
                    onboarded_ts=self._ts(max(cust.identity.onboarded_day, -3650)),
                )
            )

        # -- rings
        for ring in rings:
            for member in ring.members:
                ident = member.identity
                onboarding.append(self.onboarding_event(ident))

                warm = self.cardtest.warmup(
                    member,
                    until_day=member.activation_day
                    if member.activation_day is not None
                    else self.cfg.days,
                )
                burst = self.cardtest.generate(ring, member)
                attempts.extend(warm)
                attempts.extend(burst)

                first_fraud = min((a.day for a in burst), default=None)
                truth.append(
                    GroundTruth(
                        identity_id=ident.identity_id,
                        account_id=ident.account_id,
                        is_synthetic=True,
                        ring_id=ring.ring_id,
                        ring_preset=ring.profile_name,
                        is_lookalike=False,
                        first_fraud_ts=self._ts(first_fraud) if first_fraud is not None else None,
                        onboarded_ts=self._ts(ident.onboarded_day),
                    )
                )

        # -- serialise, in time order.
        # Auth events are streamed straight to disk rather than accumulated:
        # at full scale there are hundreds of thousands of them, and holding
        # that many pydantic models resident costs gigabytes for no benefit.
        # The lightweight AuthAttempt dataclasses are what get sorted.
        attempts.sort(key=lambda a: a.day)
        labels: list[Label] = []
        n_auth = n_tel = 0
        seen_sessions: set[str] = set()
        auth_path = out_dir / "auth_events.jsonl"
        tel_path = out_dir / "session_telemetry.jsonl"
        with auth_path.open("w", encoding="utf-8") as fa, tel_path.open("w", encoding="utf-8") as ft:
            for a in attempts:
                ev = self.auth_event(a)
                fa.write(ev.model_dump_json() + "\n")
                n_auth += 1
                # First attempt in a session carries its telemetry. Emitting here
                # keeps telemetry in time order for free and avoids holding a
                # second large list.
                if a.session_id not in seen_sessions:
                    seen_sessions.add(a.session_id)
                    ft.write(self.telemetry_for(a).model_dump_json() + "\n")
                    n_tel += 1
                lab = self.labels_for(a, ev.event_id)
                if lab is not None:
                    labels.append(lab)

        onboarding.sort(key=lambda e: e.ts)
        labels.sort(key=lambda e: e.label_available_at)

        counts = {
            "onboarding_events": _write(out_dir / "onboarding_events.jsonl", onboarding),
            "auth_events": n_auth,
            "session_telemetry": n_tel,
            "labels": _write(out_dir / "labels.jsonl", labels),
            "ground_truth": _write(out_dir / "ground_truth.jsonl", truth),
        }

        n_fraud = sum(1 for a in attempts if a.is_fraud)
        fraud_rate = n_fraud / max(1, len(attempts))
        meta = {
            "scenario": self.cfg.name,
            "seed": self.cfg.seed,
            "days": self.cfg.days,
            "start_date": self.cfg.start_date,
            "counts": counts,
            "n_fraud_events": n_fraud,
            "fraud_rate": fraud_rate,
            "n_rings": len(rings),
            "ring_sizes": [len(r.members) for r in rings],
            "dormant_per_ring": [len(r.dormant_members) for r in rings],
            "face_source": self.world.face_source.kind,
            "config": self.cfg.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - t0, 1),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        print(f"  wrote {sum(counts.values())} records to {out_dir} in {time.time()-t0:.1f}s")
        if not (self.cfg.target_fraud_rate_min <= fraud_rate <= self.cfg.target_fraud_rate_max):
            print(
                f"  WARNING fraud rate {fraud_rate:.4%} outside target band "
                f"{self.cfg.target_fraud_rate_min:.2%}-{self.cfg.target_fraud_rate_max:.2%}"
            )
        return meta


def _write(path: Path, records: list) -> int:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic payments-fraud simulator")
    ap.add_argument("--scenario", default="moderate", help="scenario name or path to YAML")
    ap.add_argument("--out", default=None, help="output directory (default data/<scenario>)")
    ap.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    args = ap.parse_args()

    cfg = load_scenario(args.scenario)
    if args.seed is not None:
        cfg.seed = args.seed
    out = Path(args.out) if args.out else Path("data") / cfg.name

    print(f"[sim] scenario={cfg.name} seed={cfg.seed} days={cfg.days}")
    meta = Simulator(cfg).run(out)
    print(
        f"[sim] fraud rate {meta['fraud_rate']:.4%} over {meta['counts']['auth_events']} auth events; "
        f"rings {meta['ring_sizes']} (dormant {meta['dormant_per_ring']})"
    )


if __name__ == "__main__":
    main()
