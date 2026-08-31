"""Card-testing traffic: the weaponisation stage of the supply chain.

The signature the design doc names is precise -- *many distinct PANs, small
amounts, high declines, short window, narrow merchant set* -- and this module
produces exactly that, with every element parameterised so a sophisticated
operator can blunt each one independently.

The interesting part is what a *good* operator does to hide: fewer attempts per
hour, spread across more merchants and more accounts, with human-looking timing
jitter, and amounts that sit inside the merchant's normal range instead of at
zero. Each of those individually defeats a single-signal rule, which is why
detection has to combine them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.schemas import AuthResponseCode, AvsResult, CvvResult
from sim.config import RingProfile
from sim.population import seasonality
from sim.rings import Ring, RingMember
from sim.world import Merchant, World


@dataclass(slots=True)
class AuthAttempt:
    """One authorisation attempt, before it becomes a contract event."""

    day: float
    account_id: str
    identity_id: str
    institution_id: str
    merchant: Merchant
    card_token: str
    pan_suffix6: str
    amount: float
    is_zero_auth: bool
    response_code: AuthResponseCode
    avs_result: AvsResult
    cvv_result: CvvResult
    device_id: str
    ip_id: str
    asn: int
    session_id: str
    is_fraud: bool
    #: Whether this session was driven by a script. Bots are too regular, and
    #: session telemetry is generated from this at write time.
    is_bot: bool = False


class CardTestGenerator:
    """Generates testing bursts for a ring's active members."""

    def __init__(self, world: World, rng: np.random.Generator) -> None:
        self.world = world
        self.rng = rng
        self.cfg = world.cfg

    def _profile_at(self, p: RingProfile, day: float) -> RingProfile:
        """Apply concept drift, if the scenario enables it.

        A ring that keeps its original tactics after being blocked is not a
        realistic adversary. Past the drift point the operator slows down,
        spreads wider, reuses fewer devices and jitters its timing -- the same
        moves a real operator makes once its burst pattern stops working.
        """
        d = self.cfg.drift
        if not d.enabled or day < self.cfg.days * d.at_fraction:
            return p
        import dataclasses

        return dataclasses.replace(
            p,
            burst_intensity=p.burst_intensity * d.burst_intensity_mult,
            merchant_spread=max(1, int(p.merchant_spread * d.merchant_spread_mult)),
            device_reuse_rate=p.device_reuse_rate * d.device_reuse_mult,
            inter_arrival_jitter=p.inter_arrival_jitter * d.inter_arrival_jitter_mult,
        )

    def burst_times(self, start_day: float, p: RingProfile) -> np.ndarray:
        """Attempt times within one burst.

        ``inter_arrival_jitter`` is the tell. At low jitter the gaps are almost
        constant -- machine-regular timing that no human produces. At high
        jitter the process approaches Poisson and looks organic, which is what
        the regularity feature has to cope with.
        """
        n = max(1, int(self.rng.poisson(p.burst_intensity * p.burst_hours)))
        mean_gap = p.burst_hours / 24.0 / n
        if p.inter_arrival_jitter <= 0.02:
            gaps = np.full(n, mean_gap)
        else:
            # Gamma with shape 1/jitter^2 interpolates between deterministic
            # (large shape) and exponential (shape 1).
            shape = max(0.35, 1.0 / (p.inter_arrival_jitter**2))
            gaps = self.rng.gamma(shape, mean_gap / shape, size=n)
        return start_day + np.cumsum(gaps)

    def generate(self, ring: Ring, member: RingMember) -> list[AuthAttempt]:
        """All card-testing attempts for one active ring member."""
        if member.dormant or member.activation_day is None:
            return []

        rng = self.rng
        base_profile = self.cfg.ring
        out: list[AuthAttempt] = []
        identity = member.identity

        for burst_ix in range(max(1, base_profile.n_bursts_per_account)):
            start = member.activation_day + burst_ix * float(rng.uniform(0.5, 4.0))
            if start >= self.cfg.days:
                break
            p = self._profile_at(base_profile, start)

            merchants = ring.target_merchants
            if p.merchant_spread > len(merchants):
                pool = self.world.card_test_merchants
                extra = min(p.merchant_spread - len(merchants), len(pool))
                if extra > 0:
                    merchants = merchants + [
                        pool[int(i)] for i in rng.choice(len(pool), size=extra, replace=False)
                    ]

            device = identity.device_ids[int(rng.integers(len(identity.device_ids)))]
            session = f"ses_{int(rng.integers(0, 2**48)):012x}"

            for t in self.burst_times(start, p):
                if t >= self.cfg.days:
                    break
                # Even a bot mostly runs when traffic is normal; testing at
                # 04:00 into a sleeping merchant is a free giveaway.
                if rng.random() > min(1.0, seasonality(t) / 1.6):
                    continue

                merchant = merchants[int(rng.integers(len(merchants)))]
                card_token, suffix = self.world.next_pan(sequential=True)

                zero_auth = bool(rng.random() < 0.35)
                if zero_auth:
                    amount = 0.0
                else:
                    # Small, but a careful operator keeps amounts plausible for
                    # the merchant rather than pinning them at the floor.
                    amount = float(np.clip(rng.gamma(1.6, 1.4), 0.2, 40.0))

                # Most tested cards are dead. That high decline ratio in a short
                # window is the loudest single signal in the whole problem.
                approved = rng.random() < 0.06
                if approved:
                    code = AuthResponseCode.APPROVED
                    cvv = CvvResult.MATCH
                    avs = AvsResult.MATCH if rng.random() < 0.5 else AvsResult.PARTIAL
                else:
                    code = AuthResponseCode(
                        str(
                            rng.choice(
                                [
                                    AuthResponseCode.DECLINED_INVALID_CARD.value,
                                    AuthResponseCode.DECLINED_CVV.value,
                                    AuthResponseCode.DECLINED_DO_NOT_HONOR.value,
                                    AuthResponseCode.DECLINED_EXPIRED.value,
                                ],
                                p=[0.45, 0.28, 0.17, 0.10],
                            )
                        )
                    )
                    cvv = CvvResult.NO_MATCH if rng.random() < 0.7 else CvvResult.NOT_PROVIDED
                    avs = AvsResult.NO_MATCH if rng.random() < 0.75 else AvsResult.UNAVAILABLE

                out.append(
                    AuthAttempt(
                        day=float(t),
                        account_id=identity.account_id,
                        identity_id=identity.identity_id,
                        institution_id=identity.institution_id,
                        merchant=merchant,
                        card_token=card_token,
                        pan_suffix6=suffix,
                        amount=amount,
                        is_zero_auth=zero_auth,
                        response_code=code,
                        avs_result=avs,
                        cvv_result=cvv,
                        device_id=device,
                        ip_id=identity.ip_id,
                        asn=identity.asn,
                        session_id=session,
                        is_fraud=True,
                        is_bot=True,
                    )
                )
        return out

    def warmup(self, member: RingMember, until_day: float) -> list[AuthAttempt]:
        """Ageing traffic: small, ordinary purchases that build a clean record.

        These are labelled **not fraud**, because they are not -- they are real
        approved purchases made to manufacture trust. That is what makes the
        ageing stage effective and what makes the seam worth scoring.
        """
        rng = self.rng
        identity = member.identity
        start = identity.onboarded_day
        end = min(until_day, self.cfg.days)
        if end <= start:
            return []

        n = int(rng.poisson(member.warmup_rate * (end - start)))
        out: list[AuthAttempt] = []
        for _ in range(n):
            t = float(rng.uniform(start, end))
            if rng.random() > min(1.0, seasonality(t) / 1.6):
                continue
            merchant = self.world.merchants[int(rng.integers(len(self.world.merchants)))]
            card_token, suffix = self.world.next_pan(sequential=False)
            approved = rng.random() > merchant.base_decline_rate
            out.append(
                AuthAttempt(
                    day=t,
                    account_id=identity.account_id,
                    identity_id=identity.identity_id,
                    institution_id=identity.institution_id,
                    merchant=merchant,
                    card_token=card_token,
                    pan_suffix6=suffix,
                    amount=float(np.clip(rng.lognormal(3.0, 0.7), 1.0, 300.0)),
                    is_zero_auth=False,
                    response_code=(
                        AuthResponseCode.APPROVED
                        if approved
                        else AuthResponseCode.DECLINED_INSUFFICIENT_FUNDS
                    ),
                    avs_result=AvsResult.MATCH,
                    cvv_result=CvvResult.MATCH,
                    device_id=identity.device_ids[0],
                    ip_id=identity.ip_id,
                    asn=identity.asn,
                    session_id=f"ses_{int(rng.integers(0, 2**48)):012x}",
                    is_fraud=False,
                )
            )
        return out
