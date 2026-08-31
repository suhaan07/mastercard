"""Legitimate customers and their transaction behaviour.

The population has to be *hard*: stable enough that anomalies mean something,
varied enough that ordinary life does not look like fraud. Customers move house,
buy new phones and travel, and the detector has to tolerate all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim.world import Identity, Merchant, World

#: Hour-of-day multipliers. People transact in the evening, not at 04:00 --
#: and a bot that ignores this is easy to spot, so the bots have to respect it
#: too or the detection problem becomes trivial.
HOURLY_SHAPE = np.array(
    [0.25, 0.15, 0.10, 0.08, 0.08, 0.12, 0.30, 0.65, 0.95, 1.05, 1.10, 1.20,
     1.30, 1.25, 1.15, 1.10, 1.15, 1.35, 1.55, 1.60, 1.40, 1.05, 0.70, 0.40],
    dtype=np.float64,
)
#: Monday..Sunday.
DAILY_SHAPE = np.array([0.95, 0.95, 1.00, 1.05, 1.25, 1.30, 0.95], dtype=np.float64)


def seasonality(day: float) -> float:
    """Combined daily and weekly multiplier at a fractional day offset."""
    hour = int((day % 1.0) * 24) % 24
    dow = int(day) % 7
    return float(HOURLY_SHAPE[hour] * DAILY_SHAPE[dow])


@dataclass
class LifeEvent:
    """A genuine change in a customer's behaviour. Not fraud, but it looks new."""

    day: float
    kind: str  # "device_change" | "travel" | "move"
    duration_days: float = 0.0
    new_device_id: str | None = None
    new_ip_id: str | None = None
    new_asn: int | None = None


@dataclass
class Customer:
    identity: Identity
    home_subnet: str
    #: Preference weights over merchant categories.
    category_prefs: dict[str, float]
    #: Merchants this customer actually uses.
    favourite_merchants: list[Merchant]
    daily_rate: float
    amount_mu: float
    amount_sigma: float
    life_events: list[LifeEvent] = field(default_factory=list)
    is_lookalike: bool = False
    lookalike_kind: str | None = None

    def state_at(self, day: float) -> tuple[str, str, int]:
        """Device, IP and ASN in effect at ``day``, after any life events."""
        device = self.identity.device_ids[0]
        ip_id, asn = self.identity.ip_id, self.identity.asn
        for ev in self.life_events:
            if ev.day > day:
                continue
            if ev.kind == "device_change" and ev.new_device_id:
                device = ev.new_device_id
            elif ev.kind == "move" and ev.new_ip_id:
                ip_id, asn = ev.new_ip_id, ev.new_asn or asn
            elif ev.kind == "travel" and day <= ev.day + ev.duration_days:
                if ev.new_ip_id:
                    ip_id, asn = ev.new_ip_id, ev.new_asn or asn
        return device, ip_id, asn


class PopulationGenerator:
    """Builds the legitimate customer base and its life events."""

    def __init__(self, world: World, rng: np.random.Generator) -> None:
        self.world = world
        self.rng = rng
        self.cfg = world.cfg

    def build(self, n: int, start_index: int = 0) -> list[Customer]:
        customers: list[Customer] = []
        pc = self.cfg.population
        for i in range(n):
            idx = start_index + i
            inst = self.world.institutions[int(self.rng.integers(len(self.world.institutions)))]
            # Most of the population predates the observation window; a minority
            # onboards during it, which is what makes account age informative.
            onboarded = float(
                -self.rng.exponential(400.0)
                if self.rng.random() < 0.85
                else self.rng.uniform(0, self.cfg.days * 0.8)
            )
            identity = self.world.make_identity(
                index=idx, institution_id=inst, onboarded_day=onboarded, is_synthetic=False
            )
            customers.append(self._behaviour(identity))
        return customers

    def _behaviour(self, identity: Identity) -> Customer:
        rng = self.rng
        pc = self.cfg.population
        w = self.world

        prefs: dict[str, float] = {}
        from sim.world import MERCHANT_CATEGORIES

        picks = rng.choice(len(MERCHANT_CATEGORIES), size=int(rng.integers(3, 7)), replace=False)
        for p in picks:
            prefs[MERCHANT_CATEGORIES[int(p)]] = float(rng.uniform(0.4, 1.0))

        favourites: list[Merchant] = []
        for cat in prefs:
            pool = w.merchants_by_category.get(cat, [])
            if pool:
                k = min(len(pool), int(rng.integers(1, 4)))
                favourites.extend(pool[j] for j in rng.choice(len(pool), size=k, replace=False))
        if not favourites:
            favourites = [w.merchants[int(rng.integers(len(w.merchants)))]]

        cust = Customer(
            identity=identity,
            home_subnet=w.new_subnet(),
            category_prefs=prefs,
            favourite_merchants=favourites,
            daily_rate=float(
                np.clip(rng.lognormal(math.log(max(pc.daily_rate_mean, 1e-3)), pc.daily_rate_sigma), 0.02, 12.0)
            ),
            amount_mu=float(rng.normal(pc.amount_mu, 0.35)),
            amount_sigma=float(np.clip(rng.normal(pc.amount_sigma, 0.15), 0.3, 2.0)),
        )
        self._add_life_events(cust)
        return cust

    def _add_life_events(self, cust: Customer) -> None:
        rng = self.rng
        pc = self.cfg.population
        days = self.cfg.days
        if rng.random() < pc.p_device_change:
            cust.life_events.append(
                LifeEvent(
                    day=float(rng.uniform(0, days)),
                    kind="device_change",
                    new_device_id=self.world.new_device_id(),
                )
            )
        if rng.random() < pc.p_travel:
            ip, asn = self.world.new_ip()
            cust.life_events.append(
                LifeEvent(
                    day=float(rng.uniform(0, max(1.0, days - 14))),
                    kind="travel",
                    duration_days=float(rng.uniform(3, 16)),
                    new_ip_id=ip,
                    new_asn=asn,
                )
            )
        if rng.random() < pc.p_move:
            ip, asn = self.world.new_ip()
            cust.life_events.append(
                LifeEvent(day=float(rng.uniform(0, days)), kind="move", new_ip_id=ip, new_asn=asn)
            )

    # -- transactions ------------------------------------------------------

    def transaction_times(self, cust: Customer, days: int) -> np.ndarray:
        """Thinned Poisson process with daily and weekly seasonality.

        Draws at the peak rate and rejects proportionally to the seasonality
        multiplier, which is the standard thinning construction and keeps the
        realised rate correct rather than merely rate-shaped.
        """
        peak = float(HOURLY_SHAPE.max() * DAILY_SHAPE.max())
        expected = cust.daily_rate * days * peak
        n = self.rng.poisson(expected)
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        candidates = np.sort(self.rng.uniform(0, days, size=int(n)))
        keep = np.array([seasonality(t) / peak for t in candidates])
        return candidates[self.rng.random(candidates.shape[0]) < keep]

    def pick_merchant(self, cust: Customer) -> Merchant:
        """Mostly a favourite; occasionally somewhere new."""
        if cust.favourite_merchants and self.rng.random() < 0.82:
            return cust.favourite_merchants[int(self.rng.integers(len(cust.favourite_merchants)))]
        return self.world.merchants[int(self.rng.integers(len(self.world.merchants)))]

    def amount(self, cust: Customer) -> float:
        return float(np.clip(self.rng.lognormal(cust.amount_mu, cust.amount_sigma), 0.5, 8000.0))
