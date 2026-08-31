"""Legitimate customers whose behaviour resembles fraud.

Reporting a detection rate without these is reporting a number that means
nothing. A model can catch every card-testing burst by declining anyone who
makes many small transactions -- and in doing so decline the coffee shop, the
subscription business and every customer who just bought a phone abroad.

Three kinds, each aimed at a different feature the detector leans on:

* **small business** -- high volume, small tickets, one merchant category.
  Attacks the "many low-value attempts" signal.
* **traveller** -- sudden new country, new ASN, unfamiliar merchants.
  Attacks the geo/ASN-novelty signal.
* **new device** -- fresh device fingerprint with no history, immediately
  active. Attacks the device-novelty signal.

Their events are labelled legitimate throughout. Any decline they receive is a
false decline, and the metrics report counts it as one.
"""

from __future__ import annotations

import numpy as np

from sim.population import Customer, LifeEvent, PopulationGenerator
from sim.world import World


class LookalikeGenerator:
    """Builds the legitimate-but-suspicious cohort."""

    def __init__(self, world: World, population: PopulationGenerator, rng: np.random.Generator) -> None:
        self.world = world
        self.population = population
        self.rng = rng
        self.cfg = world.cfg

    def build(self, start_index: int) -> tuple[list[Customer], int]:
        lc = self.cfg.lookalikes
        out: list[Customer] = []
        idx = start_index

        for _ in range(lc.n_small_business):
            cust, idx = self._one(idx, "small_business")
            # A café or a subscription service: many transactions, small
            # amounts, a narrow merchant footprint. Structurally close to a
            # testing burst, and entirely legitimate.
            cust.daily_rate = float(
                np.clip(self.rng.normal(lc.small_business_daily_rate, 6.0), 6.0, 90.0)
            )
            cust.amount_mu = float(self.rng.normal(lc.small_business_amount_mu, 0.3))
            cust.amount_sigma = 0.55
            cust.favourite_merchants = cust.favourite_merchants[:2] or cust.favourite_merchants
            out.append(cust)

        for _ in range(lc.n_travellers):
            cust, idx = self._one(idx, "traveller")
            ip, asn = self.world.new_ip()
            cust.life_events.append(
                LifeEvent(
                    day=float(self.rng.uniform(2, max(3.0, self.cfg.days - 12))),
                    kind="travel",
                    duration_days=float(self.rng.uniform(5, 20)),
                    new_ip_id=ip,
                    new_asn=asn,
                )
            )
            out.append(cust)

        for _ in range(lc.n_new_device):
            cust, idx = self._one(idx, "new_device")
            cust.life_events.append(
                LifeEvent(
                    day=float(self.rng.uniform(1, max(2.0, self.cfg.days - 3))),
                    kind="device_change",
                    new_device_id=self.world.new_device_id(),
                )
            )
            # Recently onboarded, so it is thin-file *and* newly active -- the
            # exact shape a synthetic account has, in a real customer.
            cust.identity.onboarded_day = float(self.rng.uniform(0, self.cfg.days * 0.5))
            cust.identity.credit_file_age_months = int(self.rng.integers(0, 14))
            out.append(cust)

        return out, idx

    def _one(self, idx: int, kind: str) -> tuple[Customer, int]:
        inst = self.world.institutions[int(self.rng.integers(len(self.world.institutions)))]
        onboarded = float(-self.rng.exponential(300.0))
        identity = self.world.make_identity(
            index=idx, institution_id=inst, onboarded_day=onboarded, is_synthetic=False
        )
        identity.is_lookalike = True
        identity.lookalike_kind = kind
        cust = self.population._behaviour(identity)
        cust.is_lookalike = True
        cust.lookalike_kind = kind
        return cust, idx + 1
