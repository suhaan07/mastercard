"""Fraud ring generation: batches of synthetic identities sharing infrastructure.

Rings are the key abstraction in this project. A ring is not a set of accounts
that happen to be bad -- it is a *production batch*, and the batch leaves marks:
reused devices, a tight subnet, recycled PII fragments, one document template,
faces from one generator run. Those marks are what let us light up forty
siblings when one is caught.

Every knob on :class:`~sim.config.RingProfile` moves how visible those marks
are, which is the difficulty dial the design doc asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biohash.doctemplate import DocTemplate
from biohash.images import IDENTITY_DIM
from sim.config import RingProfile, ScenarioConfig
from sim.world import Identity, Merchant, World


@dataclass
class RingMember:
    identity: Identity
    #: When the account starts testing cards, in fractional days.
    activation_day: float | None
    #: Members that never act during the run. Retro-propagation exists to catch
    #: these before they ever transact.
    dormant: bool
    #: Light, plausible activity during the ageing period.
    warmup_rate: float


@dataclass
class Ring:
    ring_id: str
    profile_name: str
    members: list[RingMember] = field(default_factory=list)
    shared_devices: list[str] = field(default_factory=list)
    subnet: str = ""
    doc_template: DocTemplate | None = None
    face_base: np.ndarray | None = None
    target_merchants: list[Merchant] = field(default_factory=list)
    onboard_day: float = 0.0

    @property
    def active_members(self) -> list[RingMember]:
        return [m for m in self.members if not m.dormant]

    @property
    def dormant_members(self) -> list[RingMember]:
        return [m for m in self.members if m.dormant]


class RingGenerator:
    """Builds rings whose members share infrastructure at profile-controlled rates."""

    def __init__(self, world: World, rng: np.random.Generator) -> None:
        self.world = world
        self.rng = rng
        self.cfg: ScenarioConfig = world.cfg

    def build(self, start_index: int) -> tuple[list[Ring], int]:
        p = self.cfg.ring
        rings: list[Ring] = []
        idx = start_index
        for r in range(p.n_rings):
            ring, idx = self._build_ring(f"ring_{r:03d}", p, idx)
            rings.append(ring)
        return rings, idx

    def _build_ring(self, ring_id: str, p: RingProfile, start_index: int) -> tuple[Ring, int]:
        rng = self.rng
        w = self.world
        size = int(rng.integers(p.ring_size_min, p.ring_size_max + 1))

        # The operator's shared kit. A sloppy operator uses few devices for many
        # accounts; a careful one buys nearly one per account.
        n_devices = max(1, int(round(size * (1.0 - p.device_reuse_rate))))
        shared_devices = [w.new_device_id() for _ in range(n_devices)]
        subnet = w.new_subnet()
        doc_template = DocTemplate.generate(f"tpl_{ring_id}", rng)
        face_base = rng.random(IDENTITY_DIM).astype(np.float32)

        spread = max(1, int(p.merchant_spread))
        pool = w.card_test_merchants
        target_merchants = [
            pool[int(i)] for i in rng.choice(len(pool), size=min(spread, len(pool)), replace=False)
        ]

        # The batch onboards over a short window -- that burst of applications is
        # itself a signal, and it is why the accounts share an age.
        onboard_day = float(
            rng.uniform(0, max(1.0, self.cfg.days * p.onboard_window_fraction))
        )

        ring = Ring(
            ring_id=ring_id,
            profile_name=p.name,
            shared_devices=shared_devices,
            subnet=subnet,
            doc_template=doc_template,
            face_base=face_base,
            target_merchants=target_merchants,
            onboard_day=onboard_day,
        )

        shared_address = None
        idx = start_index
        for k in range(size):
            inst = w.institutions[int(rng.integers(len(w.institutions)))]

            # Faces: a reused generator run, or a fresh one.
            if rng.random() < p.face_reuse_rate:
                face_vec = np.clip(
                    face_base + rng.normal(0.0, p.face_jitter, size=IDENTITY_DIM), 0.0, 1.0
                ).astype(np.float32)
            else:
                face_vec = rng.random(IDENTITY_DIM).astype(np.float32)

            template = doc_template if rng.random() < p.doc_template_reuse else None

            # Devices: from the operator's pool.
            device_ids = [shared_devices[int(rng.integers(len(shared_devices)))]]
            if rng.random() < 0.25:
                device_ids.append(w.new_device_id())

            # IPs: concentrated in the operator's subnet, or dispersed.
            if rng.random() < p.subnet_concentration:
                ip_id, asn = w.new_ip(subnet)
            else:
                ip_id, asn = w.new_ip()

            # PII recombination: fragments of one "person" reappearing under
            # another name is the classic synthetic-identity tell.
            shared_pii: dict[str, str] = {}
            if rng.random() < p.pii_recombination_rate and ring.members:
                donor = ring.members[int(rng.integers(len(ring.members)))].identity
                if rng.random() < 0.5:
                    shared_pii["dob_token"] = donor.dob_token
                else:
                    shared_pii["phone_token"] = donor.phone_token

            # Mail-drop addresses: many applicants at one building.
            address_value = None
            if rng.random() < p.doc_template_reuse * 0.6:
                if shared_address is None:
                    shared_address = w.faker.address().replace("\n", ", ")
                address_value = shared_address

            onboarded = onboard_day + float(rng.uniform(0, 6))
            identity = w.make_identity(
                index=idx,
                institution_id=inst,
                onboarded_day=onboarded,
                is_synthetic=True,
                ring_id=ring_id,
                face_vec=face_vec,
                doc_template=template,
                artifact_strength=p.artifact_strength,
                device_ids=device_ids,
                ip_id=ip_id,
                asn=asn,
                shared_pii=shared_pii or None,
                address_value=address_value,
            )
            idx += 1

            dormant = bool(rng.random() < p.dormant_share)
            dormancy = float(rng.uniform(p.dormancy_days_min, p.dormancy_days_max))
            activation = onboarded + dormancy
            if activation >= self.cfg.days:
                dormant = True

            ring.members.append(
                RingMember(
                    identity=identity,
                    activation_day=None if dormant else activation,
                    dormant=dormant,
                    # Ageing traffic: a few small, ordinary-looking purchases so
                    # the account builds a thin but clean track record.
                    warmup_rate=float(rng.uniform(0.03, 0.22)),
                )
            )

        return ring, idx
