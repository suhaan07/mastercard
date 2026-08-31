"""The shared world: institutions, merchants, infrastructure pools, identities.

This module owns everything that both legitimate customers and fraud rings draw
from, so that sharing is meaningful. A ring is detectable precisely because its
members draw from a *narrower* pool of devices, subnets, document templates and
faces than the population does -- if each generator invented its own
infrastructure, there would be nothing to detect.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from faker import Faker

from biohash.doctemplate import DocTemplate, DocTemplateHasher
from biohash.flyhash import TAG_DIM, FlyHash
from biohash.hdc import categorical_tag, identity_hypervector
from biohash.images import (
    FEATURE_DIM,
    IDENTITY_DIM,
    DescriptorNormalizer,
    FaceSource,
    get_face_source,
    image_features,
)
from contracts.schemas import SparseTag, VerificationSignals
from sim.config import ScenarioConfig

MERCHANT_CATEGORIES = (
    "grocery",
    "fuel",
    "restaurant",
    "ecommerce",
    "streaming",
    "travel",
    "pharmacy",
    "electronics",
    "apparel",
    "gaming",
    "utilities",
    "charity",
)

#: Categories card testers favour: cheap, instant, digital, low friction.
CARD_TEST_CATEGORIES = ("streaming", "gaming", "charity", "ecommerce")

#: Non-issued test BIN. Never a live BIN, anywhere in this project.
TEST_BIN = "411111"


def _token(kind: str, value: str) -> str:
    """Stable opaque token. The vault analogue -- values never travel."""
    return f"{kind}_" + hashlib.blake2b(
        f"{kind}|{value}".encode("utf-8"), digest_size=8
    ).hexdigest()


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    category: str
    institution_id: str
    #: Baseline decline rate varies by merchant; card-testing sites are not the
    #: only places with elevated declines.
    base_decline_rate: float


@dataclass
class Identity:
    """A person, real or manufactured, as the system sees them."""

    identity_id: str
    account_id: str
    institution_id: str
    name_token: str
    dob_token: str
    address_token: str
    email_token: str
    phone_token: str
    email_handle_shape: str
    device_ids: list[str]
    ip_id: str
    asn: int
    declared_age: int
    credit_file_age_months: int
    face_tag: SparseTag
    doc_template_tag: SparseTag
    hypervector: SparseTag
    signals: VerificationSignals
    onboarded_day: float
    is_synthetic: bool = False
    ring_id: str | None = None
    is_lookalike: bool = False
    lookalike_kind: str | None = None
    address_shared_count: int = 1


class World:
    """Institutions, merchants, and the machinery that mints identities."""

    def __init__(self, cfg: ScenarioConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.faker = Faker()
        Faker.seed(cfg.seed)

        self.institutions = [
            f"{cfg.institution_prefix}_{i:02d}" for i in range(cfg.population.n_institutions)
        ]

        self.merchants: list[Merchant] = []
        for i in range(cfg.population.n_merchants):
            cat = MERCHANT_CATEGORIES[int(rng.integers(len(MERCHANT_CATEGORIES)))]
            self.merchants.append(
                Merchant(
                    merchant_id=f"mer_{i:05d}",
                    category=cat,
                    institution_id=self.institutions[int(rng.integers(len(self.institutions)))],
                    base_decline_rate=float(np.clip(rng.normal(0.055, 0.02), 0.01, 0.16)),
                )
            )
        self.merchants_by_category: dict[str, list[Merchant]] = {}
        for m in self.merchants:
            self.merchants_by_category.setdefault(m.category, []).append(m)

        #: Merchants a card-testing operator would pick.
        self.card_test_merchants = [
            m for m in self.merchants if m.category in CARD_TEST_CATEGORIES
        ] or self.merchants

        # -- biometric machinery -------------------------------------------
        self.face_source: FaceSource = get_face_source(seed=cfg.seed)
        # One institution seed for the whole simulated network. Per-institution
        # seeds are what privacy/psi.py exercises; the simulator itself needs a
        # single comparable space to plant ground truth in.
        self.fly = FlyHash(input_dim=FEATURE_DIM, dim=TAG_DIM, seed_id="network")
        self.doc_hasher = DocTemplateHasher(seed_id="network", dim=TAG_DIM)
        self.tag_dim = self.fly.dim
        self.tag_len = self.fly.hash_length

        ref = np.array(
            [image_features(self.face_source.sample(synthetic=bool(i % 2))) for i in range(240)]
        )
        self.normalizer = DescriptorNormalizer().fit(ref)

        self._pan_counter = 0
        self._address_uses: dict[str, int] = {}

    # -- infrastructure ----------------------------------------------------

    def new_device_id(self) -> str:
        return f"dev_{int(self.rng.integers(0, 2**48)):012x}"

    def new_ip(self, subnet: str | None = None) -> tuple[str, int]:
        """Return an IP token and its ASN. A shared subnet is a ring signal."""
        if subnet is None:
            subnet = f"{int(self.rng.integers(1, 224))}.{int(self.rng.integers(0, 256))}.{int(self.rng.integers(0, 256))}"
        host = int(self.rng.integers(1, 255))
        asn = int(self.rng.integers(1000, 60000))
        return _token("ip", f"{subnet}.{host}"), asn

    def new_subnet(self) -> str:
        return f"{int(self.rng.integers(1, 224))}.{int(self.rng.integers(0, 256))}.{int(self.rng.integers(0, 256))}"

    def next_pan(self, sequential: bool = False) -> tuple[str, str]:
        """Return ``(card_token, pan_suffix6)`` from the non-issued test range.

        ``sequential`` walks the suffix in near-order, which is what enumeration
        looks like and what the PAN-entropy feature is built to notice.
        """
        if sequential:
            self._pan_counter += int(self.rng.integers(1, 4))
            suffix = f"{self._pan_counter % 1_000_000:06d}"
        else:
            suffix = f"{int(self.rng.integers(0, 1_000_000)):06d}"
        return _token("card", TEST_BIN + suffix), suffix

    # -- identities --------------------------------------------------------

    def _email_shape(self, handle: str) -> str:
        """Normalised handle pattern: letters to 'a', digits to '9'.

        Batch-generated identities tend to share a construction pattern even
        when no two addresses are identical.
        """
        out = []
        for ch in handle[:16]:
            out.append("9" if ch.isdigit() else "a" if ch.isalpha() else ch)
        return "".join(out)

    def make_identity(
        self,
        *,
        index: int,
        institution_id: str,
        onboarded_day: float,
        is_synthetic: bool = False,
        ring_id: str | None = None,
        face_vec: np.ndarray | None = None,
        doc_template: DocTemplate | None = None,
        artifact_strength: float | None = None,
        device_ids: list[str] | None = None,
        ip_id: str | None = None,
        asn: int | None = None,
        shared_pii: dict[str, str] | None = None,
        address_value: str | None = None,
    ) -> Identity:
        """Mint one identity, real or manufactured.

        Ring members differ from legitimate applicants only in *what they are
        handed*: a reused face vector, a shared document template, recycled PII,
        a device from a small pool. The construction path is identical, which is
        the point -- the detector has to find them from signals, not from a flag.
        """
        rng = self.rng
        from biohash.artifacts import build_verification_signals

        name = self.faker.name()
        address_value = address_value or self.faker.address().replace("\n", ", ")
        handle = f"{name.split()[0].lower()}{int(rng.integers(10, 9999))}"

        pii = shared_pii or {}
        name_token = pii.get("name_token") or _token("name", name)
        dob_token = pii.get("dob_token") or _token("dob", str(self.faker.date_of_birth()))
        phone_token = pii.get("phone_token") or _token("phone", self.faker.msisdn())
        address_token = _token("addr", address_value)
        email_token = _token("email", f"{handle}@example.com")

        self._address_uses[address_token] = self._address_uses.get(address_token, 0) + 1

        # Face: either fresh, or a near-duplicate of the ring's base vector.
        if face_vec is None:
            face_vec = rng.random(IDENTITY_DIM).astype(np.float32)
        img = self.face_source.sample(
            synthetic=is_synthetic, identity_vec=face_vec, artifact_strength=artifact_strength
        )
        face_tag = self.fly.tag(self.normalizer.transform(image_features(img)))
        signals = build_verification_signals(img, is_synthetic=is_synthetic, rng=rng)

        # Document template.
        if doc_template is None:
            doc_template = DocTemplate.generate(f"tpl_{index}", rng)
        doc_tag = self.doc_hasher.tag_template(doc_template, rng)

        addr_tag = categorical_tag(address_token, self.tag_dim, self.tag_len, "network")
        devices = device_ids or [
            self.new_device_id() for _ in range(int(rng.integers(
                self.cfg.population.devices_min, self.cfg.population.devices_max + 1)))
        ]
        dev_tag = categorical_tag(devices[0], self.tag_dim, self.tag_len, "network")

        if ip_id is None:
            ip_id, asn = self.new_ip()

        declared_age = int(np.clip(rng.normal(41, 15), 18, 92))
        if is_synthetic:
            # A manufactured identity's credit file is younger than the person
            # claims to be -- one of the classic tells.
            credit_months = int(np.clip(rng.gamma(2.0, 6.0), 0, 60))
        else:
            plausible = max(0, (declared_age - 18) * 12)
            credit_months = int(np.clip(rng.uniform(0.35, 1.0) * plausible, 0, plausible))

        return Identity(
            identity_id=f"idn_{index:07d}",
            account_id=f"acc_{index:07d}",
            institution_id=institution_id,
            name_token=name_token,
            dob_token=dob_token,
            address_token=address_token,
            email_token=email_token,
            phone_token=phone_token,
            email_handle_shape=self._email_shape(handle),
            device_ids=devices,
            ip_id=ip_id,
            asn=asn if asn is not None else int(rng.integers(1000, 60000)),
            declared_age=declared_age,
            credit_file_age_months=credit_months,
            face_tag=face_tag,
            doc_template_tag=doc_tag,
            hypervector=identity_hypervector(
                face_tag=face_tag,
                doc_template_tag=doc_tag,
                address_tag=addr_tag,
                device_tag=dev_tag,
            ),
            signals=signals,
            onboarded_day=onboarded_day,
            is_synthetic=is_synthetic,
            ring_id=ring_id,
        )

    def address_shared_count(self, address_token: str) -> int:
        return self._address_uses.get(address_token, 1)
