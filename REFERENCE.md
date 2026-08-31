# Reference: everything in this repository

A complete inventory — every layer, every feature, every field, every scenario,
and every number, with the honest limits alongside. [README.md](README.md) is
the pitch; [DEMO.md](DEMO.md) is the run order; this is the map.

**Status note.** The figures in *Measured results* were produced before the
card-wallet fix in the simulator (see [Known gaps](#known-gaps-and-honest-limits)).
They are reproducible from the data currently on this machine but not from a
fresh regeneration.

---

## Contents

1. [The problem](#1-the-problem)
2. [Repository map](#2-repository-map)
3. [The data](#3-the-data)
4. [Contracts — the frozen schemas](#4-contracts--the-frozen-schemas)
5. [The simulator](#5-the-simulator)
6. [The biometric layer](#6-the-biometric-layer)
7. [Detection, layer by layer](#7-detection-layer-by-layer)
8. [Every feature, listed](#8-every-feature-listed)
9. [The decision engine](#9-the-decision-engine)
10. [Cross-institution privacy](#10-cross-institution-privacy)
11. [The analyst copilot](#11-the-analyst-copilot)
12. [The red team](#12-the-red-team)
13. [Evaluation](#13-evaluation)
14. [Services and API surface](#14-services-and-api-surface)
15. [The console](#15-the-console)
16. [Tests](#16-tests)
17. [Running everything](#17-running-everything)
18. [Known gaps and honest limits](#18-known-gaps-and-honest-limits)

---

## 1. The problem

Two attacks that everybody defends separately, and which are actually one
supply chain:

| Stage | What happens |
|---|---|
| **Manufacture** | AI generates a person who does not exist — a face, a document, an address. Some fields are borrowed from real people and recombined. |
| **Onboard** | That identity passes KYC and opens a payment account. It is now a "new customer": no bad history, so every downstream system trusts it. |
| **Age** | The account sits quietly. A few small, normal transactions. Risk models see a boring customer. |
| **Weaponise** | The account becomes a test bench. Stolen card numbers are pushed through it to find which are live. Because the account looks legitimate, the attempts do not trip the alarms a known-bad account would. |
| **Monetise** | Validated cards are used or sold. The account is burned; the next of the batch takes over. |

**The gap.** Identity risk is scored *once, at onboarding*. Transaction risk is
scored *per transaction, at authorisation*. Nobody scores the **seam** — the
relationship between how an account was born and how it behaves in its first
weeks. The account is not the fraud; the account is the *tool*. And they are
manufactured in batches, so catching them one at a time, after each has already
been used, is losing.

Four claims follow from that, and each has a number attached:

| Claim | Measured |
|---|---|
| Ring-level, not account-level | Ring recall before first transaction **1.000**, at 0.61–0.80 precision |
| Network view beats single-player view | Sophisticated ring: one merchant **0.102** vs network **0.794** |
| Biometric similarity without biometric storage | Same face, two institution seeds: **0.0263** overlap, chance is 0.0257 |
| Detection holds as the attacker adapts | **0.70–1.00** across four operator profiles at a fixed FPR |

---

## 2. Repository map

~11,300 lines across ten modules.

| Path | Lines | What it is |
|---|---|---|
| `contracts/` | 602 | Frozen event schemas, graph vocabulary, decision types. Phase 0 — everything imports these |
| `sim/` | 1,890 | The synthetic payments world: population, rings, card testing, look-alikes, scenarios |
| `biohash/` | 1,220 | FlyHash tags, hyperdimensional binding, image pipeline, GAN-artifact detector |
| `detect/` | 3,327 | Ingest, graph, features, models, fusion, decision, evidence, red team, copilot |
| `privacy/` | 263 | Bloom-filter / MinHash cross-institution indicator exchange |
| `api/` | 477 | FastAPI scoring service |
| `eval/` | 736 | Metrics at a fixed FPR, robustness curve |
| `gateway/src/` | 356 | Fastify gateway + mock merchant and issuer services |
| `ui/src/` | 1,617 | React analyst console and guided tour |
| `tests/` | 785 | 16 property tests, 23 end-to-end tests |

Not in git (regenerable): `data/` (1.4 GB), `models/` (5.4 MB), `node_modules/`,
`.venv/`, `dist/`, `.env`.

---

## 3. The data

### What exists on disk

Five generated scenarios. Everything is seeded and reproducible.

| Scenario | Days | Onboarding | Auth events | Telemetry | Labels | Rings | Ring sizes | Fraud rate |
|---|---|---|---|---|---|---|---|---|
| `sloppy` | 90 | 4,279 | 348,662 | 295,378 | 2,087 | 4 | 51, 25, 31, 47 | 0.72% |
| `moderate` | 90 | 4,281 | 349,419 | 295,446 | 2,715 | 5 | 42, 35, 35, 24, 20 | 0.92% |
| `sophisticated` | 120 | 4,304 | 463,185 | 393,289 | 2,200 | 6 | 37, 38, 20, 24, 24, 36 | 0.53% |
| `drift` | 120 | 4,319 | 462,598 | 393,041 | 1,953 | 5 | 42, 39, 42, 37, 34 | 0.45% |
| `tiny` | 30 | 157 | 5,976 | 3,315 | 707 | 2 | 11, 11 | 15.75% |

`tiny` is the test fixture, not a demo scenario — deliberately dense so the
end-to-end suite exercises every structure in seconds.

Each scenario directory holds six JSONL files plus `meta.json`:

```
data/<scenario>/
  onboarding_events.jsonl     one per identity, at t=0 for that account
  auth_events.jsonl           every authorisation attempt
  session_telemetry.jsonl     device/behavioural telemetry per session
  labels.jsonl                chargebacks and analyst verdicts, delayed
  ground_truth.jsonl          EVALUATION ONLY — never an input
  meta.json                   seed, config, counts, generation time
```

### The critical separation

`labels.jsonl` and `ground_truth.jsonl` are **not** the same thing, and
conflating them is the mistake that quietly ruins fraud models:

- **Labels** are what a real system would have: chargebacks arriving **20–60
  days late**, at **72% coverage** (28% never labelled at all), plus a small
  rate of **false chargebacks** on genuine transactions — friendly fraud. Every
  training row is filtered on `label_available_at <= cutoff`.
- **Ground truth** is what actually happened, used only for measurement.
  `EVAL_ONLY_FIELDS` is enforced at training time: a leaked column raises
  `LeakageError` rather than producing a spectacular meaningless metric.

### Regenerating

```powershell
.venv\Scripts\python.exe -m sim.run --scenario moderate
.venv\Scripts\python.exe -m sim.verify --data data/moderate --plot
```

`sim.verify` prints class balance, ring sizes, dormancy, and an hourly-volume
sanity plot.

---

## 4. Contracts — the frozen schemas

Pydantic models in `contracts/schemas.py`. Every module imports these; nothing
redefines them.

### OnboardingEvent — 22 fields

```
event_id, ts, institution_id, application_id, identity_id, account_id,
name_token, dob_token, address_token, email_token, phone_token,
email_handle_shape, device_id, ip_id, asn, declared_age,
credit_file_age_months, address_shared_count,
face_tag, doc_template_tag, identity_hypervector, signals
```

PII is **tokenised before emission**. `name_token` is a vault token, not a name.
The three tags are sparse index sets, never embeddings.

### VerificationSignals — 7 fields

Nested inside `OnboardingEvent.signals`. What an identity-verification vendor
would emit, plus our own detector:

| Field | Source | Direction when synthetic |
|---|---|---|
| `template_match_score` | simulated vendor | lower |
| `exif_consistency` | simulated vendor | lower |
| `liveness_score` | simulated vendor | lower |
| `spectral_peak_ratio` | **measured** from the image | higher |
| `residual_kurtosis` | **measured** | higher |
| `color_corr_anomaly` | **measured** | higher |
| `saturation_clip_ratio` | **measured** | higher |

### AuthEvent — 19 fields

```
event_id, ts, institution_id, merchant_id, merchant_category, account_id,
identity_id, card_token, pan_suffix6, amount, currency, is_zero_auth,
response_code, avs_result, cvv_result, device_id, ip_id, asn, session_id
```

Card numbers come only from the non-issued test range (`TEST_BIN = "411111"`).
`pan_suffix6` exists so PAN-entropy features can see enumeration.

### SessionTelemetry — 15 fields

```
event_id, ts, institution_id, session_id, account_id, device_id, ip_id, asn,
user_agent_hash, typing_cadence_cv, mouse_entropy, page_dwell_ms,
automation_score, tz_offset_min, screen_res
```

### Label — 9 fields

```
event_id, ts, institution_id, subject_type, subject_id, is_fraud, source,
event_ts, label_available_at
```

`label_available_at` is in the schema rather than derived, because the
train/serve skew it prevents is invisible otherwise.

### GroundTruth — 8 fields

```
identity_id, account_id, is_synthetic, ring_id, ring_preset, is_lookalike,
first_fraud_ts, onboarded_ts
```

### Graph vocabulary — `contracts/graph_types.py`

**Node types:** `identity, account, device, ip_asn, email, phone, address,
card_token, merchant, face_cluster, doc_template`

**Edge types:** `observed_together, owns, transacted, similar_tag,
pii_recombination`

**`SUSPICIOUS_SHARE_WEIGHT`** — how much sharing each attribute type implicates
a group. This is domain knowledge encoded as a constant:

| Device | Face cluster | Doc template | Address | Phone | Email | IP/ASN | Card | Merchant |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 0.9 | 0.7 | 0.6 | 0.6 | 0.4 | 0.3 | **0.0** |

Merchant is zero — everyone shops somewhere.

---

## 5. The simulator

### Design decision, stated on the slide

The simulator produces **event records and feature vectors only**. It does not
generate fake faces or forged documents, and it uses only non-issued test card
ranges. Identity fraud is modelled at the level of *signals a verification
vendor would emit*, drawn from separate genuine and synthetic distributions.

That keeps this a defence tool rather than a forgery tool, and it costs nothing
— the detection layer consumes signals, not raw images, exactly as a real
system does.

### Components

| File | What it builds |
|---|---|
| `sim/world.py` | Institutions, merchants, identities, tokenisation, test PANs |
| `sim/population.py` | Legitimate customers: log-normal amounts, Poisson inter-arrival with daily/weekly seasonality, 1–3 card wallet, 1–3 devices, life events |
| `sim/rings.py` | Fraud rings from an operator profile |
| `sim/cardtest.py` | Card-testing bursts |
| `sim/lookalikes.py` | Genuine customers who *look* like fraud |
| `sim/verify.py` | Sanity report over generated data |
| `sim/config.py` | Scenario config dataclasses |
| `sim/run.py` | Orchestration and JSONL emission |

### Statistical models used

| Thing | Distribution |
|---|---|
| Transaction amounts | Log-normal |
| Inter-arrival times | Poisson with daily/weekly seasonality |
| Account tenure | Exponential — 85% predate the observation window |
| Verification signals | **Beta**, separate genuine/synthetic draws with deliberate overlap: genuine `Beta(8.0, 1.6)`, synthetic `Beta(6.0, 2.2)` |
| Chargeback delay | Uniform 20–60 days |

**No generative model.** No GAN, no diffusion, no face synthesis.

### The operator profile — the difficulty dial

Every ring is parameterised by seven knobs. Turn them up for a lazy operator,
down for a patient one:

| Knob | `sloppy` | `moderate` | `sophisticated` | Controls |
|---|---|---|---|---|
| `device_reuse_rate` | 0.85 | 0.55 | 0.18 | How often members share a device |
| `subnet_concentration` | 0.90 | 0.60 | 0.22 | How tightly IPs cluster |
| `pii_recombination_rate` | 0.60 | 0.35 | 0.12 | How much real PII is recycled |
| `doc_template_reuse` | 0.90 | 0.60 | 0.28 | Shared document generator |
| `face_reuse_rate` | 0.80 | 0.50 | 0.25 | Reused / near-duplicate faces |
| `artifact_strength` | 1.60 | 0.80 | 0.25 | How loudly the generator leaves artifacts |
| `dormancy_days_min` | 2 | 10 | 30 | How long accounts age before use |
| `burst_intensity` | 45/hr | 30 | 9 | Card-testing attempts per hour |
| `burst_hours` | 2.0 | 3.0 | 8.0 | How long a burst runs |
| `merchant_spread` | 2 | 5 | 14 | Merchants the testing is split across |
| `inter_arrival_jitter` | 0.10 | 0.35 | 0.85 | Timing randomness — low is robotic |
| `dormant_share` | 0.60 | 0.65 | 0.62 | Fraction that never transact |

A fourth scenario, **`drift`**, starts at moderate and shifts parameters
mid-run: burst intensity × 0.35, merchant spread × 2.5, device reuse × 0.4,
jitter × 2.5. The attacker adapting while you watch.

### Realism knobs that make it hard

- **Class imbalance** — fraud at 0.45–0.92% of events, not 50%
- **Label delay** — chargebacks 20–60 days late, 72% coverage
- **False chargebacks** — friendly fraud on genuine transactions
- **Look-alikes** — a genuine small business doing many small transactions, a
  traveller abroad, a customer on a new device
- **Life events** — device changes, moves, travel, card reissues
- **Concept drift** — parameters shift mid-run

---

## 6. The biometric layer

### The decision: FlyHash, not embeddings

Face embeddings are the obvious choice and the wrong one. They are
**invertible** — template-inversion attacks reconstruct a recognisable face, so
an embedding database is a face database with extra steps — and **linkable**, so
the same embedding at two institutions identifies a person across them, and
cross-institution sharing dies on GDPR/DPDP grounds.

Instead, after the fruit fly's olfactory circuit (Dasgupta, Stevens & Navlakha,
*Science* 2017): sparse random **expansion** into a wide layer, then
winner-take-all keeping the top ~5%. The output is a sparse binary **tag**.

Three properties an embedding cannot offer:

1. **A tag is a set, not a vector.** Near-duplicate detection is set
   intersection, so it composes directly with the Bloom-filter exchange — face
   matching and cross-institution sharing become *one* mechanism.
2. **Revocable and unlinkable.** Per-institution secret seeds mean the same face
   yields non-comparable tags at different institutions; reseeding revokes.
3. **Lossy by construction.** Winner-take-all discards magnitude.

**Claim discipline:** revocability, unlinkability and lossy construction under a
stated threat model (`biohash/flyhash.py::THREAT_MODEL`). *Not* cryptographic
non-invertibility — that is unproven for FlyHash, and overclaiming it does not
survive a technical question.

### Files

| File | What it does |
|---|---|
| `flyhash.py` | Sparse random projection + winner-take-all; MinHash and banding |
| `hdc.py` | Hyperdimensional binding/bundling (Kanerva) — face ⊕ doc ⊕ address ⊕ device into one hypervector |
| `images.py` | Image pipeline, descriptors, population whitening, procedural face source |
| `artifacts.py` | GAN-artifact feature extraction |
| `doctemplate.py` | Document template structure hashing |

### The artifact detector

Four **measured** statistics, no classifier — they feed the L4 model as
interpretable inputs:

| Feature | What it measures |
|---|---|
| `spectral_peak_ratio` | Upsampling grid energy in the FFT — a transposed-convolution fingerprint |
| `residual_kurtosis` | Kurtosis of the high-pass residual |
| `color_corr_anomaly` | Deviation from natural channel correlation |
| `saturation_clip_ratio` | Fraction of clipped pixels |

### What we generate, and what we deliberately do not

The AI-generated face class is meant to come from a published deepfake-detection
research corpus — we *consume* generated faces rather than manufacture them.
`images.py` keeps a seeded procedural backend behind the same interface so the
pipeline always runs; **all artifact numbers are from that procedural backend**
and should be re-quoted against a real corpus before they go in a deck.

Identity documents are never rendered. Template reuse is a *structural* signal,
detected structurally on visibly non-realistic layouts.

All image work happens at onboarding. The auth path does none.

---

## 7. Detection, layer by layer

```
onboarding ─┐
auth       ─┼─► L1 ingest + PII vault ──┬─► L2 identity graph ──┐
telemetry  ─┘   (detect/ingest.py)      │   (detect/graph/)     │
                                        ├─► L3 stream features ─┤
                                        │   (detect/features/)  │
                                        └─► L4 models ──────────┤
                                            (detect/models/)    │
                                                                ▼
                                              L5 fusion + retro-propagation
                                                   (detect/fusion.py)
                                                                ▼
                                              L6 decision engine + reason codes
                                                  (detect/decision.py)
                                                                ▼
                     React console ◄── Fastify gateway ◄── FastAPI scorer
                        (ui/)          (gateway/)            (api/)
```

### L1 — Ingest and normalise (`detect/ingest.py`)

One read path for all four streams, and one rule: **values do not cross this
boundary, only tokens and tags do.** `PIIVault.resolve()` raises
`PermissionError` unconditionally — it is a locked door with nothing behind it,
which is the correct amount of PII for a demo to hold.

The other job is **view scoping**, enforced at read time rather than display
time. `ViewScope.MERCHANT` or `ISSUER` requires an `institution_id` and returns
only that slice. That is what makes the network-view delta a measurement rather
than a filter on a chart.

### L2 — Identity graph (`detect/graph/`)

- **`build.py`** — nodes and edges from the three event streams
- **`entity_res.py`** — identity-to-identity links
- **`communities.py`** — Louvain, then ranking

**Entity-resolution link kinds**, with measured same-ring precision on `sloppy`:

| Kind | Links | Precision |
|---|---|---|
| `near_duplicate_identity_hypervector` | 1,096 | **1.000** |
| `shared_address_token` | 1,079 | **1.000** |
| `shared_device_id` | 488 | **1.000** |
| `shared_phone_token` / `pii_recombination_phone_token` | 71 | **1.000** |
| `shared_ip_id` | 10 | 1.000 |
| `near_duplicate_doc_template_tag` | 1,340 | 0.977 |
| `near_duplicate_face_tag` | 1,010 | 0.912 |
| `shared_dob_token` / `pii_recombination_dob_token` | 300 | 0.257 |
| `email_shape` | 1,078 | **0.000** → now rarity-gated |

The hypervector bundle is the strongest linker, because a coincidental match on
any single attribute cannot produce a match on the bundle.

**Community ranking** is `cohesion × suspicion × log(size)`:

- **cohesion** — mean edge weight within the community
- **suspicion** — the share of connective tissue coming from things legitimate
  people do not have in common, weighted by `SUSPICIOUS_SHARE_WEIGHT`
- **`log(size)`** — because density falls off quadratically with size, and
  without it a household of four outranks a 51-account ring

### L3 — Behavioural stream features (`detect/features/stream.py`)

Rolling windows at **1 min / 1 hr / 24 hr / 7 d**, keyed by account, device and
IP. O(1) updates, because this is the only layer on the auth path.

Each event is scored **before** being folded into the windows, so a feature row
never contains the event it describes.

### L4 — Models (`detect/models/train.py`)

Two LightGBM classifiers. Configuration:

```
n_estimators 300, learning_rate 0.05, num_leaves 31, min_child_samples 30,
subsample 0.9, colsample_bytree 0.8, reg_lambda 1.0,
scale_pos_weight = neg/pos
```

Two guards run before any fitting, and both **fail the run** rather than warn:

1. **No eval-only field reaches a feature frame.** A single leaked column would
   make every metric meaningless while looking spectacular.
2. **No label is used before it existed.** Every training row is filtered on
   `label_available_at <= cutoff`.

Plus a third: below **25 positive training labels** the trainer raises rather
than returning a model.

### L5 — Fusion and retro-propagation (`detect/fusion.py`)

**This is the contribution.** Two directions:

- **Forward** — `fuse_forward(behaviour, onboarding, weight=0.35)`: a
  multiplicative lift, so onboarding can only *raise* a score, never suppress
  it. Averaging would let a well-manufactured identity dilute genuine
  behavioural evidence, which is exactly backwards.
- **Backward** — when an account is confirmed bad, evidence propagates through
  the graph to its ring siblings, **including dormant accounts that have done
  nothing yet**.

Implemented as **weighted score diffusion**, not a GNN — best-first traversal
so the path that survives to each identity is the strongest one, not merely the
shortest. No learned parameters, and every hop is traceable.

```
contribution = 0.55 (decay) × edge conductance × node conductance × weight strength
max 3 hops, floor 0.02
```

**Edge conductance:** `similar_tag` 1.0, `pii_recombination` 1.0, `owns` 0.95,
`observed_together` 0.8, `transacted` 0.15

**Node conductance:** device 1.0, account 0.9, identity 0.9, address 0.7, phone
0.65, email 0.6, IP/ASN 0.35, card token 0.25, **merchant 0.0**

Three hops is the ceiling because beyond it almost everything connects to
everything, and propagation spreads suspicion rather than evidence.

### L6 — Decision engine (`detect/decision.py`)

See [section 9](#9-the-decision-engine).

### L7 — Analyst console (`ui/`)

See [section 15](#15-the-console).

### Ring evidence (`detect/evidence.py`)

Not a numbered layer — the presentation layer for everything above. Assembles
26 metrics into the three stages of the supply chain, each beside the same
figure for everyone *outside* the ring. Built once with the graph, off the auth
path. See [section 8](#8-every-feature-listed).

---

## 8. Every feature, listed

### Onboarding model — 24 features

```
address_shared_count, color_corr_anomaly, credit_age_ratio,
credit_file_age_months, declared_age, dob_name_variety,
email_shape_frequency, exif_consistency, is_thin_file, liveness_score,
near_dup_doc_template_tag_count, near_dup_face_tag_count,
near_dup_identity_hypervector_count, phone_name_variety, residual_kurtosis,
saturation_clip_ratio, shared_address_token_count, shared_device_id_count,
shared_dob_token_count, shared_ip_id_count, shared_phone_token_count,
spectral_peak_ratio, tag_sparsity, template_match_score
```

**Top by importance** (`moderate`): `exif_consistency` 239,
`shared_device_id_count` 195, `saturation_clip_ratio` 153, `liveness_score` 90,
`color_corr_anomaly` 69, `credit_file_age_months` 52, `template_match_score` 44.

The biohash layer is doing most of the work here.

**Classic synthetic-identity tells, feature-engineered:**

- `credit_age_ratio` — credit file age against the maximum plausible for the
  declared age. The tell is the *inconsistency*, not youth: a 45-year-old with a
  four-month file is odd; a 19-year-old with one is normal.
- `dob_name_variety`, `phone_name_variety` — PII recombination: one person's
  details appearing under more than one name.
- `address_shared_count` — mail-drop detection.
- `near_dup_*_count` — near-duplicate face, document template, hypervector.

### Behaviour model — 54 features

```
acct_age_days, acct_is_new_merchant, acct_known_merchants, amount,
avs_bad, cvv_bad, day_of_week, hour_of_day, is_zero_auth,
acct_pan_entropy_1h, acct_pans_per_merchant_1h, acct_timing_cv_1h,

# each at 1m / 1h / 24h / 7d:
acct_n_*, acct_decline_ratio_*, acct_cvv_bad_ratio_*, acct_avs_bad_ratio_*,
acct_distinct_pans_*, acct_distinct_merchants_*, acct_low_ticket_ratio_*,
acct_mean_amount_*, acct_zero_auth_ratio_*,

# device- and IP-keyed, 1h:
device_n_1h, device_distinct_pans_1h, device_decline_ratio_1h,
ip_n_1h, ip_distinct_pans_1h, ip_decline_ratio_1h
```

**Top by importance** (`moderate`): `amount` 1331, `acct_age_days` 873,
`acct_mean_amount_7d` 849, `hour_of_day` 754, `acct_mean_amount_24h` 742,
`acct_known_merchants` 359, `day_of_week` 353, `acct_distinct_pans_7d` 350,
`acct_timing_cv_1h` 310.

**The card-testing signature:** many distinct PANs, small amounts, high
declines, short window, narrow merchant set. `acct_pan_entropy_1h` catches
enumeration — sequential guessing has low digit entropy. `acct_timing_cv_1h`
catches bots, which are too regular.

**Note `acct_age_days` at rank 2.** That is the mechanism behind the thin-file
fairness failure, visible in the model's own weights.

### Ring evidence — 26 metrics across three stages

**1 · Manufacture** — the AI-generated face and document

| Metric | Suspicious direction |
|---|---|
| `spectral_peak_ratio` | higher |
| `residual_kurtosis` | higher |
| `color_corr_anomaly` | higher |
| `saturation_clip_ratio` | higher |
| `template_match_score` | lower |
| `exif_consistency` | lower |
| `liveness_score` | lower |

**2 · Onboard** — how the identity passed KYC

| Metric | Direction |
|---|---|
| `thin_file_share` | higher |
| `credit_age_ratio` | lower |
| `address_shared_count` | higher |
| `pii_recombination_share` | higher |
| `shared_device_share` | higher |
| `shared_address_share` | higher |
| `shared_phone_share` | higher |
| `declared_age` | neutral (context) |

**3 · Weaponise** — the card testing

| Metric | Direction |
|---|---|
| `distinct_pans_per_account` | higher |
| `decline_ratio` | higher |
| `cvv_mismatch_rate` | higher |
| `avs_mismatch_rate` | higher |
| `zero_auth_ratio` | higher |
| `low_ticket_ratio` | higher |
| `peak_attempts_per_hour` | higher |
| `pan_digit_entropy` | lower |
| `merchants_per_account` | **neutral** — a sloppy ring concentrates, a sophisticated one sprays; both readings are real |
| `accounts_transacting`, `attempts` | counts, no ratio |

Example output for the top ring on `moderate` (36 accounts, 6 institutions):

```
MANUFACTURE   peak   5x   4/7 elevated
ONBOARD       peak  36x   7/7 elevated
WEAPONISE     peak 114x   7/8 elevated
```

---

## 9. The decision engine

Graduated responses, not just block. Every action emits reason codes.

| Band | Action | What it does |
|---|---|---|
| `low` | `allow` | Nothing |
| `medium` | `step_up` | 3DS challenge |
| `elevated` | `throttle` | Silently rate-limit attempts |
| `high` | `honeypot` | Plausible responses that poison the attacker's results |
| `confirmed` | `block` | Block, freeze ring, case to analyst |

### The honeypot

`honeypot_response(card_token)` is deterministic in the token — an attacker
retrying the same card gets a consistent answer and cannot detect the honeypot
by probing for inconsistency. Roughly **one in twenty "approves"**, matching the
rate a real testing run would see. An endpoint that declines everything is as
informative as one that approves everything.

### 23 reason codes

**Card testing:** `high_decline_ratio`, `many_distinct_pans`, `low_pan_entropy`,
`zero_auth_burst`, `low_ticket_concentration`, `cvv_avs_mismatch_rate`,
`machine_regular_timing`, `narrow_merchant_set`, `velocity_spike`

**Synthetic identity:** `synthetic_identity_signals`, `gan_artifacts_detected`,
`near_duplicate_face_tag`, `shared_doc_template`, `pii_recombination`,
`thin_file_sudden_activity`, `credit_file_age_inconsistent`, `mail_drop_address`

**Ring:** `ring_membership`, `ring_sibling_confirmed_fraud`,
`shared_device_with_flagged`, `subnet_concentration`

**Network:** `cross_institution_indicator`, `network_wide_velocity`

Every code has human-readable text in `REASON_TEXT`, enforced by a test. A block
a human cannot explain is a block a regulator will not accept.

### Rules — `detect/rules.yaml`

A YAML rule file with band thresholds and twelve feature triggers, evaluated
alongside the model score. Editable without touching code.

---

## 10. Cross-institution privacy

`privacy/psi.py`. How two banks share "this identity is bad" without sharing
customers.

**Mechanism:** per-consortium FlyHash tags → MinHash signature → banded →
inserted into a Bloom filter. Only a bit array crosses the wire.

**Measured:**

```
bank_b publishes    402 identities as 12,864 indicators
filter size         23,120 bytes (58 bytes per identity)
matched             3/3 planted shared identities (21–24 of 32 bands)
false positives     0 of 400
same face, each bank's secret seed: raw overlap 0.0354 (chance ~0.026)
```

No names, no tokens, no tags, no images, and no way to enumerate the other
bank's customers from it. Internal tags stay mutually unlinkable — only
consortium-seeded indicators are shareable, and only as filter bits.

---

## 11. The analyst copilot

`detect/copilot.py`. The **only** Anthropic API call in the repository.

| | |
|---|---|
| Model | `claude-haiku-4-5` (override with `FRAUD_NARRATIVE_MODEL`) |
| Endpoint | `POST /narrate/{ring_id}` → "Write the case" in the console |
| Input | The evidence block as JSON — tokens, counts, scores |
| Cost | ~half a cent per narrative |
| Without a key | Deterministic template, which is what ships by default |

**The rule this module enforces:** the model never sees PII and never decides
anything. Every number in the prose is passed in; nothing is computed by the
model. If it hallucinates a figure, the evidence block beside it contradicts it
visibly — the correct failure mode for an assistive tool.

**Not in the auth path.** The latency budget is ~50 ms for scoring inside a
round trip of a few hundred; a model call is two orders of magnitude away.
Saying so out loud is part of the pitch.

---

## 12. The red team

`detect/redteam.py`. A **perturbation search**, not an LLM — it needs no API key.

Mutates the ring's operator profile against the deployed detector, and prices
each evasion by what it costs the operator to run:

```
objective = detection_rate + 0.35 × operator_cost
```

Sample run (`moderate`, 4 rounds):

```
baseline detection 0.9114 -> best evasion 0.8970
operator cost      0.494  -> 0.524

knobs the attacker turned:
  subnet_concentration   0.600 -> 0.808
  face_reuse_rate        0.500 -> 0.020
  dormancy_days_min     10.000 -> 9.000
```

---

## 13. Evaluation

`eval/report.py` and `eval/robustness.py`. Everything at a **fixed 0.1%
false-positive rate**, on the **out-of-time holdout** — the same split the model
was trained against. Windows are still built over the whole stream, because an
event's features depend on everything before it, but the training period is not
scored.

### Metrics reported

- Detection rate at fixed FPR — the headline
- **Attempts-to-detection** — how many card tests get through before the block
- **Ring recall before first transaction** — the differentiating number
- **Propagation precision** — what fraction of what lit up is actually synthetic
- False-decline rate on look-alikes
- Thin-file fairness check
- Merchant-view vs network-view delta
- Robustness curve across sophistication levels

### Current results

| Scenario | AUC | Detection | Realised FPR | Median attempts | Never caught |
|---|---|---|---|---|---|
| sloppy | 1.0000 | 1.0000 | 0.00100 | 1 | 0 |
| moderate | 1.0000 | 0.9961 | 0.00099 | 1 | 0 |
| sophisticated | 0.8515 | 0.7993 | 0.00099 | 2 (p90 6) | 0 |
| drift | 0.9849 | 0.7028 | 0.00099 | 1 | 0 |

**Ring recall before first transaction**

| Scenario | Never transacted | Flagged | Recall | Flagged total | Of those, synthetic |
|---|---|---|---|---|---|
| sloppy | 95 | 95 | 1.000 | 187 (4.4%) | 0.802 |
| moderate | 95 | 95 | 1.000 | 247 (5.8%) | 0.611 |
| sophisticated | 121 | 62 | 0.512 | 146 (3.4%) | 0.630 |
| drift | 119 | 119 | 1.000 | 282 (6.5%) | 0.667 |

**Single-player vs network view**

| Scenario | One merchant | Network | Delta |
|---|---|---|---|
| sloppy | 0.978 | 1.000 | +0.022 |
| moderate | 0.947 | 0.995 | +0.048 |
| **sophisticated** | **0.102** | **0.794** | **+0.692** |
| drift | 0.963 | 0.708 | −0.256 |

**Ring detection** (`detect/graph/communities.py`)

| Scenario | Ring recall | F1 |
|---|---|---|
| sloppy | 1.000 | 0.967 |
| moderate | 1.000 | 0.856 |
| drift | 0.995 | 0.941 |
| sophisticated | 0.787 | 0.819 |

**Latency:** p99 **5.4 ms** over 2,000 scored events through the service,
against a ~50 ms budget — measured while a training job competed for CPU, so
conservative.

---

## 14. Services and API surface

Four processes plus a UI. The gateway is a **real network hop**, and the mock
institution services each hold only their own traffic — so the merchant-vs-
network delta is a property of what a process can see, not a filter on a chart.

### FastAPI scorer — port 8000

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Readiness, scenario, models loaded |
| POST | `/score/auth` | **The latency-critical path** |
| POST | `/score/onboarding` | Score an application at t=0 |
| POST | `/honeypot` | Plausible, uninformative auth result |
| GET | `/communities` | Ranked candidate rings |
| GET | `/graph/ring/{id}` | Nodes and edges for one community |
| GET | `/ring/{id}/evidence` | The three-stage evidence block |
| POST | `/confirm/{identity}` | Confirm fraud, retro-propagate |
| POST | `/narrate/{id}` | Case narrative |
| GET | `/explain/{event_id}` | Reason codes for a decision |
| GET | `/metrics` | Latency percentiles, model metrics |
| GET | `/stream` | SSE replay of scored traffic |
| POST | `/admin/build-graph` | Build graph + evidence index (off the auth path) |

### Fastify gateway — port 8080

`/health`, `/authorize`, `/onboard`, `/metrics`, `/communities`,
`/graph/ring/:id`, `/ring/:id/evidence`, `/confirm/:id`, `/narrate/:id`,
`/explain/:id`, `/stream` (SSE passthrough)

### Mock institutions — ports 8101, 8102

`/health`, `/authorize`, `/device/:deviceId`. Merchant `inst_00`, issuer
`inst_01` — separate processes, each holding one institution's traffic.

---

## 15. The console

React + Vite + Tailwind, `ui/src/App.tsx` and `ui/src/Tour.tsx`.

| Panel | What it shows |
|---|---|
| Header | The attack in one sentence, merchant/network toggle, guided-tour button |
| Thesis line | The gap being attacked |
| Stat tiles | Candidate rings, ring size, flagged by propagation, never transacted |
| Candidate rings | "Suspected ring #1 · 36 accounts · 6 banks" |
| Force graph | Identities and shared infrastructure, colour legend, fit-to-canvas |
| **Why this is a ring** | Verdict sentence, then 26 metrics in three stages with log-scaled contrast bars |
| Retro-propagation | Every flagged sibling with score, hops, dormancy, evidence path |
| Case narrative | Model-written or deterministic, labelled which |
| Live stream | SSE replay with per-event latency |
| Explain | Reason codes in plain English |

### The guided tour

Eleven steps that **drive the console** rather than instructing the visitor — a
step selects a ring, confirms an account, or starts the replay, then explains
what happened. Auto-starts on first visit; repeatable from the header. Arrow
keys and Enter advance, Escape leaves.

---

## 16. Tests

```powershell
.\run.ps1 -Test          # both suites, under a minute
```

**`tests/test_biohash.py` — 16 property tests.** These lock in claims. If one
fails, a slide is wrong, not just a test: unlinkability across institution
seeds, overlap at chance for unrelated tags, revocation by reseeding, bind
bijectivity, ring separation from population.

**`tests/test_pipeline.py` — 23 end-to-end tests.** These lock in *seams* — and
every one corresponds to a failure that actually happened: model persistence
across modules, probability-not-label predictions, propagation tie handling,
propagation flooding, entity-resolution link precision, leakage guards, view
scoping, band monotonicity, reason-code coverage.

---

## 17. Running everything

```powershell
python -m venv --system-site-packages .venv
.\run.ps1 -Scenario moderate -Rebuild    # simulate, verify, train, serve
.\run.ps1 -Test                          # 39 tests
.\run.ps1 -Stop
```

Console at <http://127.0.0.1:5173>, scorer docs at <http://127.0.0.1:8000/docs>.

Individual stages:

```powershell
.venv\Scripts\python.exe -m sim.run --scenario sophisticated
.venv\Scripts\python.exe -m sim.verify --data data/sophisticated --plot
.venv\Scripts\python.exe -m detect.graph.communities --data data/sophisticated
.venv\Scripts\python.exe -m detect.models.train --data data/sophisticated
.venv\Scripts\python.exe -m eval.report --data data/sophisticated
.venv\Scripts\python.exe -m eval.robustness
.venv\Scripts\python.exe -m detect.redteam --data data/moderate --rounds 8
.venv\Scripts\python.exe -m detect.copilot --data data/moderate
.venv\Scripts\python.exe -m privacy.psi
```

Optional, for the model-written narrative:

```powershell
cp .env.example .env      # paste key into ANTHROPIC_API_KEY=
```

---

## 18. Known gaps and honest limits

Listed because a project that hides these is less trustworthy than one that
does not.

### The numbers above predate a simulator fix

The simulator was minting a fresh card token for every legitimate transaction,
so a legitimate account carried 61 distinct cards and a ring account carried 7 —
inverting the headline card-testing signal. Fixed (customers now hold a 1–3 card
wallet), but the fix shifts the random stream, so **every scenario needs
regenerating, retraining and re-reporting** before the tables are true of the
current code. Direction should hold — detection was carried by decline ratio,
CVV failure and timing rather than distinct-PAN counts — but the figures will
move.

### The thin-file fairness check is failing

| Scenario | Thin-file | Thick-file | Ratio |
|---|---|---|---|
| sloppy | 0.00114 | 0.00098 | 1.2x |
| moderate | 0.00299 | 0.00079 | **3.8x** |
| sophisticated | 0.00265 | 0.00082 | 3.2x |
| drift | 0.00414 | 0.00067 | **6.2x** |

Thin-file customers are disproportionately young, migrant or low-income and are
legitimately thin-file. The likely route is `acct_age_days` — the model's
second-strongest feature — which a brand-new genuine customer shares with a
freshly-minted synthetic account. The onboarding model handles this correctly
via `credit_age_ratio`; the behavioural model has no equivalent normalisation.
**Unfixed.**

### The network-view delta inverts on `drift`

−0.256: the merchant-scoped view measures *higher* than the network one. The
hypothesis is that dilution pulls drifted traffic back toward the training
distribution while compressing the negatives, but that is a hypothesis, not a
finding. Reported rather than dropped, because a delta quoted only where it is
favourable is not a measurement.

### Per-institution delta is 0.000 everywhere

In this simulator every one of an account's authorisations lands on its own
institution, so institution scoping does not dilute an account's windows at all.
The network advantage here is a **per-merchant** effect. Claiming otherwise
would be claiming something the data does not show.

### `sloppy` is a bad demo scenario

Its rings burst early, so its out-of-time holdout contains only 135 fraud events
across 3 accounts. A detection rate of 1.000 on three accounts is not a result
worth quoting. Demo on `moderate`, quote `sophisticated`.

### Propagation precision is 0.61–0.80

Roughly a quarter of what lights up on a confirmation is not synthetic. Reported
beside the recall, because a recall of 1.000 means nothing without it.

### Artifact numbers are from the procedural backend

The image pipeline is real, but the AI-generated face class comes from a seeded
procedural generator rather than a published deepfake corpus. Re-quote before a
deck.

### Scale

`entity_res.py` computes exact pairwise Jaccard in memory-bounded chunks —
comfortable to the low tens of thousands of identities. Beyond that,
MinHash-banded blocking becomes necessary, and must be validated for recall
rather than assumed.

### Cut deliberately

Federated learning (described, not built), GNN (score diffusion is explainable
and sufficient), sequence models (window features are enough), Docker Compose
(`run.ps1` covers the demo).

---

## Constraints honoured

| Constraint | Where |
|---|---|
| p99 auth scoring under ~50 ms | `api/main.py`; O(1) windows in `detect/features/stream.py` — measured 5.4 ms |
| Everything at a fixed FPR | `eval/report.py::threshold_for_fpr` |
| Out-of-time evaluation | `eval/report.py` scores only events after the model's own `split_ts` |
| No PII downstream of L1 | `detect/ingest.py::PIIVault` |
| Every block carries a reason | `contracts/decisions.py::REASON_TEXT` |
| Fairness on thin-file customers | `eval/report.py` — reported, and currently failing |
| Test card ranges only | `sim/world.py::TEST_BIN` |
| No LLM in the auth path | `detect/copilot.py` is reachable only from `/narrate` |
