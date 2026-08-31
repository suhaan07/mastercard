# Synthetic Identity as Fraud Infrastructure — Design Doc

**Context:** Mastercard Innovation Challenge entry
**Author:** Suhaan
**Status:** Draft v1 — for team review before build starts

---

## 1. The problem, restated more sharply

Your PS describes two attacks. The strong version of the pitch is that they are **one supply chain**, and everybody defends them as if they were two.

**The chain:**

1. **Manufacture** — AI generates a fake person: a face that doesn't exist, an ID document that looks real, an address and phone that are real-ish. Some fields are borrowed from real people, recombined.
2. **Onboard** — that identity passes KYC and opens a payment account. It is now a "new customer," which means it has no bad history, which means every downstream system trusts it.
3. **Age** — the account sits quietly. A few small, normal-looking transactions. It builds a thin but clean track record. Risk models see a boring customer.
4. **Weaponise** — the account becomes a test bench. Attackers push card numbers through it to find which ones are live. Because the account looks legitimate, the attempts don't trip the alarms that a known-bad account would.
5. **Monetise** — validated cards get used for real purchases or sold on. The synthetic account is burned and the next one in the batch takes over.

**The gap we attack:** identity risk is scored *once, at onboarding*. Transaction risk is scored *per transaction, at authorisation*. Nobody scores the **seam** — the relationship between how an account was born and how it behaves in its first weeks. The account is not the fraud. The account is the *tool*. Today's systems catch tools one at a time, after they've already been used.

### How to enhance the PS (what makes it a winning framing)

Add these four dimensions. Each one is a whole extra slide of substance and each is defensible in front of judges.

**a) Ring-level, not account-level.** Synthetic identities are produced in batches, not one-offs. Batches leave shared fingerprints: reused device signatures, IP subnets, email-construction patterns, address variations of the same building, document template reuse, near-duplicate face embeddings. If you catch one, you should be able to catch the other 400 *before they act*. Nobody in a hackathon usually builds this. It is the highest-value part of the project.

**b) Network-level visibility is Mastercard's actual moat.** Card testing sprays across many merchants and many issuers deliberately, so each individual player sees only a sliver — 3 attempts here, 5 there, nothing alarming. The network sees the whole pattern. Your solution should be explicitly designed to sit where that cross-institution view exists, and should show what it catches that a single-merchant view provably cannot. Build a demo toggle: *merchant view* vs *network view*. That single toggle sells the whole project.

**c) The defender's real cost function is false declines.** Blocking fraud is easy if you don't care about blocking customers. A false decline costs a lifetime of customer value plus interchange revenue. Every metric you report must be at a **fixed false-positive budget**. Say this out loud in the pitch; it signals you understand payments rather than just ML.

**d) The adversary adapts.** This is an AI-vs-AI problem. Once your defence blocks the obvious burst pattern, the attacker slows down, spreads across accounts, and mimics organic traffic. A static rule set has a half-life. Design for drift, and demo it: run the attacker at three sophistication levels and show your detection holding.

### Explicit constraints to state in the doc (judges look for these)

| Constraint | Target |
|---|---|
| Auth-time scoring latency | p99 under ~50 ms; the whole authorisation round trip is single-digit hundreds of ms |
| False decline rate | Report everything at a fixed FPR (e.g. 0.1%) |
| Privacy | Cross-institution signals must be shared without sharing PII — RBI/DPDP in India, GDPR + PSD2 SCA in the EU |
| Explainability | Every block needs a human-readable reason (regulatory + analyst workflow) |
| Fairness | Onboarding models must not proxy for protected attributes — thin-file customers are disproportionately young, migrant, or low-income and are *legitimately* thin-file |

---

## 2. Replicating the problem — the simulation harness

You cannot get real fraud data, and you shouldn't want it. Build a synthetic payments world you fully control, where you own the ground-truth labels. This is a deliverable in its own right — judges respect a team that built its own testbed.

### Design decision worth stating explicitly in your submission

The simulator produces **event records and feature vectors only**. It does not generate fake faces or forged document images, and it uses only non-issued test card ranges (the standard `4111...` style test PANs) — never live BINs. The identity-fraud side is modelled at the level of *signals a verification vendor would emit* (e.g. `template_match_score`, `face_embedding_id`, `exif_consistency`, `liveness_score`), drawn from two different distributions for genuine and synthetic applicants.

This is a deliberate scope choice, and say so on the slide: it keeps the project a defence tool rather than a forgery tool, and it's what you want to be demoing in a room full of Mastercard people. It costs you nothing — your detection layer consumes signals, not raw images, exactly as a real system does.

### Components

**A. Population generator.** N legitimate customers with stable-ish behaviour: home geo, 1–3 devices, merchant-category preferences, transaction amounts from a log-normal distribution, inter-arrival times from a Poisson process with daily and weekly seasonality. Add life events — a move, a new phone, a holiday abroad — so your model has to tolerate genuine change.

**B. Fraud ring generator.** Rings are the key abstraction. Each ring has a hidden "operator profile" that controls how sloppy it is:

- `device_reuse_rate` — how often ring members share a device fingerprint
- `subnet_concentration` — how tightly IPs cluster
- `pii_recombination_rate` — how much real PII gets recycled across identities
- `doc_template_reuse` — how many share a document-generation artifact
- `dormancy_days` — how long accounts age before use
- `burst_intensity` — attempts per hour during the testing phase
- `merchant_spread` — how many merchants the testing is split across

Turn all of these down and you get a sophisticated ring; turn them up and you get a lazy one. This gives you your difficulty dial.

**C. Card-testing traffic generator.** Bursts of low-value or zero-value authorisation attempts with a high decline rate, testing sequential-ish PANs from the test range, with CVV/AVS mismatches. Parameterise: burst length, spread across accounts, spread across merchants, jitter in inter-arrival timing.

**D. Realism knobs that make it hard.**
- Class imbalance: fraud at 0.1–1% of events, not 50%
- Label delay: chargebacks arrive 20–60 days later, so your "labels" are late and incomplete
- Legitimate look-alikes: a genuine small business doing many small transactions, a genuine customer with a new device abroad
- Concept drift: attacker parameters shift mid-run

**E. Adversarial loop (stretch goal, big pitch value).** After your detector is trained, run a red-team agent that mutates ring parameters toward whatever gets through, then retrain. Show the arms race converging. Keep it entirely inside the simulator.

### Outputs

Three event streams, all timestamped, plus a ground-truth table:
`onboarding_events`, `auth_events`, `session_telemetry`, `labels`.
Freeze a seed so results are reproducible — judges ask.

---

## 3. Solving it — system architecture

Seven layers. Each one is a separate module with a defined contract, so parallel agents can build them independently.

```
                 ┌──────────────────────────────────────┐
   onboarding ──►│ L1  Ingest + normalise               │
   auth      ──►│     (schema, enrichment, PII vault)   │
   telemetry ──►└──────────────┬───────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 ┌─────────────┐      ┌─────────────────┐    ┌─────────────────┐
 │ L2 Identity │      │ L3 Behavioural  │    │ L4 Identity     │
 │    graph    │      │    stream       │    │    risk model   │
 │ entity res, │      │ velocity, decline│   │ onboarding-time │
 │ communities │      │ ratio, entropy   │   │ score           │
 └──────┬──────┘      └────────┬────────┘    └────────┬────────┘
        └──────────────┬───────┴──────────────────────┘
                       ▼
            ┌────────────────────────┐
            │ L5 Fusion + retro-     │  ◄── the novel bit
            │    propagation         │
            └───────────┬────────────┘
                        ▼
            ┌────────────────────────┐
            │ L6 Decision engine     │
            │ allow / step-up /      │
            │ throttle / honeypot /  │
            │ block  (+ reason codes)│
            └───────────┬────────────┘
                        ▼
            ┌────────────────────────┐
            │ L7 Analyst console +   │
            │    feedback loop       │
            └────────────────────────┘
```

### L1 — Ingest and normalise
One event schema for everything. PII goes into a vault and only tokens/hashes flow downstream — this is both correct practice and a slide about privacy-by-design.

### L2 — Identity graph
Nodes: identity, device, IP/ASN, email, phone, address, card token, merchant.
Edges: observed-together, with timestamps and weights.
Run entity resolution (fuzzy name/address/DOB matching, normalised email handles), then community detection (Louvain or Leiden) to propose candidate rings. Score each community on cohesion + suspicious-shared-attribute density.

**Classic synthetic-identity tells to feature-engineer:** PII recombination (this SSN/PAN-equivalent appears with a different name elsewhere), impossible credit-file age vs claimed age, thin file with sudden high activity, address that is a mail-drop or is shared by many unrelated applicants, near-duplicate face embeddings across applicants.

### L3 — Behavioural stream features
Rolling windows (1 min / 1 hr / 24 hr / 7 d) keyed by account, card, merchant, device, IP:
decline ratio, unique-PAN count, PAN numeric entropy (sequential guessing has low entropy), CVV/AVS mismatch rate, low-ticket ratio, zero-auth ratio, inter-arrival regularity (bots are too regular), first-time-merchant ratio, time-since-account-creation.

The specific card-testing signature: **many distinct PANs, small amounts, high declines, short window, narrow merchant set.**

### L4 — Onboarding risk model
Consumes verification signals plus graph features available at t=0. Outputs a score *and* the ring it was assigned to.

### L5 — Fusion and retro-propagation *(this is the contribution)*
Two directions:

- **Forward:** an account's onboarding score conditions its behavioural thresholds. A borderline-onboarded account testing cards gets caught faster than a ten-year customer doing the same thing.
- **Backward (the good part):** when an account is confirmed bad, propagate that evidence back through the graph to its ring siblings — including dormant accounts that have done nothing yet — and pre-emptively raise their scores. **You block accounts before their first fraudulent transaction.** Demo this on stage: one account gets caught, forty light up red, zero of them have transacted.

Implement as belief propagation over the graph, or a GNN, or — for a hackathon — weighted score diffusion with a decay factor over hops. The simple version works and is explainable.

### L6 — Decision engine
Graduated responses, not just block:

| Score band | Action |
|---|---|
| Low | Allow |
| Medium | Step-up auth / 3DS challenge |
| Elevated | Velocity throttle — silently rate-limit attempts |
| High | Honeypot mode — return plausible responses that poison the attacker's results, so they can't tell live cards from dead ones |
| Confirmed | Block + freeze ring + case to analyst |

Every action emits reason codes.

### L7 — Analyst console + feedback
React dashboard: live event stream, ring graph visualisation, case queue, an "explain this decision" panel, and the merchant-view/network-view toggle. Analyst verdicts feed back as labels.

### Cross-institution privacy layer (differentiator)
How do two banks share "this device is bad" without sharing customers? Options to mention, pick one to prototype: hashed-indicator exchange with Bloom filters, private set intersection, or federated learning where only model updates move. Even a shallow prototype here reads as serious.

---

## 4. Tech stack options

Three tiers. Pick from Tier 1 to actually ship; cite Tier 2/3 in the doc as "production path" so judges see you know the difference.

| Layer | Tier 1 — hackathon (ship in days) | Tier 2 — production-realistic | Tier 3 — enterprise / network scale |
|---|---|---|---|
| Simulator | Python + NumPy + Faker, simple event loop or SimPy | Same, containerised, seeded scenario configs | Load-gen against a staging network |
| Event transport | Redis Streams, or an in-process queue | Kafka or Redpanda | Kafka + schema registry, Azure Event Hubs |
| Stream processing | Python consumers / Bytewax | Flink or Spark Structured Streaming | Flink on k8s |
| Feature store | Redis (online) + Parquet/DuckDB (offline) | Feast | Tecton / in-house |
| Graph | NetworkX + igraph (Leiden) | Neo4j or Memgraph | TigerGraph, or Spark GraphFrames |
| Graph ML | Score diffusion (hand-rolled) | PyTorch Geometric — GraphSAGE / GAT | Same, distributed |
| Tabular models | LightGBM / XGBoost | + calibration, SHAP | Same, with champion/challenger |
| Anomaly / unsupervised | IsolationForest, autoencoder | + River for online learning | Same |
| Sequence models | GRU or small temporal transformer over auth sequences | Same, ONNX-exported | Same |
| Rules | YAML rule file + evaluator | Durable-rules / OPA | Commercial decision engine |
| Serving | FastAPI + ONNX Runtime | + gRPC, autoscaling | Same, multi-region |
| Storage | Postgres + DuckDB | Postgres/Timescale + ClickHouse | ClickHouse / Snowflake |
| Frontend | React + Vite + Tailwind + shadcn/ui, Recharts, Cytoscape.js or react-force-graph | Same + SSE/WebSocket streaming | Same |
| Infra | Docker Compose, one command | Azure Container Apps | AKS |
| PII handling | Microsoft Presidio for detection/redaction | + vault, tokenisation | HSM-backed tokenisation |
| LLM usage | Case-narrative generation for analysts; red-team agent in the simulator | Same, with guardrails | Same |

### Recommended concrete pick, given your stack

- **Simulator + ML + graph:** Python. Non-negotiable — the libraries only exist here.
- **Real-time scoring API:** FastAPI. Keep it Python so models load natively.
- **Orchestration / gateway / mock merchant + issuer services:** Fastify + TypeScript — this is your home turf and it's genuinely the right tool for I/O-bound glue.
- **Frontend:** React + Vite + Tailwind, which you already run daily.
- **Everything in Docker Compose**, one `docker compose up` for the demo. Windows 11 + WSL2 handles this fine; do the Python work inside WSL, not native Windows, or you'll lose hours to build tooling.

### Where LLMs genuinely earn their place (don't bolt them on)
1. **Red-team agent** — proposes new evasion parameter sets against your detector inside the simulator.
2. **Analyst copilot** — turns a graph subgraph + feature vector into a two-paragraph case narrative with the evidence cited. Real analyst time saved, easy to demo.
3. **Reason-code writer** — converts SHAP values into plain-English decline explanations.

Do *not* put an LLM in the auth path. Latency budget forbids it, and saying so out loud in the pitch shows judgement.

---

## 5. Metrics to report

- Detection rate at **fixed 0.1% FPR** (headline number)
- **Attempts-to-detection** — how many card tests get through before the block. Lower is the whole point.
- **Ring recall** — of the accounts in a confirmed ring, what fraction did we flag *before* they transacted? This is the number that differentiates you.
- False decline rate on legitimate look-alikes
- p99 scoring latency
- Merchant-view vs network-view delta — the "why Mastercard" number
- Robustness curve: detection rate vs attacker sophistication level

---

## 6. Build plan

**Phase 0 — contracts first (day 1).** Freeze the event schemas, the graph node/edge types, and the scoring API contract. Everything else parallelises off this. Given you'll run several Claude Code agents at once, this phase is the whole ballgame — write the schemas as a single `contracts/` package that every module imports.

**Phase 1 (parallel):**
- Agent A: simulator + scenario configs
- Agent B: ingest, feature store, streaming features
- Agent C: identity graph + entity resolution + community detection
- Agent D: frontend shell + graph viz with mocked data from the contracts

**Phase 2:** models (baseline LightGBM → graph features → fusion), decision engine, retro-propagation.

**Phase 3:** demo polish, adversarial levels, metrics report, pitch deck.

**Cut list if time runs out:** federated/PSI layer → describe, don't build. GNN → score diffusion is fine. Sequence model → window features are fine. Never cut: the ring graph visual and the retro-propagation demo. Those *are* the pitch.

---

## 7. Kickoff brief for a Claude Code agent (paste-ready, Phase 1 / Agent A)

> Build a synthetic payments-fraud simulator in Python. It emits four JSONL streams: `onboarding_events`, `auth_events`, `session_telemetry`, `labels`. Schemas are defined in `contracts/schemas.py` — import them, do not redefine.
>
> Generate a population of N legitimate customers with stable geo, 1–3 devices, log-normal transaction amounts, and Poisson inter-arrival times with daily/weekly seasonality. Include occasional genuine life events (device change, travel).
>
> Generate fraud rings. Each ring has an operator profile controlling: device_reuse_rate, subnet_concentration, pii_recombination_rate, doc_template_reuse, dormancy_days, burst_intensity, merchant_spread. Expose three presets: `sloppy`, `moderate`, `sophisticated`.
>
> Ring accounts onboard, stay dormant for dormancy_days with light benign activity, then run card-testing bursts: many distinct PANs, small or zero amounts, high decline rate, narrow merchant set.
>
> Card numbers must come only from non-issued test ranges (4111-style). Identity verification is modelled as numeric signal fields only (template_match_score, face_embedding_id, exif_consistency, liveness_score) drawn from separate genuine/synthetic distributions — do not generate or handle any document or face images.
>
> Fraud must be 0.1–1% of all events. Chargeback labels must be emitted with a 20–60 day delay. Everything seeded and reproducible via a YAML scenario config. Include a `verify.py` that prints class balance, ring sizes, and a sanity plot of hourly volume.

---

## 8. Open questions for the team

1. Are we pitching this as a Mastercard-network-level service, an issuer-level product, or a merchant-level one? The answer changes what data we're allowed to assume. *(Recommendation: network-level — it's the only framing where our best feature is possible.)*
2. Do we prototype the privacy-preserving sharing layer, or describe it?
3. How many attacker sophistication levels do we demo — one clean win, or the full arms race?
4. Who owns the pitch deck, and does deck work start on day 1 or day 3?
