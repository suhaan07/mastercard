# Demo script and pitch outline

Twelve minutes of demo, in the order that makes the argument. Every number below
is one this repo produces; where a number is quoted, the command that prints it
is named beside it. Re-run before the pitch and re-quote — a stale figure on a
slide is the one thing a judge can catch you on.

---

## Before the room

```powershell
.\run.ps1 -Scenario moderate -Rebuild   # ~45 min cold; do this the night before
.\run.ps1 -Scenario moderate            # ~5 min warm; do this on the morning
.\run.ps1 -Test                         # 39 tests, under a minute
```

**Demo on `moderate`, quote `sophisticated`.** `sloppy` is the easier default in
the README, but its rings burst early, so its out-of-time holdout has only 135
fraud events across three accounts — a detection rate of 1.000 on three accounts
is not a number to say out loud. `moderate` is strong across the board;
`sophisticated` is where the network-view argument lands hardest.

Cold start is not a stage activity. `-Rebuild` simulates, verifies, trains and
then serves; without it the script serves what is already built. Check three
things before you walk up:

| Check | Expect |
|---|---|
| `http://127.0.0.1:5173` | console loads, header says **connected** |
| Candidate rings list | populated (the graph build ran) |
| `http://127.0.0.1:8000/docs` | scorer is up, `/health` shows both models loaded |

Have a second terminal open at the repo root. Two moments in the script want a
command run live, and both are fast.

**Fallback if the network in the room is bad:** everything is local — the
console, the gateway, the scorer, the two institution services. Nothing calls
out. The one exception is the case narrative, which uses the Claude API when
`ANTHROPIC_API_KEY` is set and falls back to a deterministic narrative when it
is not. Unset the key and the demo is fully offline.

---

## The script

### 1 — The frame (60s, no screen)

Two attacks in the problem statement; one supply chain. A synthetic identity is
manufactured, onboarded, aged quietly, then used as a test bench for stolen card
numbers, then burned. **The account is not the fraud. The account is the tool.**

Identity risk is scored once, at onboarding. Transaction risk is scored per
authorisation. Nobody scores the seam between them — and the seam is where the
whole chain is visible.

> Land this before touching the keyboard. Everything after is evidence for it.

### 2 — The testbed (60s)

`data/` and `sim/scenarios/`. We could not use real fraud data and did not want
to, so we built the world: legitimate customers with life events, fraud rings
with a seven-knob operator profile, card testing, look-alikes that *should* be
allowed, chargebacks arriving 20–60 days late, and fraud at ~0.7% of events.

Say the scope choice out loud: the simulator emits **signals a verification
vendor would emit**, never faces or documents, and only non-issued test card
ranges. It is a defence tool, not a forgery tool.

```powershell
.venv\Scripts\python.exe -m sim.verify --data data/moderate
```

### 3 — The console, network view (90s)

Point at the ring list. These are candidate rings from Louvain over the identity
graph, ranked by cohesion × suspicion × log(size) — not by density, because
density falls off quadratically with size and a household of four outranks a
51-account ring.

Click the top ring. Let the force graph settle. Name the node types: identity,
device, IP/ASN, address, phone, card token. **The shape is the point** — real
customers do not share devices and subnets like this.

### 4 — The moment (2 min) — *this is the demo*

Press **Confirm one account as fraud**.

One confirmation. Watch the siblings light up. Then read the panel underneath:

> *N* siblings raised, of which *M* **have never transacted**.

Say it plainly: those accounts have no behavioural signal of any kind. No
transaction-level model can reach them, because there are no transactions. They
are reachable only through shared infrastructure — and we blocked them before
their first fraudulent authorisation.

Measured across the whole run, not a stage trick — and quote the second line
with the first, because recall on its own is not a result:

```
ring recall before transacting     1.000   (95 of 95 dormant accounts, moderate)
flagged in total                   247     (5.8% of all identities)
of those, actually synthetic       0.611
```

Diffusion that reached most of the population would score a recall of 1.000 and
mean nothing. Saying the precision unprompted is the difference between a demo
and a claim.

*Source: `python -m eval.report --data data/moderate` → "ring recall BEFORE transacting".*

### 5 — Why Mastercard (2 min)

Flip the toggle to **merchant view**. Nodes disappear. Say what that is: not a
filter on the chart — the merchant process only ever received its own
institution's traffic. The gateway and the two mock institution services are
separate processes for exactly this reason.

Then give the measured number, which is the harder version of the same claim:

| Scenario | One merchant alone | Network view | Delta |
|---|---|---|---|
| sloppy | 0.978 | 1.000 | +2.2 pts |
| moderate | 0.947 | 0.995 | +4.8 pts |
| **sophisticated** | **0.102** | **0.794** | **+69.2 pts** |

Same trained model, same events, same fixed 0.1% FPR, both sides thresholded on
the same pool. The only difference is which events were available when the
rolling windows were built. A merchant that sees five attempts of a forty-attempt
spray cannot see a spray.

**The direction is the argument.** The delta grows with attacker sophistication,
because `merchant_spread` is exactly the knob a competent operator turns. Against
the patient ring — the one you actually want to catch — a single merchant sees
one tenth of what the network sees.

Two things to say before you are asked, because both will be found:

- Per *institution* the delta is **0.000** everywhere. In this simulator every
  one of an account's authorisations lands on its own institution, so
  institution scoping dilutes nothing. The effect is per-merchant, and claiming
  otherwise would be claiming something the data does not show.
- On the drift scenario the delta is **−0.256** — the merchant view measures
  higher. We report it and we have not explained it; the hypothesis is that
  dilution pulls drifted traffic back toward the training distribution. If a
  judge finds this before you say it, the slide is worth less.

*Source: `python -m eval.report` → "single-player view vs network view".*

### 6 — The cost function (60s)

Everything above is quoted at a **fixed 0.1% false-positive rate**, because the
defender's real cost function is false declines — a false decline costs a
lifetime of customer value plus interchange.

Two cohorts we watch on purpose:

- **Look-alikes** — a genuine small business doing many small transactions, a
  customer on a new device abroad. Reported separately.
- **Thin-file customers** — disproportionately young, migrant or low-income, and
  legitimately thin-file. Their decline rate is reported next to the thick-file
  rate; if the two diverge the model has learned a proxy.

**Say this one out loud, because it currently fails.** Thin-file customers
decline at 3-6x the thick-file rate depending on scenario. The likely route is
`acct_age_days`, which a brand-new genuine customer shares with a freshly-minted
synthetic account. The onboarding model already handles this correctly — it uses
`credit_age_ratio`, because the tell is the inconsistency between credit-file age
and declared age, not youth — and the behavioural model has no equivalent
normalisation. That is the next piece of work, and naming it yourself is worth
more than being asked.

Look-alikes, by contrast, decline *below* the general legitimate rate in every
scenario. That cohort is not what this model gets wrong.

*Source: `eval/report.py` → "false declines".*

### 7 — The adversary adapts (90s)

A static rule set has a half-life, so we ran the attacker against our own
detector:

```powershell
.venv\Scripts\python.exe -m detect.redteam --data data/moderate --rounds 8
```

It mutates the ring's operator profile toward whatever gets through, and prices
each evasion by what it costs the operator to run. Show the knobs it turned.

Then the robustness curve — same detector, same fixed FPR, only the operator
profile changes:

```powershell
.venv\Scripts\python.exe -m eval.robustness
```

| Scenario | AUC | Detection @ 0.1% FPR | Ring recall pre-transaction |
|---|---|---|---|
| sloppy | 1.0000 | 1.000 | 1.000 |
| moderate | 1.0000 | 0.996 | 1.000 |
| sophisticated | 0.8515 | 0.799 | 0.512 |
| drift | 0.9849 | 0.703 | 1.000 |

The curve bends rather than falling off a cliff: 0.70 at the hardest setting, at
a fixed 0.1% false-positive rate, with zero accounts escaping entirely. Ring
recall is the piece that degrades most against the patient operator, which is
the honest read — long dormancy and low device reuse are exactly what the graph
depends on.

### 8 — The biometric decision (90s)

Only if the room is technical, and skip it if you are behind.

Face embeddings are the obvious choice and the wrong one: they are
**invertible** — an embedding database is a face database with extra steps — and
**linkable**, so the same face at two institutions identifies a person across
them and cross-institution sharing dies on GDPR/DPDP grounds.

Instead, FlyHash (Dasgupta, Stevens & Navlakha, *Science* 2017): sparse random
expansion, winner-take-all, out comes a sparse binary tag. A tag is a **set**,
so near-duplicate detection is set intersection — which composes directly with
the Bloom-filter exchange. Face matching and cross-institution sharing become
one mechanism.

| Property | Measured |
|---|---|
| Same face, different institution seed | 0.0263 overlap — chance is 0.0257 |
| Hypervector linker at population scale | precision 1.000 |

State the claim discipline out loud: revocability, unlinkability and lossy
construction under a stated threat model — **not** cryptographic
non-invertibility, which is unproven for FlyHash. Overclaiming it does not
survive the first technical question.

```powershell
.venv\Scripts\python.exe -m privacy.psi
```

3/3 planted shared identities matched, 0 false positives of 400, 58 bytes per
identity. Only a bit array crosses the wire.

### 9 — The close (45s)

- Latency: **p99 5.4 ms** of a ~50 ms budget, measured over 2,000 scored events
  through the service (and measured while a training job was competing for the
  CPU, so it is a conservative figure). The auth path is a feature lookup
  and one gradient-boosted model — no graph traversal, no image work, no LLM.
- Where the LLM does earn its place: the red-team search, and the case narrative
  in the console — both **off** the auth path. Say that you deliberately kept it
  out of the auth path; it signals judgement.
- One line to end on: *catching the tool one at a time, after it has been used,
  is the thing this replaces.*

---

## If you have 5 minutes instead of 12

Sections 1, 4, 5, 9. The frame, the retro-propagation moment, the network delta,
the close. Everything else is supporting material.

---

## Deck outline

Twelve slides, one idea each. The demo carries slides 5–7; those slides are
backup for when the demo cannot run.

| # | Slide | The one thing it says | Evidence on it |
|---|---|---|---|
| 1 | One supply chain | Manufacture → onboard → age → weaponise → monetise | The five-stage diagram |
| 2 | The unscored seam | Identity scored once, transactions scored each; nobody scores the relationship | Where each existing control sits |
| 3 | The testbed | We own the ground truth, and the scope choice is deliberate | Class balance, ring sizes, label delay |
| 4 | Architecture | Seven layers, one contract per seam | The L1–L7 diagram |
| 5 | **Ring-level, not account-level** | One confirmation, ninety-five accounts, zero transactions | Ring recall **1.000** at 0.61 precision |
| 6 | **Why the network sees it** | A merchant cannot see a spray it only got five attempts of | Sophisticated ring: **0.102 vs 0.794** |
| 7 | Robustness | Detection bends, it does not break | **0.70-1.00** across four operator profiles |
| 8 | False declines | Everything at a fixed 0.1% FPR — and the fairness check is currently failing | Look-alikes below baseline; thin-file 3-6x |
| 9 | Privacy by construction | Unlinkable tags, Bloom-filter exchange, 58 bytes per identity | Cross-seed overlap at chance |
| 10 | Latency and explainability | p99 ~5 ms; every block carries a reason code | The reason-code list |
| 11 | What we got wrong | Twelve findings, several of which would have produced impressive numbers meaning nothing | The README's list |
| 12 | Production path | Tier 1 shipped, Tier 2/3 named | The stack table |

**Slide 11 is not filler.** A team that shows the AUC of exactly 1.0000 it
refused to believe, and what was actually wrong, reads as a team that can be
trusted with the other eleven slides. Keep it.

---

## Questions to have an answer ready for

**"Isn't this just graph analytics?"** The graph proposes candidates; the
contribution is the direction evidence flows. Forward, onboarding conditions
behavioural thresholds. Backward, a confirmation reaches dormant siblings.
Backward is the part nobody runs, and it is the only thing that catches an
account before its first fraudulent transaction.

**"What is your false positive rate really?"** 0.1%, fixed, and every number is
quoted at it. Look-alikes and thin-file customers are reported separately
because those are where a false decline actually hurts.

**"How does this work across banks without sharing data?"** Bloom-filter
indicator exchange over MinHash bands of per-consortium tags. 58 bytes per
identity, a bit array on the wire, and the internal tags stay mutually
unlinkable. Prototyped, not described — `python -m privacy.psi`.

**"What happens when the attacker adapts?"** We ran that. `detect/redteam.py`
searches the operator profile against our deployed detector and prices each
evasion. The robustness curve is the answer, and where it dips we say so.

**"Is the LLM in the authorisation path?"** No, and deliberately not — the
latency budget forbids it. It writes case narratives and drives the red-team
search, both off the path.

**"Are these real faces?"** No faces or documents are generated anywhere. The
identity side is modelled as the numeric signals a verification vendor emits.
The image pipeline consumes generated faces from a research corpus rather than
manufacturing them, and the artifact numbers in the README are from the
procedural backend and labelled as such.
