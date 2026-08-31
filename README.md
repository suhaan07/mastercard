# Synthetic Identity as Fraud Infrastructure

Mastercard Innovation Challenge entry. Treats synthetic-identity manufacture and
card testing as **one supply chain** and scores the seam between them: the
relationship between how an account was born and how it behaves in its first
weeks.

Identity risk is scored once, at onboarding. Transaction risk is scored per
authorisation. Nobody scores the seam — and the account is not the fraud, the
account is the *tool*.

---

## The four claims, and where each is demonstrated

| Claim | Where | Measured |
|---|---|---|
| Ring-level, not account-level | `detect/fusion.py` | Ring recall before first transaction **1.000**, flagging 4-7% of identities at 0.61-0.80 precision |
| Network view beats merchant view | `gateway/src/institution.ts`, `eval/report.py` | Against the sophisticated ring, one merchant **0.102** vs network **0.794** |
| Biometric similarity without biometric storage | `biohash/` | Same face under two institution seeds overlaps at **0.0263**, chance is 0.0257 |
| Detection holds as the attacker adapts | `sim/scenarios/`, `detect/redteam.py` | **0.70-1.00** across four operator profiles at a fixed FPR |

Everything is reported at a **fixed 0.1% false-positive rate**, because the
defender's real cost function is false declines — and on the out-of-time
holdout, not on the data the model was fitted to.

Two results run against us and are reported anyway: the thin-file fairness check
**fails** (a 3-6x decline-rate divergence, mechanism identified, unfixed), and
the network-view delta **inverts** on the drift scenario. Both are in
[Measured results](#measured-results).

---

## Quick start

```powershell
python -m venv --system-site-packages .venv
.\run.ps1 -Scenario moderate -Rebuild  # simulate, verify, train, serve
.\run.ps1 -Stop
```

Console at <http://127.0.0.1:5173>, scorer docs at <http://127.0.0.1:8000/docs>.
The demo run order, with the number each step should show, is in
[DEMO.md](DEMO.md). `.\run.ps1 -Test` runs both suites and exits.

Individual stages:

```powershell
.venv\Scripts\python.exe -m sim.run --scenario sophisticated
.venv\Scripts\python.exe -m sim.verify --data data/sophisticated --plot
.venv\Scripts\python.exe -m detect.graph.communities --data data/sophisticated
.venv\Scripts\python.exe -m detect.models.train --data data/sophisticated
.venv\Scripts\python.exe -m eval.report --data data/sophisticated
.venv\Scripts\python.exe -m eval.robustness      # detection vs sophistication
.venv\Scripts\python.exe -m detect.redteam --data data/moderate --rounds 8
.venv\Scripts\python.exe -m detect.copilot --data data/moderate # case narrative
.venv\Scripts\python.exe -m privacy.psi          # cross-institution demo
.venv\Scripts\python.exe tests\test_biohash.py   # 16 property tests
.venv\Scripts\python.exe tests\test_pipeline.py  # 23 end-to-end tests
```

---

## Architecture

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

The gateway is a real network hop, and the mock merchant/issuer services each
hold only their own institution's traffic — so the merchant-vs-network delta is
a property of what a process can see, not a filter applied to a chart.

---

## The biometric decision: FlyHash, not embeddings

Face embeddings are the obvious choice and the wrong one. They are **invertible**
— template-inversion attacks reconstruct a recognisable face, so an embedding
database is a face database with extra steps — and **linkable**, so the same
embedding at two institutions identifies a person across them and cross-
institution sharing dies on GDPR/DPDP grounds.

Instead, after the fruit fly's olfactory circuit (Dasgupta, Stevens & Navlakha,
*Science* 2017): sparse random **expansion** into a wide layer, then
winner-take-all keeping the top ~5%. The output is a sparse binary **tag**.

Three properties an embedding cannot offer:

1. **A tag is a set, not a vector.** Near-duplicate detection is set
   intersection, so it composes directly with the Bloom-filter exchange in
   `privacy/psi.py` — face matching and cross-institution sharing become *one*
   mechanism.
2. **Revocable and unlinkable.** Per-institution secret seeds mean the same face
   yields non-comparable tags at different institutions; reseeding revokes.
3. **Lossy by construction.** Winner-take-all discards magnitude and keeps only
   which cells won.

Bound with hyperdimensional binding (Kanerva), an identity's face, document,
address and device tags become one hypervector — and that bundle turns out to be
by far the strongest linker, because a coincidental match on any single
attribute cannot produce a match on the bundle.

**Claim discipline:** revocability, unlinkability and lossy construction under a
stated threat model (`biohash/flyhash.py::THREAT_MODEL`). *Not* cryptographic
non-invertibility — that is unproven for FlyHash, and overclaiming it does not
survive a technical question.

### What we generate, and what we deliberately do not

The image pipeline is real. The AI-generated face class is meant to come from a
published deepfake-detection research corpus — we *consume* generated faces
rather than manufacture them. `biohash/images.py` keeps a seeded procedural
backend behind the same interface so the pipeline always runs; **all artifact
numbers below are from that procedural backend** and should be re-quoted against
a real corpus before they go in a deck.

Identity documents are never rendered. Template reuse is a *structural* signal,
so it is detected structurally on visibly non-realistic layouts.

All image work happens at onboarding. The auth path is feature lookup plus one
gradient-boosted model — no image work, no graph traversal, no LLM.

---

## Where the LLM sits, and where it does not

Not in the authorisation path. The budget there is ~50 ms for scoring inside a
round trip of a few hundred, and a model call is two orders of magnitude away
from that. The auth path is a feature lookup and one gradient-boosted model.

It earns its place twice, both off that path:

1. **Red-team search** (`detect/redteam.py`) — mutates the ring's operator
   profile against the deployed detector and prices each evasion by what it
   costs the operator to run.
2. **Case narrative** (`detect/copilot.py`, `POST /narrate/{ring_id}`) — reads a
   scored subgraph and writes the paragraph an analyst would otherwise write.
   It sees tokens, counts and scores; never PII, and it decides nothing. Every
   figure in the prose is passed in, so a hallucinated number is contradicted by
   the evidence block rendered beside it.

Without `ANTHROPIC_API_KEY` the narrative falls back to a deterministic template
that is good enough to ship — the demo has no dependency on a secret or on the
network.

## Measured results

Every number below is produced by this repo — `python -m eval.report --data
data/<scenario>` for the detection tables, `python -m eval.robustness` for the
curve — and everything is quoted at a **fixed 0.1% false-positive rate**.

Scoring is on the **out-of-time holdout**, the same split the model was trained
against. Rolling windows are still built over the whole stream, because an
event's features depend on everything before it, but the training period is not
scored. That matters: scoring it too was inflating every figure here.

### Detection, at 0.1% FPR

| Scenario | AUC | Detection | Realised FPR | Median attempts to detection | Accounts never caught |
|---|---|---|---|---|---|
| sloppy | 1.0000 | 1.0000 | 0.00100 | 1 | 0 |
| moderate | 1.0000 | 0.9961 | 0.00099 | 1 | 0 |
| sophisticated | 0.8515 | **0.7993** | 0.00099 | 2 (p90 6) | 0 |
| drift | 0.9849 | 0.7028 | 0.00099 | 1 | 0 |

**Read the sloppy row with care.** Its rings burst early, so its holdout
contains only 135 fraud events across 3 accounts — a detection rate of 1.000 on
three accounts is not a result worth quoting. `sophisticated` (1,131 fraud
events, 26 accounts) and `drift` (1,090 events, 59 accounts) are the rows that
carry weight. The fix, if it matters later, is to widen `days` or stagger ring
onboarding so testing traffic spans the whole run.

### Ring recall before first transaction — the differentiating number

| Scenario | Never transacted | Flagged anyway | Recall | Flagged in total | Of those, synthetic |
|---|---|---|---|---|---|
| sloppy | 95 | 95 | **1.000** | 187 (4.4% of identities) | 0.802 |
| moderate | 95 | 95 | **1.000** | 247 (5.8%) | 0.611 |
| sophisticated | 121 | 62 | 0.512 | 146 (3.4%) | 0.630 |
| drift | 119 | 119 | **1.000** | 282 (6.5%) | 0.667 |

One analyst confirmation per planted ring, propagated through shared
infrastructure. The right-hand columns are the ones that keep the left-hand ones
honest: diffusion that reached most of the population would score a recall of
1.000 and mean nothing, so the report prints how much it flagged in total and
what fraction of that is actually synthetic.

### Single-player view vs network view — the "why Mastercard" number

Same trained model, same events, same fixed FPR, both sides thresholded on the
same pool. The only difference is which events were available when the rolling
windows were built.

| Scenario | One merchant alone | Network view | Delta |
|---|---|---|---|
| sloppy | 0.978 | 1.000 | +0.022 |
| moderate | 0.947 | 0.995 | +0.048 |
| **sophisticated** | **0.102** | **0.794** | **+0.692** |
| drift | 0.963 | 0.708 | **−0.256** |

The sophisticated row is the argument: against the patient operator who sprays
across fourteen merchants, a single merchant catches **10%** of what the network
catches **79%** of. The advantage grows with attacker sophistication, because
`merchant_spread` is exactly the knob a competent operator turns.

Two honest qualifications:

- **Per *institution*, the delta is 0.000 in every scenario.** In this simulator
  every one of an account's authorisations lands on its own institution, so
  institution scoping does not dilute an account's windows at all. The network
  advantage here is a *per-merchant* effect, and saying otherwise would be
  claiming something the data does not show.
- **The drift scenario runs the other way**, and we have not explained it. The
  attacker's parameters shift mid-run, and the merchant-scoped view measures
  *higher* than the network one. The plausible mechanism is that dilution pulls
  drifted traffic back toward the distribution the model was trained on while
  also compressing the negatives, but that is a hypothesis, not a finding. It is
  reported rather than dropped, because a delta quoted only where it is
  favourable is not a measurement.

### Fairness: thin-file customers

| Scenario | Thin-file decline rate | Thick-file | Ratio |
|---|---|---|---|
| sloppy | 0.00114 | 0.00098 | 1.2x |
| moderate | 0.00299 | 0.00079 | 3.8x |
| sophisticated | 0.00265 | 0.00082 | 3.2x |
| drift | 0.00414 | 0.00067 | 6.2x |

**This is the check failing, and it is reported as such.** Thin-file customers
are disproportionately young, migrant or low-income and are legitimately
thin-file; a decline rate three to six times the thick-file rate means the model
has learned a proxy. The likely route is `acct_age_days`, which a brand-new
genuine customer and a freshly-minted synthetic account share. The onboarding
model already handles this correctly — it uses `credit_age_ratio`, because the
tell is the *inconsistency* between credit-file age and declared age, not youth
— and the behavioural model has no equivalent normalisation. Fixing it means
retraining and re-measuring, which is the next piece of work rather than
something quietly patched before a demo.

### Look-alike false declines

Genuine small businesses doing many small transactions, and customers on a new
device abroad, decline at 0.00000–0.00067 against a general legitimate rate of
0.00108–0.00125 — i.e. *below* baseline in every scenario. The look-alike
cohort is not what this model gets wrong.

### Biometric layer (`tests/test_biohash.py`, 16/16 passing)

| Property | Measured |
|---|---|
| Unrelated tags overlap | 0.0253 (predicted chance 0.0257) |
| **Same face, different institution seed** | **0.0263 — indistinguishable from chance** |
| Face linker (ring vs population) | AUC 0.996 |
| Hypervector linker at population scale | **precision 1.000**, zero false links |
| GAN-artifact AUC by generator quality | sloppy 0.999 · moderate 0.961 · sophisticated 0.800 |

### Ring detection (`detect/graph/communities.py`)

Louvain over the identity graph, communities ranked by cohesion x suspicion x
log(size). Recall is over the planted rings; F1 is against the best-matching
community for each.

| Scenario | Ring recall | F1 |
|---|---|---|
| sloppy | 1.000 | 0.967 |
| moderate | 1.000 | 0.856 |
| drift | 0.995 | 0.941 |
| sophisticated | 0.787 | 0.819 |

Unchanged by the entity-resolution fix, which is itself informative: the 1,078
zero-precision email-shape links that were flooding retro-propagation were not
contributing anything to community detection either.

### Cross-institution exchange (`privacy/psi.py`)

3/3 planted shared identities matched (21–24 of 32 minhash bands) with **0 false
positives from 400**, at 58 bytes per identity. Only a bit array crosses the
wire.

---

## Things that were wrong, and what fixed them

Kept because each one is a real finding, and several would have produced
impressive-looking numbers that meant nothing.

- **Unrelated faces overlapped at 0.26** against a 0.026 chance floor. A
  low-frequency DCT descriptor is dominated by skin and hair, so two people with
  similar colouring looked identical to it. Fixed by high-passing before the DCT
  plus population whitening.
- **Artifact AUC was exactly 1.0000 at every generator quality** — a red flag,
  not a result. The simulated camera pipeline was too consistent. Adding real
  8×8 block-DCT compression (whose periodic frequency signature is the *same* as
  transposed-convolution upsampling) produced the honest 0.999 → 0.800 curve.
- **The LSH banding was broken.** It hashed consecutive slices of the index set
  and required ~25 indices to match exactly; same-ring recall measured **0.5%**.
  Similarity means overlapping sets, not identical runs. Replaced with exact
  sparse computation for entity resolution and proper MinHash banding for the
  privacy exchange.
- **Community scoring on edge density ranked every planted ring 104th–119th of
  130.** Density falls off quadratically with size, so a household of four
  outscored a 51-account ring. Weighting by `log(size)` moved all four rings into
  the top nine.
- **A 0.6% pairwise false-link rate is catastrophic at population scale.** Across
  9.15M pairs it produced 19,404 false links that chained legitimate people into
  phantom clusters — 93% of "legitimate" communities were held together by face
  similarity alone. Fixed with per-attribute thresholds; the hypervector reaches
  precision 1.000.
- **The behaviour model's holdout had zero positives, twice.** First because
  chargebacks arrive 20–60 days late, so recent events have no labels — resolved
  by separating *training* labels (delayed, realistic) from *measurement* ground
  truth. Then because ring onboarding was compressed into the first third of the
  run, making any time-ordered holdout structurally fraud-free.
- **One confirmation lit up 73% of the population.** Ring recall was 1.000 and
  meant nothing, because retro-propagation was reaching most of everybody.
  The cause was a single link kind: identities were linked when their email
  handles shared a *construction shape*, and the common shapes are just what
  most people's email looks like — 1,078 such links on the sloppy scenario at
  **zero** same-ring precision, while every other link kind ran between 0.26 and
  1.00. Gating the shape on population rarity cut propagation from 115 flagged
  identities to 12, kept 10 of 11 true ring members, and moved precision from
  0.18 to 0.83. Recall without precision is not a result, so the report now
  prints both.
- **The report was scoring its own training data.** Metrics were computed over
  every event in the run, including the period the model was fitted on. On the
  easy scenarios that flattered the numbers; on the sophisticated one it did
  something worse — in-sample rows saturate the model's probabilities at exactly
  1.0, enough negatives tied at the top that no threshold could meet the FPR
  budget, and the reported detection rate collapsed to **0.000** while the same
  model scored 0.799 on its own holdout. Windows are still built over the whole
  stream, because an event's features depend on everything before it, but only
  the out-of-time holdout is scored.
- **The merchant beat the network, which is not a believable result.** Each view
  was thresholded on its own negatives at a fixed 0.1% FPR — faithful, since a
  merchant does calibrate on its own traffic, but at that FPR a 2,500-event
  merchant has a budget of two and a half false positives, so the threshold was
  fit to two or three points. The drift scenario came back at −0.62, the
  merchant apparently outperforming the network view. Both sides are now
  thresholded once on the pooled result, which puts the same number of negatives
  behind each threshold and leaves feature construction as the only difference.

- **Everything downstream of training died on a module name.** The fitted model
  was pickled as `self`, so the class was recorded as `__main__.TrainedModel` --
  `__main__` being `detect.models.train` at fit time and something else
  everywhere it was loaded. The scorer, the report and the red-team search all
  failed at load with `AttributeError`. Models are now saved as a dict payload,
  which carries no class reference at all.
- **The serving path was scoring in hard labels.** `TrainedModel.predict` called
  `LGBMClassifier.predict`, which returns 0/1, while every threshold in the
  report was calibrated on `predict_proba`. The report looked right because it
  called `predict_proba` itself; the API silently collapsed to two score values,
  so the graduated bands -- step-up, throttle, honeypot -- could never fire.
  Nothing failed. That is what made it worth a test of its own.
- **Retro-propagation crashed on a tie.** Evidence paths were pushed into a heap
  as the last element of a tuple, so two entries with the same score and node
  fell through to comparing two dicts. Fixed with a monotonic counter, which is
  also what makes the traversal order deterministic.
- **A fixed-FPR threshold that was not fixed.** Taking the `(1 - fpr)` percentile
  of the negative scores is wrong when scores are tied, and a saturating model
  ties hard: with the top 0.23% of negatives all at exactly 1.0, the realised
  FPR came out at 0.23% against a 0.1% budget. Every headline number is quoted
  at a fixed FPR, so this quietly invalidated the comparison it existed to make.
  The threshold is now the lowest score at which the budget actually holds.
- **The behaviour model can now refuse to train.** One scenario had been fitted
  on 22 labelled positives, of which ground truth said none were fraud, and
  reported AUC 0.985. Below 25 positives the trainer raises and names the fix
  (widen the run, or start the rings earlier) rather than returning a model.
- **The test suite did not import.** `cluster_key` was replaced by MinHash
  banding and the test that covered it was never updated, so the "16/16
  passing" claim had been untrue since the LSH fix.

- **The onboarding model scored AUC 0.500.** 85% of customers onboard before the
  observation window, so a time-ordered split put every positive on one side.
  The model now trains on in-window applications only, which is also the honest
  framing: it scores *new* applications.

---

## Layout

```
contracts/   frozen schemas, graph types, decision types  (Phase 0)
biohash/     FlyHash, HDC binding, image pipeline, artifact detector
sim/         population, rings, card testing, look-alikes, scenarios
detect/      ingest, graph, features, models, fusion, decision, red team, copilot
privacy/     Bloom-filter / MinHash indicator exchange
api/         FastAPI scorer
gateway/     Fastify gateway + mock merchant and issuer services
ui/          React analyst console
eval/        metrics at fixed FPR, robustness curve across scenarios
tests/       property tests (biohash) and end-to-end tests (pipeline)
```

## Constraints honoured

| Constraint | Where |
|---|---|
| p99 auth scoring under ~50 ms | `api/main.py`; O(1) window updates in `detect/features/stream.py` |
| Everything at a fixed FPR | `eval/report.py`, `threshold_for_fpr` |
| No PII downstream of L1 | `detect/ingest.py::PIIVault` |
| Every block carries a reason | `contracts/decisions.py::REASON_TEXT` |
| Fairness on thin-file customers | `eval/report.py` reports their decline rate separately — **and it currently diverges 3-6x; see Measured results** |
| Out-of-time evaluation | `eval/report.py` scores only events after the model's own `split_ts` |
| Test card ranges only | `sim/world.py::TEST_BIN` |
