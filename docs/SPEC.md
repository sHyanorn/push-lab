# push-lab Specification

## 1. One-line summary

An offline lab for push notification recommendation: it builds the full pipeline (trigger, recall, coarse ranking, fine ranking, delivery policy with volume, timing and copy selection) on public data, and labels every result with how far it can be trusted when there is no live traffic.

## 2. What the project covers

| Area | Where it shows |
|---|---|
| Algorithms and data structures | Two-tower recall with an ANN index, coarse-to-fine ranking cascade, budget allocation under a global volume cap, bandit selection |
| Machine learning | GBDT ranking with two targets, meta-learners and a causal forest for uplift |
| Causal inference and evaluation | Qini and AUUC on randomized trials, off-policy estimators checked against known ground truth |
| Engineering | Reproducible data pipeline, simulator with explicit assumptions, versioned LLM-generated copy, CPU-only budget, CI |

## 3. Core story

This is a hypothesis to be tested, not a result. The comparisons are pre-registered in `docs/METHODS.md` and `docs/DECISIONS.md` before the simulations run, and are reported as they come out, including if the story does not hold.

1. A policy that sends the highest-scoring item to every user every day wins on clicks in week one
2. In the simulator it loses users to fatigue and opt-outs by week four
3. A policy that sends only when the predicted uplift clears a threshold, under a global volume budget, at each user's usual active hour and with copy chosen by a bandit that avoids repeating itself, sends fewer pushes and keeps more users active
4. The conclusion holds across most of the plausible simulator settings, and the report shows where it flips
5. Every number carries an evidence tier (Section 5), so a reader can see which parts rest on real randomized data and which rest on simulator assumptions

## 4. Scope

### Required

- Data layer
  - KuaiRand-Pure, Hillstrom, Criteo Uplift v2.1, Open Bandit Dataset small sample (Section 9)
  - Download scripts with checksums; Parquet storage; a data quality report
  - Synthetic fixtures with the same schemas, so CI never downloads data
- Trigger
  - One decision per user per send slot (four slots a day); the policy decides whether to send, what and with which copy
- Candidate pool and recall
  - Sources: personal (two-tower) and trending (recent popularity), deduplicated
  - Two-tower model in PyTorch with in-batch negatives and log-Q correction; faiss HNSW index checked against exact search
  - Baseline: popularity
- Coarse and fine ranking
  - Coarse: two-tower score, keeps the top N of the pool
  - Fine: LightGBM with two targets (click, long view) fused by a weighted score; only the top-1 item is pushed
  - Truncation study: N in {50, 100, 200, 500, all}; with 7.5k items the fine ranker can score the full pool, which gives the reference for what truncation loses
  - Trained on standard logs, evaluated on random exposures (Section 9, split rules)
- Uplift methods on randomized trials (Tier A)
  - S-, T- and X-learners on LightGBM; causal forest from EconML
  - Own Qini and AUUC implementation, cross-checked against scikit-uplift on Hillstrom
  - Budget-constrained targeting: uplift ranking against response ranking and random, at several budgets
- Push simulator
  - Users, days and send slots built from KuaiRand-Pure: response model fitted on its own half of the random exposures; hourly activity profiles from organic logs
  - Response base rate, fatigue, opt-out, return and copy effects as parametric models with documented ranges (Section 6)
  - Exploration policy: epsilon-greedy over send or skip, slot and template, with logged propensities
- Delivery policies
  - Baselines: fixed daily push at a fixed slot, score threshold
  - Uplift-gated send, with the uplift model fitted on the simulator's exploration logs
  - Global volume budget allocated by marginal gain: one multiplier per day in the primary analysis, a single Lagrange multiplier as a secondary variant (D028)
  - Send time: each user's most active slot, estimated from earlier logs than the ones the simulator uses, with a swept shift and noise on top (Section 6)
  - Sensitivity sweep over the simulator parameters
- Copy (Section 7)
  - Template families filled only from fields the data has; variants generated once with the Claude API and committed with a manifest
  - Rule-based filters: schema, length, placeholder validity, near-duplicates, banned claims
  - Template selection with a recovering bandit (novelty penalty that decays with days since the user last saw a template), against uniform and greedy without the penalty
- Off-policy evaluation
  - DM, IPS, SNIPS and DR, written in this repository
  - Validation: Open Bandit Dataset (estimate the BTS policy from random logs, compare with its on-policy value) and simulator logs against simulator truth
- Evaluation report: evidence matrix, bootstrap confidence intervals, negative results
- Demo notebook and a static HTML report

### Optional, in order of value

1. Duolingo notification data: replicate the sleeping, recovering bandit off-policy on one sampled training part (D010)
2. LLM judge score as a bandit prior, with its correlation to the simulated copy effect swept from zero
3. DR with optimistic shrinkage; DR-learner; ALS baseline; recent-favorite-author recall source
4. DeepFM or DCN ranker against LightGBM (FuxiCTR)
5. KuaiRand-1K (4.4M videos) to show ANN recall and coarse-ranking latency at scale
6. Full Open Bandit Dataset (downloaded once locally) for an estimator comparison on the full Men's campaign (D042)
7. causalml cross-check in a separate Python 3.12 environment

### Out of scope

- Any claim about real push CTR, DAU or retention lift, or that one piece of generated copy works better than another for real users
- Serving infrastructure, streaming triggers, real-time features
- A web dashboard

## 5. Evaluation protocol (project focus)

There is no live traffic, so each claim is tied to one of three tiers and states its limits.

| Tier | Evidence | What it can support | What it cannot |
|---|---|---|---|
| A | Randomized real data: KuaiRand random exposures, Hillstrom, Criteo, Open Bandit Dataset | A component ranks or targets better in that domain; on the Open Bandit Dataset, how close an off-policy estimator comes to a known policy value | Anything about push, fatigue or long-term effects |
| B | Off-policy estimates from an estimator first validated against a known answer | The value of a new policy within the support of the logging policy | Policies outside that support; long horizons |
| C | Simulator runs with a sensitivity sweep | Policy A beats B across a stated share of assumptions, or flips at a stated threshold | Real-world effect sizes |

No result reached Tier B: on the only real logs with two policies and known propensities (the Open Bandit Dataset), no estimator passed the check on the full Men's campaign (D046; D063 records the post hoc intervals over ten-minute blocks). The measured accuracy of the estimators there is a Tier A result about the estimators, not a policy value.

Rules:

- Primary metrics and comparisons are fixed in `docs/METHODS.md` before results are computed
- Confidence intervals by bootstrap over users; Holm correction across policy comparisons
- Every off-policy estimate is reported with all estimators side by side, the effective sample size and the largest weight; disagreement beyond the intervals is reported as inconclusive
- Qini and AUUC state the curve variant, the treatment contrast and the test-set size
- Uplift results on Hillstrom and Criteo validate the methods (Tier A); the gate used in the simulator is fitted on simulator logs and its results are Tier C
- Send-time and copy results are Tier C
- Simulator results are never reported without the sweep

## 6. Simulator assumptions

| Component | Source | Treatment in the report |
|---|---|---|
| Response to a pushed item | Fitted on one half of KuaiRand random exposures (tab 1); scaled by a base-rate factor, since in-feed click rates (about 18%) are far above push open rates | Factor swept |
| Active hours | KuaiRand logs (`time_ms`, D023) from 04-22 to 05-08 | Proxy: in-app activity is assumed to predict push open times |
| Send-time policy input | Profiles from 04-09 to 04-21 logs (D025), plus a swept shift (hours) and noise | Prevents the policy from reading the simulator's own profile |
| Novelty or fatigue | Functional form from Yancey and Settles (2020): penalty `gamma * 0.5^(d/h)` | Swept, including zero |
| Opt-out hazard | Increases with recent volume; shape after Zhang et al. (2023) (D025) | Swept, including zero |
| Return after a push | Visit probability raised by the response, decays over days | Swept |
| Copy effect | Each template gets a latent multiplier on response: family effect plus template noise | Spread swept, including zero (then no copy policy should beat uniform) |
| Annoyance (added in M4, D023) | Recent volume lowers organic visits | Elasticity swept, including zero; conclusions also reported at zero |

All parameter ranges and their sources are in `docs/METHODS.md`.

## 7. Copy

KuaiRand has no text: `tag` holds anonymous category ids and there are no titles or captions. Copy is therefore written per template, and a template may only use placeholders the data can fill at send time.

| Family | Example placeholders | Source field |
|---|---|---|
| Creator update | a creator the user watched recently | `author_id`, user history |
| Popular now | the item's rank among recent popular videos | trending list before the cutoff (D035) |
| Personal pick | none | ranker score only |
| Short or long watch | duration bucket | `video_duration` |
| Welcome back | days since last visit | user history |
| Time of day | morning, evening | send slot |

- Generation: one request per family asks Claude for 16 English variants, of which the first 10 that pass the filters are kept (D036) (title up to 50 characters, body up to 100), returned as structured JSON. The model is passed on the command line; the script records the model the API reports, effort, prompt, request ids, token usage and date in `data/creative/manifest.json`. Current models do not accept sampling parameters, so a rerun gives different text; the committed outputs are the reference and nobody needs a key to use them
- Filters before use: schema, length, placeholder validity, near-duplicate removal (character n-gram Jaccard), and a banned-claims list (no invented numbers, urgency or rewards)
- Rejected sources: KuaiRec captions (Chinese, and its video ids do not join KuaiRand), MIND titles (research-only license), Duolingo (no template text)
- Limitations: copy is not tied to item content beyond these fields; the simulated copy effect is not derived from the text, so results show that the selection policy learns and handles novelty decay, not which text works

## 8. Architecture

```
download ──▶ Parquet ──▶ features                      creative/ (committed JSON)
                            │                                    │
          ┌─────────────────┼──────────────────┐                 │
          ▼                 ▼                  ▼                 │
   candidate pool     uplift methods      response model,       │
 (two-tower + faiss,  (Tier A: Hillstrom,  activity profiles     │
     trending)         Criteo)                 │                 │
          │                                    ▼                 │
          ▼                              push simulator ◀── swept parameters
   coarse rank (two-tower, top N)              │
          │                                    ▼
          ▼                      exploration logs + propensities
   fine rank (LightGBM, top-1)       │                    │
          │                          ▼                    ▼
          │                 uplift gate (fit)       OPE estimators
          │                          │                    │
          └────────▶ delivery policy ◀┴───────────────────┼──── template bandit
              (send gate, budget multiplier, slot)         │
                            │                             │
                            ▼                             ▼
                 evaluation report (tiers, CIs) ──▶ notebook, static HTML
```

Ranker and uplift scores are precomputed per user and day outside the slot loop; decision logs are written only for exploration and OPE runs.

## 9. Data

| Dataset | Use | License | Size |
|---|---|---|---|
| KuaiRand-Pure | Recall, ranking, simulator | CC BY-SA 4.0 (LICENSE in the archive; Zenodo metadata says CC BY 4.0) | 47 MB; 27,285 users; 1,436,609 standard-log rows and 1,186,059 random-exposure rows |
| Hillstrom | Uplift, CI-sized; contrast: women's e-mail against no e-mail | None stated | 64,000 rows, about 21k per arm |
| Criteo Uplift v2.1 | Uplift at scale | CC BY-NC-SA 4.0 | 13,979,592 rows; train on a 10% sample, evaluate on the remaining rows |
| Open Bandit Dataset (small) | OPE validation, Men's campaign (34 actions) | CC BY 4.0 (Saito et al. 2021) | 10k rows per policy and campaign |
| Duolingo notifications (optional) | Template bandit replication | CC BY-NC 4.0 | about 9.35 GB compressed; one part sampled locally |

KuaiRand split rules:

- Only `tab == 1` rows: `is_click` means a click in the two-column UI there and a valid play in the single-column UI, and 99% of random exposures are tab 1
- Ranker and two-tower: train on standard logs from 2022-04-08 to 04-21; features frozen at 04-21
- Random exposures (04-22 to 05-08) split by user into two halves: one evaluates the rankers, the other fits the simulator response model

No raw data or row-level samples are committed. CI runs on synthetic fixtures.

## 10. Compute budget

Target: 4 vCPU, no GPU, 16 GB RAM. Full reproduction under one hour; CI under ten minutes. Measured: about four hours, mostly the three sweeps (D050).

| Step | Estimate |
|---|---|
| Two-tower on about 1M training rows, embedding 32, 5 epochs | under 10 minutes |
| faiss index over 7.5k items | seconds |
| LightGBM ranker, about 1M rows | 1 to 2 minutes |
| Truncation study, fine ranker over the full pool for the evaluation users | a few minutes |
| Meta-learners on Criteo 10% (1.4M rows), scoring 12.6M rows | a few minutes each |
| Causal forest, 200k rows, 100 trees | about 1 minute (measured on Criteo: 18 to 20 s, D021) |
| Simulator, 27k users x 28 days x 4 slots (3M decisions), one policy and seed | about 5 seconds, vectorized over users, scores precomputed (measured: under a second, D024) |
| Sensitivity sweep, 263 settings x 3 seeds on 5k users, policy models refitted per setting | about 2 to 3 hours on 4 processes (D031; measured: 6,097 s, D034) |
| Copy generation (one-off, not in CI) | 6 requests; under 1 USD at 2026-11-08 prices (D011) |

Estimates are measured in M1 to M4 and recorded in DECISIONS.md.

## 11. Testing strategy

- Metrics: recall@K and NDCG against hand-computed examples; Qini and AUUC against scikit-uplift on Hillstrom
- Ranking cascade: with N equal to the pool size, the cascade returns the fine ranker's full-pool top-1
- Uplift: on synthetic data with a known effect, learners recover the effect within tolerance; with no effect, Qini is near zero
- OPE: on synthetic bandit data with known policy values, IPS and DR are unbiased within the bootstrap interval; when the target equals the logging policy, IPS returns the on-policy mean
- Simulator: invariants (no sends means no fatigue; with zero fatigue and opt-out, more sends never lower clicks; with zero copy spread, template choice does not change the outcome); fixed seeds give identical outputs
- Budget allocation: never exceeds the cap; matches brute force on small instances where marginal gains are non-increasing, and the test documents a case where they are not
- Copy: every committed template passes the filters and fills from fixture data; the generation script is tested against a recorded response, never the live API

## 12. Known limitations

- No public dataset links push sends to in-app consumption. The end-to-end push story is a simulation built on video feed data
- The uplift trials are e-mail and ads, not push; they validate methods, not the simulator's gate
- Long-term effects of volume cannot be measured here, only assumed and swept
- A single budget multiplier is optimal only when each user's marginal gain falls with every extra send; fatigue can break this
- In-app activity hours stand in for push open times
- Copy effects are simulated and not derived from the text (Section 7)
- Social relations are not covered: the public data has no social graph

## 13. Deliverables

- Demo notebook telling the story in Section 3
- Static HTML report with the evidence matrix
- `docs/METHODS.md` with every formula, assumption and parameter range
- English README: one-line summary, report screenshot, core story, evidence matrix
