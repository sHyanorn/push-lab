# Tasks

Hour estimates are for planning; the required core is about 85 hours. If behind, cut in this order (about 10 hours in total):

1. Holm correction and flip thresholds become a plain table with intervals
2. Criteo (keep Hillstrom)
3. Simulator OPE validation for far-from-logging policies (keep near)
4. X-learner
5. Trending recall source
6. Greedy baseline for the template bandit (keep uniform)

## M0 Setup (3 h)
- [x] Initialize a Python 3.14 project with uv; configure ruff, mypy strict mode and pytest
- [x] Package structure `src/push_lab/{data,recall,rank,uplift,sim,policy,creative,ope,report}`
- [x] CI on Linux and Windows: lint, type checking, tests on synthetic fixtures
- [x] MIT license, README stub, `DATA_SOURCES.md` with attributions

## M1 Data (8 h)
- [x] KuaiRand-Pure download with checksum, conversion to Parquet, `tab == 1` filter
- [x] Hillstrom download; Criteo Uplift v2.1 from Hugging Face with a seeded 10% training sample
- [x] Open Bandit Dataset small sample download
- [x] Synthetic fixtures with the same schemas for CI
- [x] Data quality report: row counts, missing values, label rates by tab, treatment balance
- [x] Measure load times and memory; record in DECISIONS.md

## M2 Recall and ranking (15 h)
- [x] METHODS.md: split rules, two-tower loss with log-Q correction, cascade, metrics, ranking targets
- [x] Splits: standard logs for training, random exposures split by user into evaluation and simulator halves
- [x] Popularity baseline and trending source
- [x] Two-tower model with in-batch negatives; faiss HNSW index checked against exact search
- [x] Coarse ranking: two-tower score keeps the top N
- [x] LightGBM fine ranker with click and long-view targets, weighted fusion
- [x] Truncation study over N, against full-pool fine ranking
- [x] Evaluation on the random-exposure half with bootstrap intervals; record training times

## M3 Uplift methods (9 h)
- [x] METHODS.md: meta-learners, causal forest, Qini and AUUC variants, Hillstrom contrast
- [x] Qini and AUUC implementation, cross-checked against scikit-uplift on Hillstrom
- [x] S-, T- and X-learners on LightGBM; EconML causal forest
- [x] Synthetic-effect tests: known effect recovered, zero effect gives Qini near zero
- [x] Hillstrom and Criteo results with bootstrap intervals (Criteo evaluated on all rows outside the training sample)
- [x] Budget-constrained targeting: uplift against response ranking and random

## M4 Push simulator (12 h)
- [x] METHODS.md: simulator equations, every parameter with its source and range
- [ ] Response model fitted on the simulator half of the random exposures, with a base-rate factor
- [ ] Hourly activity profiles from organic logs, mapped to send slots
- [ ] Fatigue, opt-out, return and copy-effect models
- [ ] Vectorized environment: users x days x slots, seeded, scores precomputed per user and day
- [ ] Epsilon-greedy exploration policy over send or skip, slot and template, with logged propensities
- [ ] Invariant tests (SPEC Section 11)

## M5 Delivery policies (14 h)
- [ ] Pre-register sweep ranges, policy pairs and decision rules in DECISIONS.md and push them before running any comparison
- [ ] Baselines: fixed daily push at a fixed slot, score threshold
- [ ] Uplift gate fitted on exploration logs (Tier C)
- [ ] Global volume budget: marginal-gain allocation with one Lagrange multiplier, tested against brute force
- [ ] Send-time policy from earlier-period profiles
- [ ] Profile misspecification: swept shift and noise
- [ ] Sensitivity sweep (Latin hypercube over base rate, fatigue, opt-out, return, copy spread and misspecification, including zero)
- [ ] Policy comparison with Holm correction and flip thresholds

## M6 Copy and templates (10 h)
- [ ] METHODS.md: template families, placeholders, filters, recovering bandit
- [ ] Generation script: Claude API with structured output, manifest of model, effort, prompt, request ids and usage; tested against a recorded response
- [ ] Generate once and commit the outputs and manifest
- [ ] Filters: schema, length, placeholders, near-duplicates, banned claims
- [ ] Recovering bandit against uniform and greedy without the penalty, in the simulator with the copy-spread sweep

## M7 Off-policy evaluation (8 h)
- [ ] METHODS.md: DM, IPS, SNIPS, DR
- [ ] Estimators with effective sample size and largest weight, tested on synthetic bandit data
- [ ] Validation on Open Bandit Dataset (Men's campaign): random logs to BTS value, RMSE against the on-policy mean over bootstrap
- [ ] Validation in the simulator: estimates from exploration logs against simulator truth, for policies near and far from the logging policy
- [ ] Evidence matrix: each claim with its tier

## M8 Presentation (6 h)
- [ ] Demo notebook tells the core story
- [ ] Static HTML report generated from results JSON
- [ ] `reproduce` workflow (manual dispatch) runs the full pipeline on a subsample
- [ ] English README with report screenshot
- [ ] Tidy up DECISIONS.md

## Optional
- [ ] Duolingo: sample one training part locally, check propensities, replicate the recovering bandit off-policy
- [ ] LLM judge score as a bandit prior, with its correlation to the simulated copy effect swept from zero
- [ ] DR with optimistic shrinkage; DR-learner; ALS baseline; recent-favorite-author source
- [ ] DeepFM or DCN with FuxiCTR against LightGBM
- [ ] KuaiRand-1K for ANN recall and coarse-ranking latency at scale
- [ ] Full Open Bandit Dataset (downloaded once locally) for a 300k-row estimator comparison
- [ ] causalml cross-check in a separate Python 3.12 environment
