# push-lab

An offline lab for push notification recommendation: recall, ranking, uplift gating, send volume, timing and copy selection built on public data, with every result labeled by how far it can be trusted without live traffic.

[![Report](docs/report.png)](https://shyanorn.github.io/push-lab/)

The [report](https://shyanorn.github.io/push-lab/) and the numbers below are generated from the committed results in [`results/`](results); the [demo notebook](notebooks/demo.ipynb) recomputes the main summaries from them. Brackets give 95% intervals.

## Evidence at a glance

<!-- results:evidence -->
| Tier | Evidence | Can support | Cannot support | Claims |
|---|---|---|---|---|
| A | Randomized real data: KuaiRand random exposures, Hillstrom, Criteo | A component ranks or targets better in that domain | Anything about push, fatigue or long-term effects | 9 |
| B | Off-policy estimate on real logs from an estimator first validated against a known answer | The value of a new policy within the support of the logging policy | Policies outside that support; long horizons | 0 |
| C | Simulator runs with a sensitivity sweep | Policy A beats B across a stated share of assumptions, or flips at a stated threshold | Real-world effect sizes | 40 |

The project has no Tier B result: no estimator passed the check on the full Men's campaign of OBD, the only real logs with two policies and known propensities here.
<!-- /results:evidence -->

Simulator results are always shown with their sensitivity sweep, and post hoc analyses are labeled. Pre-registrations and results are in [docs/DECISIONS.md](docs/DECISIONS.md); formulas, assumptions and parameter ranges in [docs/METHODS.md](docs/METHODS.md).

## Main findings

<!-- results:findings -->
### 1. Per-day off-policy estimates favor more pushes when sending annoys users

*Tier C. In this simulator, along annoyance elasticities from 0 to 0.1 with the other parameters at the reference, over 24 one-at-a-time settings.*

At the reference setting (annoyance 0.05), per-day DR on the exploration logs has a bias in active-day share of −0.0181 (−0.0189 to −0.0173) for no pushes, −0.0091 (−0.0096 to −0.0086) for half the volume and +0.0182 (0.0166 to 0.0199) for two pushes a day. The bias for half the volume changes sign at an annoyance of 0.0058 (0.0014 to 0.0106) and for no pushes at 0.0074 (0.0037 to 0.0113); above that the estimator underestimates lower volumes and overestimates two pushes a day. For the three targets that send about one push a day, as the logs do, DR is validated at 24 of 24 settings each. *Post hoc:* At the reference the simulator's truth puts no pushes ahead of one push a day by +0.0129 (0.0125 to 0.0133) in active-day share, while DR estimates −0.0044 (−0.0065 to −0.0023), the opposite sign. DR reverses this ordering at 23 of 24 settings and that of two a day against one at 22 of 24; its estimate of no pushes minus one a day stays between −0.0049 and −0.0036 along the annoyance grid while the truth moves from −0.0065 to +0.0315.

- **What it shows:** Per-day importance weights cannot see that recent volume lowers later organic visits, so in this simulator they favor higher volumes above the crossings, by an amount that grows with annoyance.
- **What it does not show:** The size or sign of annoyance in a real app. The sweep moves one parameter at a time; fatigue, opt-out, return and novelty changed no direction.

### 2. Whether a daily push pays turns on annoyance and the base rate

*Tier C. 64 Latin hypercube settings (main sweep), the same settings without annoyance, and a one-at-a-time grid.*

At the reference setting one push a day changes the daily active share by −2.3% (−2.4% to −2.2%) against no pushes. On the one-at-a-time grid it changes sign along two parameters only: it gains below an annoyance elasticity of 0.016 (0.015 to 0.017) and above a base-rate factor of 0.18 (0.17 to 0.19). Across the 64 main settings, its lift is below the band of published program effects (0.07% to 0.5%) in 37, inside it in 2, above it in 21 and straddles it in 4. At equal volume the decomposed gate G2 beats the response threshold S in most settings, while the T-learner gate G is worse on average (H1). Sending at each user's best slot shows no detectable difference from one fixed slot (H4), and is slightly worse on average without annoyance.

- **What it shows:** Within the swept assumptions, the sign of a daily push's effect is set by annoyance and the base rate. At equal volume, a gate that scores open probability and baseline visits separately beats a response threshold, and a T-learner uplift gate does not.
- **What it does not show:** Real effect sizes. Part of the G2 win comes from the simulator treating opens and organic visits as independent given the features, which G2 assumes.

### 3. Checks on randomized real data, two of them against expectation

*Tier A. KuaiRand random exposures, Hillstrom, Criteo and the Open Bandit Dataset.*

The fused fine ranker ranks random exposures worse than the two-tower score it refines (C2). On Criteo the T-learner targets visits worse than a plain response model (U4). On the full Men's campaign of the Open Bandit Dataset, IPS estimates the BTS click rate at 0.891 (0.810 to 0.969) of its on-policy value and DR at 0.896 (0.816 to 0.968), so no estimator passed and the project has no Tier B result. The target is BTS with its fixed production prior, which the deployed BTS was not. Post hoc, re-weighting to the BTS logs' own frequencies moves IPS to 0.917 (0.844 to 0.983).

- **What it shows:** A ranker fitted on logged impressions can rank random candidates worse than its retrieval score; where uplift grows with baseline response, a response threshold is a strong baseline; standard estimators fell short of a known policy value on real logs.
- **What it does not show:** Anything about push: the trials are video feed, e-mail and ads. The OBD ratio measures the estimators, not the value of any push policy.
<!-- /results:findings -->

## Core story

[SPEC Section 3](docs/SPEC.md) states the story as a hypothesis registered before the simulations ran. How each part came out:

<!-- results:story -->
| Item | Outcome |
|---|---|
| Sending the top item to every user every day wins on clicks in week one | Not tested: no comparison of week-one opens was registered; the policy comparisons use active days (D034). |
| It loses users to fatigue and opt-outs by week four | Partly (F-N, D034). At the reference a daily push changes the daily active share by −2.3% (−2.4% to −2.2%). On the one-at-a-time grid the sign turns on annoyance (a gain below 0.016 (0.015 to 0.017)) and the base rate (a gain above 0.18 (0.17 to 0.19)), never on fatigue or opt-out. |
| An uplift gate under a volume budget, at each user's active slot, with a bandit choosing copy, sends fewer pushes and keeps more users active | Not as stated (H1 to H4, B1, B2; D034, D040). The T-learner gate loses to a response threshold at equal volume; the decomposed gate G2 wins. At half volume the gate beats a daily push in 43 of 64 main settings and in none without annoyance. The best slot adds nothing detectable. The bandit beats uniform templates on average; the recency penalty has no detectable effect. |
| The conclusion holds across most plausible settings, and the report shows where it flips | Partly (F-N, H3; D034). Flips are located for F-N along annoyance at 0.016 (0.015 to 0.017) and the base rate at 0.18 (0.17 to 0.19), and for H3 along annoyance at 0.013 (0.012 to 0.015); every other crossing of H1 to H4 is unresolved. |
| Every number carries an evidence tier | Holds, with no Tier B result (D046). |
<!-- /results:story -->

## Reproduce

Needs Python 3.14 and [uv](https://docs.astral.sh/uv/); everything runs on 4 CPU cores without a GPU.

```bash
uv sync
uv run pytest                                   # synthetic fixtures only
uv run python -m push_lab.data.download         # datasets into data/raw, checksums verified
uv run python -m push_lab.data.prepare          # Parquet into data/processed
uv run python -m push_lab.report.build          # report in site/ and the README numbers
```

Each step of the pipeline has its own module (`rank.evaluate`, `uplift.evaluate`, `sim.build`, `policy.sweep`, `policy.template_sweep`, `ope.obd`, `ope.simulator`, `ope.carryover`); measured run times are recorded in the decisions. The `reproduce` workflow runs the pipeline on synthetic data, or reruns part of it on a small real sample and compares the numbers with the committed results.

## Layout

```
src/push_lab/   data, recall, rank, uplift, sim, policy, creative, ope, report
results/        aggregated results (JSON)
data/creative/  generated copy and its manifest
notebooks/      demo notebook
docs/           specification, methods, decisions, tasks
```

## License

Code: [MIT](LICENSE). The datasets are not redistributed here and keep their own licenses; see [DATA_SOURCES.md](DATA_SOURCES.md).
