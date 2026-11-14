# push-lab

[![CI](https://github.com/sHyanorn/push-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sHyanorn/push-lab/actions/workflows/ci.yml?query=branch%3Amain)

An offline lab for push notification recommendation: recall, ranking, uplift gating, send volume, timing and copy selection built on public data, with every result labeled by how far it can be trusted without live traffic. Results are tiered: A for randomized real data, B for an off-policy estimate from an estimator first validated against a known answer, C for simulator runs with a sensitivity sweep. No estimator passed that validation on the full Men's campaign of the Open Bandit Dataset, the only real logs here with two policies and known propensities, so the project has no Tier B result. Send volume, timing and copy are tested only in the simulator, which cannot give real-world effect sizes.

[![Report](docs/report.png)](https://shyanorn.github.io/push-lab/)

## Results

The numbers below and the [report](https://shyanorn.github.io/push-lab/) are generated from the committed results in [`results/`](results). Brackets give 95% intervals.

<!-- results:findings -->
| # | Finding | Evidence |
|---|---|---|
| 1 | In the simulator, per-day doubly robust (DR) estimates favor more pushes when sending annoys users: their bias for half the volume changes sign at an annoyance of 0.0058 (0.0014 to 0.0106). | Tier C: the registered reference run and a sweep of 24 settings on 5 one-at-a-time grids, of which the annoyance grid runs from 0 to 0.1 with the other parameters at the reference |
| 2 | In the simulator, on the one-at-a-time grid, whether a daily push pays turns on two parameters only, annoyance and the base rate; at the reference setting it changes the daily active share by −2.3% (−2.4% to −2.2%) against no pushes. | Tier C: 64 Latin hypercube settings, the same settings without annoyance, and a one-at-a-time grid |
| 3 | On randomized real data, the fused fine ranker ranks random exposures worse than the two-tower score it refines (C2), the T-learner targets Criteo visits worse than a plain response model (U4), and on the full Men's campaign of the Open Bandit Dataset IPS estimates the BTS click rate at 0.891 (0.815 to 0.966) of its on-policy value and DR at 0.897 (0.824 to 0.975), so no estimator passed. | Tier A: KuaiRand random exposures, Hillstrom, Criteo and the Open Bandit Dataset |

The [report](https://shyanorn.github.io/push-lab/) gives each finding in full: its sweep, its post hoc analyses, and what it shows and does not show.
<!-- /results:findings -->

<!-- results:evidence -->
| Tier | Evidence | Can support | Cannot support | Claims |
|---|---|---|---|---|
| A | Randomized real data: KuaiRand random exposures, Hillstrom, Criteo, Open Bandit Dataset, Duolingo notification logs | A component ranks or targets better in that domain; on the Open Bandit Dataset, how close an off-policy estimator comes to a known policy value; on the Duolingo logs, the value of one notification under a template policy given the logged histories | Push outside the Duolingo logs; volume fatigue or long-term effects | 9 |
| B | Off-policy estimate on real logs from an estimator first validated against a known answer | The value of a new policy within the support of the logging policy | Policies outside that support; long horizons | 0 |
| C | Simulator runs with a sensitivity sweep | Policy A beats B across a stated share of assumptions, or flips at a stated threshold | Real-world effect sizes | 41 |

The project has no Tier B result: no estimator passed the check on the full Men's campaign of OBD, the only real logs with two policies and known propensities here.
<!-- /results:evidence -->

### The registered story

[SPEC Section 3](docs/SPEC.md) states the story as a hypothesis registered before the simulations ran.

<!-- results:story -->
| Item | Outcome |
|---|---|
| Sending the top item to every user every day wins on clicks in week one | Holds in the simulator, as a registered secondary result without a test: by point estimate in 64 of 64 main settings |
| It loses users to fatigue and opt-outs by week four | Partly, in the simulator: by point estimate in 45 of 64 main settings and in none of the 64 without annoyance |
| An uplift gate under a volume budget, at each user's active slot, with a bandit choosing copy, sends fewer pushes and keeps more users active | Not as stated, in the simulator: at half volume the gate beats a daily push in 43 of 64 main settings and loses in 19 |
| The conclusion holds across most plausible settings, and the report shows where it flips | Partly, in the simulator: flips are located for F-N and H3; every other crossing of H1 to H4 is unresolved |
| Every number carries an evidence tier | Holds for the results in the evidence matrix, each Tier A or C; counts of the generated copy and its rating are descriptive, and registered verdicts later overturned are kept as history |

The [report](https://shyanorn.github.io/push-lab/) gives each outcome with its numbers.
<!-- /results:story -->

## How it works

Ranking is evaluated on KuaiRand random exposures, and uplift models on Hillstrom and Criteo. A simulator built from KuaiRand tables runs delivery policies that differ in volume, gating, timing and copy, each over a sensitivity sweep. Copy templates are generated once, rated, and chosen by a bandit in the simulator. Off-policy estimators are checked against a known policy value on the Open Bandit Dataset, and in the simulator. A report step reads `results/*.json` and writes the report page and the generated blocks of this README ([D049](docs/DECISIONS.md)).

Each result in the evidence matrix carries a tier, simulator results are shown only with their sensitivity sweep, and post hoc analyses are labeled. Pre-registrations and results are in [docs/DECISIONS.md](docs/DECISIONS.md); formulas, assumptions and parameter ranges in [docs/METHODS.md](docs/METHODS.md).

## Run it

Needs Python 3.14 and [uv](https://docs.astral.sh/uv/); everything runs on 4 CPU cores without a GPU.

```bash
uv sync
uv run pytest                                   # synthetic fixtures and the committed results; no downloads
uv run python -m push_lab.data.download         # datasets into data/raw, checksums verified
uv run python -m push_lab.data.prepare          # Parquet into data/processed
uv run python -m push_lab.report.build          # report in site/ and the README numbers
```

[docs/PIPELINE.md](docs/PIPELINE.md) lists every step of the pipeline with what it reads and writes. The [demo notebook](notebooks/demo.ipynb) recomputes the main summaries from the committed results.

## Limits

- Tier C results come from a simulator. They can show that one policy beats another across a stated share of assumptions; they cannot give real-world effect sizes. Settings are points drawn over the parameter box, not probabilities of the real world.
- Finding 1 does not show the size or sign of annoyance in a real app, or how parameters interact: the sweep moves one parameter at a time.
- The decomposed gate G2 assumes that an open and an organic visit are independent given the features. This holds in the simulator by construction, so part of any G2 win comes from the simulator's structure.
- Per-setting verdicts of H1 and H3 rest on one model fit and understate fit noise. Crossing intervals cover user sampling with the grid's seeds fixed, not the interpolation or model-fit randomness (D034).
- The simulator does not read the copy, so nothing here says which text works.
- The data of finding 3 are video feed, e-mail, ads and e-commerce recommendations, so finding 3 says nothing about push. The OBD ratio measures the estimators, not the value of any push policy.
- On the full Men's campaign of OBD, the target is BTS with its fixed production prior, which the deployed BTS was not. The registered intervals resample impressions, which are not independent in these logs. Post hoc, with ten-minute blocks as resampling units, the registered rule would pass both IPS and DR, and the IPS pass is marginal and depends on the random stream; the registered verdict stands, and no estimator is treated as validated (D063). In a separate registered study, with the same kind of block intervals, the IPS intervals on the All and Women's campaigns contain their own on-policy values and are labeled "no shortfall shown" (D065). These data do not show where the gap below the on-policy value comes from.
- The results reproduce exactly on the machine that produced them, apart from a rare last-bit deviation in the two-tower fit and the few numbers that follow from it ([D062](docs/DECISIONS.md)). On other machines the fitted models and what follows from them differ ([D053](docs/DECISIONS.md)).

## Layout

```
src/push_lab/   data, recall, rank, uplift, sim, policy, creative, ope, report
results/        aggregated results (JSON)
data/creative/  generated copy and its manifest
notebooks/      demo notebook
docs/           specification, methods, decisions, tasks, pipeline
```

Code: [MIT](LICENSE). The datasets are not redistributed here and keep their own licenses; see [DATA_SOURCES.md](DATA_SOURCES.md).
