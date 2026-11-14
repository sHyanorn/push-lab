# Pipeline

The README has the short way to run the project. This is every step, in order.

The pipeline in order; each step is `uv run python -m push_lab.<module>`:

| # | Module | Reads | Writes |
|---|---|---|---|
| 1 | `data.download` | the sources in [DATA_SOURCES.md](../DATA_SOURCES.md) | `data/raw/` |
| 2 | `data.prepare` | `data/raw/` | `data/processed/` |
| 3 | `data.quality` | `data/processed/` | `results/data_quality.json` |
| 4 | `rank.evaluate` | KuaiRand tables | `results/ranking.json` |
| 5 | `uplift.evaluate` | Hillstrom and Criteo tables | `results/uplift.json` |
| 6 | `sim.build` | KuaiRand tables | `data/processed/sim/inputs.npz`, `results/simulator.json` |
| 7 | `policy.design` | constants in the code | `results/policy_design.json` |
| 8 | `policy.sweep` | simulator inputs, policy design | `results/runs/policy/`, then runs step 9 |
| 9 | `policy.compare` | `results/runs/policy/` | `results/policy.json` |
| 10 | `creative.generate` (one-off, needs an API key), then `creative.rating` | family definitions in the code | `data/creative/`, `results/copy.json` |
| 11 | `policy.template_design` | policy design | `results/template_design.json` |
| 12 | `policy.template_sweep` | simulator inputs, template design, `data/creative/` | `results/runs/templates/`, then runs step 13 |
| 13 | `policy.template_compare` | `results/runs/templates/` | `results/templates.json` |
| 14 | `ope.obd`, then `ope.obd_blocks` (post hoc, full campaign only) | Open Bandit tables | `results/ope_obd.json`, `results/ope_obd_blocks.json` |
| 15 | `ope.simulator` | simulator inputs | `results/runs/ope/`, `results/ope_simulator_amended.json` |
| 16 | `ope.carryover` | simulator inputs | `results/runs/ope_carryover/`, `results/ope_carryover.json` |
| 17 | `report.evidence` | `results/*.json` | `results/evidence.json` |
| 18 | `report.build` | `results/*.json` | `site/index.html`, the generated blocks of the README |

Steps 4 and 5 are independent of each other, as are 14 to 16. `results/runs/` is not committed, and `results/ope_simulator.json` is the first run of step 15, kept as history ([D044](DECISIONS.md)). Measured run times are recorded in the decisions. The manual `reproduce` workflow runs the steps that work on synthetic fixtures, or reruns part of the pipeline on a small real sample and compares the numbers with the committed results ([D052](DECISIONS.md)). The results reproduce exactly on the machine that produced them, apart from a rare last-bit deviation in the two-tower fit and the few numbers that follow from it ([D062](DECISIONS.md)); on other machines the fitted models and what follows from them differ ([D053](DECISIONS.md)).
