# CLAUDE.md

This repository is an offline lab for push notification recommendation. Its focus is honest evaluation without live traffic: every result says whether it rests on randomized real data, a validated off-policy estimate or simulator assumptions.

## Read before starting

1. `docs/SPEC.md`
2. `docs/TASKS.md`: start from the first unchecked milestone
3. `docs/DECISIONS.md`
4. `docs/METHODS.md`: formulas, simulator assumptions and parameter ranges; implementations must match it

## Directory structure

```
src/push_lab/
  data/      Downloads, checksums, Parquet, quality report, synthetic fixtures
  recall/    Candidate pool, two-tower model, ANN index
  rank/      Coarse and fine ranking
  uplift/    Meta-learners, causal forest, Qini and AUUC
  sim/       Push simulator
  policy/    Send gate, budget allocation, send time
  creative/  Copy templates, generation script, filters, template bandit
  ope/       Off-policy estimators
  report/    Evidence matrix, static HTML report
data/creative/  Committed generated copy and its manifest
results/        Committed aggregated results (JSON); per-run output in results/runs/ is ignored
tests/
notebooks/      Demo notebook
```

## Working practices

- Write a plan before starting each milestone
- One commit per task, using Conventional Commits in English
- Check off tasks when done; record new decisions in DECISIONS.md
- Before implementing a method or a simulator component, write its formula, assumptions and source in METHODS.md
- Everything committed to GitHub (code, comments, docs, commit messages) is in English and concise: say what changed and why.

## Common commands

```bash
uv sync
uv run pytest
uv run ruff check . && uv run mypy
uv run python -m push_lab.data.download          # Fetch datasets into data/raw
uv run python -m push_lab.data.prepare           # Convert them to Parquet in data/processed
uv run python -m push_lab.data.quality           # Write results/data_quality.json
uv run python -m push_lab.rank.evaluate          # Write results/ranking.json
uv run python -m push_lab.creative.generate --model <model-id>  # One-off; needs ANTHROPIC_API_KEY
uv run python -m push_lab.report.build           # Rebuild results and the HTML report
```

## Hard rules

- Every reported number carries its evidence tier (A, B or C) and a confidence interval
- Ranking is evaluated on the random-exposure evaluation half only, never on outcomes the simulator was fitted on; uplift methods only on randomized trials
- KuaiRand rows are filtered to `tab == 1`, where `is_click` has one meaning
- Simulator results are never shown without the sensitivity sweep
- Simulator comparisons are pre-registered in DECISIONS.md and pushed before they are run; results are reported as they come out
- No temporal leakage: features for a decision at time t use data before t
- Fix the random seed for every random process
- Do not commit raw data or row-level samples; synthetic fixtures, generated copy and aggregated results only
- Tests and CI never call the Claude API; the API key is read from the environment and never written to a file
- Everything must run on 4 CPU cores without a GPU; CI uses synthetic fixtures only
- Functions must have type annotations and pass mypy strict mode
