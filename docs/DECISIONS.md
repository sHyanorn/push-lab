# Technical Decisions

## D001 Python only, static report
- Date: 2026-11-08
- Rationale: the work is modeling and evaluation; a notebook and a static HTML report show it without a server

## D002 Offline only, with evidence tiers
- Date: 2026-11-08
- Rationale: there is no live traffic. Each result is labeled Tier A (randomized real data), B (validated off-policy estimate) or C (simulator with sensitivity sweep), as in SPEC Section 5. Published notification systems report small online effects (about 0.1% to 0.5% DAU in recent papers) and offline estimates that agree least on the longest-horizon metric (Prabhakar et al. 2022: offline-online r² 0.94 for volume, 0.68 for WAU), so the project does not report effect sizes as if they were real

## D003 Raw data is not committed
- Date: 2026-11-08
- Rationale: Criteo (CC BY-NC-SA 4.0) and Duolingo (CC BY-NC 4.0) are non-commercial; Hillstrom states no license. KuaiRand (CC BY-SA 4.0) and the Open Bandit Dataset (CC BY 4.0) allow redistribution with attribution, but one rule for all keeps the repository light. Only download scripts, checksums, synthetic fixtures, generated copy and aggregated results are committed; attributions are in `DATA_SOURCES.md`

## D004 Datasets
- Date: 2026-11-08
- KuaiRand-Pure for recall, ranking and the simulator: 47 MB, downloads non-interactively from Zenodo, and 1,186,059 of its rows are random exposures, enough to split by user between ranker evaluation and the simulator's response model
- Only `tab == 1`: `is_click` means a click in the two-column UI and a valid play in the single-column UI, and 99% of random exposures are tab 1
- Hillstrom for uplift in CI (4 MB). Contrast: women's e-mail against no e-mail
- Criteo Uplift v2.1 from Hugging Face; scikit-uplift's `fetch_criteo` mirror returned 404 on 2026-11-08, and v1 has a known leak. Train on a 10% sample, evaluate on the remaining rows: Diemert et al. (2021) found methods indistinguishable below about 1M test points. The sample is a boolean `train` column in the one Parquet file, drawn with seed 0
- Open Bandit Dataset small sample, Men's campaign, for OPE validation: two logging policies with known propensities
- Rejected: MIND (research-only license), X5 (no license, download may need login), KuaiRec (KuaiRand already provides random exposures)
- Limitation: no dataset links push sends with consumption (SPEC Section 12)

## D005 Python 3.14 without causalml
- Date: 2026-11-08
- Rationale: matches the other projects. causalml 0.18 ships wheels for CPython 3.11 and 3.12 only. torch 2.15, faiss-cpu 1.15.1, EconML 0.17 and its compiled dependencies, implicit, scipy, pyarrow and pandas 3.1 all have 3.14 wheels; LightGBM is a py3 wheel. Meta-learners are short to write on LightGBM, and EconML provides the causal forest
- scikit-learn is pinned below 1.10: scikit-uplift 0.5.1 (dev-only, for metric cross-checks) calls `stable_cumsum`, which 1.10 removes, and EconML already requires it. torch-rechub pins `pandas<3`, so FuxiCTR is the option for a deep CTR model

## D006 Own OPE estimators
- Date: 2026-11-08
- Rationale: Open Bandit Pipeline (obp 0.5.7, released 2023-04) has had no commit since 2022-11 and declares support up to Python 3.10. The estimators are about a hundred lines; obp is a reference and the results in Saito et al. (2021) a check

## D007 faiss-cpu for ANN
- Date: 2026-11-08
- Rationale: wheels for CPython 3.10 to 3.14, MIT. hnswlib 0.8.0 on PyPI is source-only and needs a compiler in CI. With 7.5k items exact search is cheap, so HNSW recall is measured against it; the index matters at KuaiRand-1K scale (optional)

## D008 LightGBM for fine ranking, two-tower score for coarse ranking
- Date: 2026-11-08
- Rationale: LightGBM is strong on tabular features, fast on CPU and MIT. The two-tower score is already computed for recall, so the coarse stage costs nothing extra. With 7.5k items the fine ranker can also score the full pool, which measures what truncation loses; at KuaiRand-1K scale it could not. A deep CTR model is optional and must beat LightGBM on the random-exposure half

## D009 Volume control is separate from ranking
- Date: 2026-11-08
- Rationale: following Zhao et al. (2018), the ranker picks what to send and the policy decides whether to send. Changing one does not confound evaluation of the other. The uplift gate threshold and the budget's Lagrange multiplier are the same dual variable; a single multiplier is optimal only when marginal gains fall with each extra send, which fatigue can break, and the brute-force test checks this

## D010 Duolingo data is optional
- Date: 2026-11-08
- Rationale: it is the only public data from a real recurring notification system, but it is about 9.35 GB compressed, non-commercial, has no user id or template text, and whether its logging propensities allow off-policy replication is not yet verified

## D011 Copy generated once with the Claude API and committed
- Date: 2026-11-08
- Model: a current Claude model, chosen for quality since the cost difference is negligible at six requests; the exact model is recorded in the manifest. The script takes the model as a command-line argument and does not hard-code it. Standard Messages API with structured output; the Batch API can take up to 24 hours, which buys nothing for six requests
- Cost: under 1 USD at 2026-11-08 prices
- Reproducibility: current models do not accept sampling parameters, so a rerun gives different text. The script, prompts and manifest (the model the API reports, effort, request ids, usage, date) are committed with the outputs, and the committed outputs are what the project uses. No key is needed to run anything else, and CI never calls the API
- Source of text: template families filled from fields KuaiRand has (SPEC Section 7). KuaiRec captions do not join KuaiRand ids, MIND is research-only and Duolingo has no template text

## D012 Statistics conventions
- Date: 2026-11-08
- Bootstrap over users, 1,000 resamples, 95% percentile intervals; Holm correction across policy comparisons; every random process takes an explicit seed

## D013 Simulator uplift gate is learned inside the simulator
- Date: 2026-11-08
- Rationale: models trained on Hillstrom or Criteo cannot score KuaiRand users (different features, different treatments). The simulator runs an epsilon-greedy exploration policy over send or skip, slot and template; the gate is fitted on those logs, and the same logs give the OPE estimators overlap with the target policies. Tier A uplift results validate the methods; the gate's results are Tier C

## D014 Send-time policy does not read the simulator's profile
- Date: 2026-11-08
- Rationale: if one activity profile drove both the policy and the simulator, the policy would win by construction. The simulator uses profiles from 04-22 to 05-08, the policy estimates its own from 04-08 to 04-21, and the sweep adds a shift and noise on top

## D015 Data layout, sources and measured cost
- Date: 2026-11-08
- Layout: downloads in `data/raw/<dataset>/`, Parquet tables in `data/processed/<dataset>/`, both ignored by git; the quality report in `results/data_quality.json` is committed (D016)
- Checksums: KuaiRand-Pure md5 as published by Zenodo; Criteo sha256 is the Git LFS object id published by Hugging Face; Hillstrom and the four Open Bandit Dataset files have no published checksum, so their sha256 was computed on 2026-11-08, the OBD files from a commit of the Open Bandit Pipeline repository pinned in `sources.py`
- Hillstrom is fetched over plain HTTP: the host answers HTTPS with a self-signed default certificate. The pinned size and sha256 detect any change in transit or at the source, and the file is only read as CSV, so the risk is a failed download, not a silent one
- Zenodo ignores Range requests and sends the whole file again; the downloader detects this and restarts the file
- Criteo is converted in 16 MB CSV blocks: 64 MB blocks raised peak memory from 0.9 GB to 2.6 GB with no gain in speed. Blocks are buffered into row groups of about 1M rows, so a `train` filter can skip groups
- Measured on 2026-11-08 in a cloud container (4 vCPU, 15 GB RAM, Linux 6.18, Python 3.14.10), from empty directories; peak memory is the maximum resident set size from GNU time; download times depend on the network

| Step | Size | Time | Peak memory |
|---|---|---|---|
| Download KuaiRand-Pure | 47.4 MB | 36 s | 34 MB |
| Download Hillstrom | 4.0 MB | 0.5 s | 34 MB |
| Download Criteo | 311 MB | 12 s | 34 MB |
| Download OBD sample | 6.6 MB | 0.8 s | 36 MB |
| Prepare KuaiRand-Pure (extracted CSV 203 MB) | 42 MB Parquet | 2.0 s | 0.54 GB |
| Prepare Hillstrom | 0.3 MB Parquet | 0.5 s | 0.26 GB |
| Prepare Criteo | 252 MB Parquet, 13 row groups | 10 s | 0.93 GB |
| Prepare OBD sample | 0.5 MB Parquet | 0.6 s | 0.24 GB |
| Quality report (reads only the columns it checks) | | 0.6 s | 0.46 GB |
| Load KuaiRand random exposures, tab 1 (1,178,025 rows) | 170 MB in pandas | 0.1 s | 0.77 GB |
| Load Criteo, all rows | 1.35 GB in pandas | 0.7 s | 4.7 GB |
| Load Criteo, training sample (1,398,727 rows) | 134 MB in pandas | 0.5 s | 1.0 GB |

- Full Criteo scoring in M3 should read only the columns it needs or go in row batches; the whole table in pandas peaks at 4.7 GB

## D016 Data quality checks
- Date: 2026-11-08
- Errors (the command exits with status 1): a missing processed table; a Parquet schema that differs from the shared schema; a row count that differs from the published figure (KuaiRand-Pure logs 1,141,112, 295,497 and 1,186,059 rows, 27,285 users; Criteo 13,979,592 rows); `is_rand` not 1 on every random-exposure row or not 0 on every standard-log row; Criteo treatment share more than 0.005 from the published 0.85; Criteo training share more than 0.01 from 0.1; Hillstrom treatment share outside 0.45 to 0.55; an OBD propensity outside (0, 1], or a random-policy propensity that is not 1/34
- Reported without a threshold: missing shares per column (from Parquet statistics), KuaiRand click rate by tab, visit and conversion rates by arm, OBD click rates and propensity ranges
- The report goes to `results/data_quality.json` and is committed; it has no timestamp, so it changes only when the data or the checks do

## D017 CPU-only torch
- Date: 2026-11-08
- torch comes from the PyTorch CPU index (`[tool.uv.sources]` in `pyproject.toml`); the PyPI build pulls CUDA libraries on Linux, which a CPU-only project never uses. The lock resolves `2.15.0+cpu` on Linux and Windows and `2.15.0` on macOS, with no CUDA packages
- faiss-cpu 1.15.1 ships its own type information, so mypy checks it without an override

## D018 HNSW search breadth
- Date: 2026-11-08
- METHODS first set `efSearch = 128`. faiss HNSW keeps at most `efSearch` candidates while searching, so with K = 200 the result can only partly agree with exact search. Measured at cutoff 04-22 on the 13,500 evaluation users and 7,583 items:

| efSearch | Overlap@200 | Overlap@500 | Search time, 13,500 users, K = 200 |
|---|---|---|---|
| 128 | 0.869 | 0.784 | 0.3 s |
| 256 | 0.963 | 0.925 | 0.6 s |
| 512 | 0.994 | 0.985 | 0.9 s |
| 1024 | 1.000 | 0.999 | 2.1 s |

- `efSearch = 1024` meets the 0.99 requirement at both K. Ties do not explain the gap: 7,231 of the 7,583 item vectors are distinct and a tie-aware overlap gives the same numbers
- At this catalog size exact search is faster (0.2 s). HNSW stays in the pipeline because it is what scales to KuaiRand-1K (optional); the evaluation reports both


## D019 Model-internal truncation measures on a user sample
- Date: 2026-11-08
- Top-1 agreement, score regret and latency need the fine ranker's score for every catalog item. For all 13,500 evaluation users that is about 102M feature rows, beyond the memory budget. They are computed for 1,000 evaluation users drawn with `numpy.random.default_rng(0)`; survival needs no fine scores and covers every evaluation user
- METHODS Section 4 says "per evaluation user"; this narrows the model-internal measures only, which carry no significance claims

## D020 Ranking evaluation cost and first results
- Date: 2026-11-08
- Measured in the same container as D015: the whole evaluation (`python -m push_lab.rank.evaluate`) takes about 100 s with a peak of 3.3 GB. Two-tower fits take 22 s (cutoff 04-22, 442k clicked pairs) and 15 s (04-15); the fine ranker 3.6 s (both heads stop early at 70 rounds); fine scoring of the whole catalog for 1,000 users 25 s, or 24 ms per user
- Pre-registered results (METHODS Section 6), reported as they came out:
  - C1: two-tower Recall@200 exceeds popularity by 0.47 points (95% CI 0.24 to 0.70; 4.7% against 4.2%); rejected under Holm
  - C2: the fused fine ranker's click GAUC is 5.3 points *below* the two-tower score's (0.545 against 0.598; CI -5.8 to -4.9); rejected under Holm, in the direction opposite to the expected one
- On its own validation rows (standard logs 04-20 and 04-21) the fine ranker beats the two-tower score (0.567 against 0.515), so the reversal comes from the shift from the production recommender's impressions to random exposures, not from a broken model. Its largest gains are user-level features (user click rate, `onehot_feat3`, `onehot_feat8`), which are constant within a user and cannot rank items for that user; its item statistics come from impressions the recommender chose. The drift diagnostics show the shift: the median item impressions per day are 43 on the training rows and 1.8 on the random exposures
- This is the main lesson M2 hands to the simulator: a ranker fitted on logged impressions can rank random candidates worse than the retrieval score it was meant to refine. Hyperparameters stay as pre-registered; any change to the fine ranker is a new, separately reported variant
