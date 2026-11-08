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
- Criteo Uplift v2.1 from Hugging Face; scikit-uplift's `fetch_criteo` mirror returned 404 on 2026-11-08, and v1 has a known leak. Train on a 10% sample, evaluate on the remaining rows: Diemert et al. (2021) found methods indistinguishable below about 1M test points
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
