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
- At this catalog size exact search is faster (0.2 s), and every reported retrieval number uses it, so results do not depend on HNSW's parallel graph construction. HNSW stays in the pipeline because it is what scales to KuaiRand-1K (optional); its overlap with exact search is reported as a diagnostic


## D019 Model-internal truncation measures on a user sample
- Date: 2026-11-08
- Top-1 agreement, score regret and latency need the fine ranker's score for every catalog item. For all 13,500 evaluation users that is about 102M feature rows, beyond the memory budget. They are computed for 1,000 evaluation users drawn with `numpy.random.default_rng(0)`; survival needs no fine scores and covers every evaluation user
- METHODS Section 4 says "per evaluation user"; this narrows the model-internal measures only, which carry no significance claims

## D020 Ranking evaluation cost and first results
- Date: 2026-11-08
- Measured in the same container as D015: the whole evaluation (`python -m push_lab.rank.evaluate`) takes about 100 s with a peak of 3.3 GB when run alone; a concurrent 4-thread job oversubscribes the 4 cores and slowed one tower fit from 22 s to 15 minutes. Two-tower fits take 22 s (cutoff 04-22, 442k clicked pairs) and 15 s (04-15); the fine ranker 3.6 s (both heads stop early at 70 rounds); fine scoring of the whole catalog for 1,000 users 25 s
- Pre-registered results (METHODS Section 6), reported as they came out:
  - C1: two-tower Recall@200 exceeds popularity by 0.47 points (95% CI 0.24 to 0.70; 4.7% against 4.2%); rejected under Holm
  - C2: the fused fine ranker's click GAUC is 5.3 points *below* the two-tower score's (0.545 against 0.598; CI -5.8 to -4.9); rejected under Holm, in the direction opposite to the expected one
- Why C2 reversed. Checks ruled out a misalignment between training and evaluation (same columns, types, categories and feature order at both cutoffs; per-user AUC and NDCG match an independent implementation). The fine ranker beats the two-tower score on its own validation rows (0.567 against 0.515, standard logs, optimistic because the same rows drive early stopping), so the reversal comes with the move from the production recommender's impressions to random exposures. Scored alone on the random exposures, the smoothed rates the ranker relies on carry little within-user signal (item click rate 0.524, user-author 0.500, user-tag 0.518), while plain exposure counts carry more (item impressions per day 0.570, popularity rank 0.573) and the two-tower score most (0.598). The rates are learned on impressions with a 53% click rate against 17.7% on random exposures; the ranker gives them about ten times the gain of the counts. In refits made during review (not part of the committed results), dropping the user-level features (0.541) or the tower score (0.544) did not change the result, so neither is the cause
- The tower score is also on a different scale at the two cutoffs: on the same standard-log rows its median is -0.03 with the 04-15 tower and -0.40 with the 04-22 tower, so at evaluation it lies outside the range the ranker was trained on. Count features are rescaled per day; scores are not. A per-cutoff normalization is a candidate variant, not run
- Lesson for the simulator: a ranker fitted on logged impressions can rank random candidates worse than the retrieval score it was meant to refine. Hyperparameters stay as pre-registered; any change to the fine ranker is a new, separately reported variant

## D021 Uplift evaluation cost and first results
- Date: 2026-11-08
- Measured in the same container as D015, run alone: `python -m push_lab.uplift.evaluate` takes 564 s with a peak of 4.1 GB. On Criteo each LightGBM learner fits in 6 to 8 s on 1.4M rows and the causal forest in 18 to 20 s on 200k rows; one pass over the 12.6M test rows scores all ten models in 398 s; the cell bootstrap for both Criteo outcomes, U4 and the decile diagnostic takes 40 s. Hillstrom takes seconds
- The Hillstrom Qini coefficients and AUUC of every method and outcome equal scikit-uplift's to the last bit (largest absolute difference 0)
- Criteo bins on visits: tied scores leave 193 of 200 bins for the S-learner, 195 for the response model and 98 for the X-learner, whose 95th-percentile bin holds 46,188 control rows, about five times an even bin, so large groups of rows share one X-learner score. The smallest bin of any method holds 350 control rows and no bin lacks control rows; on conversion the X-learner has 76 bins and the smallest bin holds 869 control rows
- Pre-registered results (METHODS Section 10), Qini coefficient on visits, reported as they came out. A p-value of 0 means below the resolution 2/1,000:
  - U1: the T-learner beats random targeting on Hillstrom, 0.041 (95% CI 0.017 to 0.064, p = 0.002); rejected under Holm
  - U2: the T-learner does not differ detectably from the response model on Hillstrom, difference 0.003 (CI -0.019 to 0.023, p = 0.82); not rejected
  - U3: the T-learner beats random targeting on Criteo, 0.076 (CI 0.073 to 0.079, p < 0.002); rejected under Holm
  - U4: the T-learner is *below* the response model on Criteo, difference -0.008 (CI -0.011 to -0.005, p < 0.002); rejected under Holm, in the direction opposite to the expected one
- Secondary, untested (Criteo visits, Qini coefficient): S-learner 0.087 (CI 0.084 to 0.090), response model 0.084 (0.081 to 0.087), X-learner 0.079 (0.076 to 0.082), T-learner 0.076 (0.073 to 0.079), causal forest 0.069 (0.065 to 0.072, fitted on 200k rows)
- Reading U4, with the post hoc decile diagnostic (D022). By response-score decile, uplift falls steadily from 51.2 visits per 1,000 (CI 48.7 to 53.6) in the top decile to 0.1 (-0.1 to 0.2) in the bottom one, so on this trial predicted visits order uplift well. By T-learner decile the top decile is higher, 56.6 (54.4 to 58.7), but the bottom decile has 8.0 (6.6 to 9.5), more than any of deciles 3 to 9, and a control visit rate of 88.4 per 1,000 (87.2 to 89.7): the T-learner ranks last a group of frequent visitors who do respond to the ads. In the budget table this shows as a lead at 10% (5.7 visits per 1,000, CI 5.5 to 5.9, against 5.1, CI 4.9 to 5.4) and a deficit at 50% (9.1, CI 8.9 to 9.4, against 9.8, CI 9.5 to 10.1); these are separate intervals, not a paired test. The T-learner's control model sees only the 15% control share of the training rows (210k), so its difference of two models is noisier than one response model; this cause is consistent with the numbers but not tested
- Lesson for the delivery policies: where uplift grows with the baseline response, a response-score threshold is a strong baseline, and the uplift gate (M5) has to beat it, not only random sending

## D022 Changes to the uplift evaluation after review
- Date: 2026-11-08
- Criteo treatment and outcomes are now read from the same batches that are scored. Before, they came from a second filtered read whose row order matched the scoring pass only because pyarrow keeps it; the review confirmed all 12,580,865 rows matched, so the first results were not affected
- The causal forest now predicts with one thread per forest, and rows are split across threads instead. EconML summed tree outputs across threads in completion order, so the last bit of an effect changed between runs, which moved a few Criteo rows across bin edges and changed the bootstrap draws: the forest's Criteo intervals moved by up to 0.0007 between runs, with identical point estimates. Every other number was unchanged. Rounding the effects was tried first and failed in CI, since a value near a rounding boundary still flipped. Two consecutive full runs now give the same `results/uplift.json` apart from timings
- METHODS Section 9 now describes the bootstrap as implemented: multinomial draws over (score group, $t$, $y$) cells on both datasets, which equals row resampling, with Criteo bin edges fixed at their full-sample values. Section 10 describes the budget boundary as $U(bn)$ with linear interpolation instead of the top $\lceil bn \rceil$ rows
- Added so that every reported number has an interval: 95% bands for the Qini curves, and the budget share left out when the treat-everyone effect's interval contains 0
- Added after U4 was known, and labeled post hoc in the results: response rates and uplift per score decile of the T-learner and the response model on Criteo visits. It explains U4 and is not a pre-registered comparison

## D023 Simulator components added or changed against SPEC Section 6
- Date: 2026-11-08
- Annoyance: recent push volume lowers organic visits, with elasticity $\epsilon_v \in [0, 0.1]$ at one push a day (METHODS Section 11.4). Without it no volume can lower activity below the organic level, so losses could come only from opt-outs. Evidence goes both ways (LinkedIn WAU fell with more volume, Twitter DAU rose), so zero is in the range and M5 reports its conclusions at $\epsilon_v = 0$ as well as over the full sweep
- Active hours come from `time_ms` in UTC+8 instead of `hourmin`, which is a whole-hour bucket off by up to an hour; the time zone is an assumption checked by the overnight low in activity
- Send slots are fixed at 08:00, 12:00, 17:00 and 21:00, each with a three-hour response window; activity outside the windows does not enter the profiles

## D024 Simulator inputs: cost and checks
- Date: 2026-11-08
- Measured in the same container as D015, run alone: `python -m push_lab.sim.build` takes 136 s with a peak of 2.0 GB, of which the two towers take 45 s, the fine ranker 4 s, the response model 5 s and both candidate lists for 27,285 users 49 s. One 28-day run of the exploration policy over all users takes 0.3 s, so the M5 sweep is bounded by policy fitting, not by the simulator
- Response model, on the 20% of simulator-half users held out from fitting (118,405 rows, 2,700 users; they also drove early stopping, which chose 51 rounds; Tier A): AUC 0.694 (95% CI 0.682 to 0.705), GAUC 0.635 (0.628 to 0.643), log loss 0.424 (0.411 to 0.436); mean prediction 0.168 (0.162 to 0.174) against a click rate of 0.171 (0.163 to 0.179). By decile of prediction, the mean prediction lies inside the interval of the observed rate in 6 of 10 deciles; it is too high in the lowest two (0.053 against 0.036, CI 0.030 to 0.042; 0.070 against 0.063, CI 0.056 to 0.069) and too low in deciles 5 and 7 (0.137 against 0.149, CI 0.137 to 0.164; 0.192 against 0.208, CI 0.198 to 0.220)
- The largest shares of the response model's gain are user attributes and rates (`onehot_feat3` 26%, user click rate 21%, `onehot_feat8` 17%, user long-view rate 13%), then the first tag (9%) and the two-tower score (6%). The truth therefore leans partly on the tower score, which also orders the two-tower candidate list; M5 must state this when it pre-registers which list the policies use
- The response model's GAUC (0.635) is higher than the fine ranker's on the evaluation half (0.545, D020). This is consistent with learning from random exposures rather than the recommender's impressions, but the two numbers come from different users, models and training periods and are not a paired comparison
- Organic visit rate: median 0.59 (5th to 95th percentile 0.27 to 0.90)
- Time zone: the `date` field equals the calendar date of `time_ms` for 99.25% of rows in UTC+8, against 95.95% in UTC+7 and 95.63% in UTC+9, and activity is lowest at 04:00 UTC+8
- The best slot under the policy profile (04-09 to 04-21) equals the best slot under the simulator profile (04-22 to 05-08) for 38.8% of users (CI 38.3 to 39.4), against 29.4% if the two were independent with the same skew toward 21:00. This describes logs, not outcomes. 2,795 users have no policy-period rows in any window; their policy profile is the prior and their best slot 21:00
- Timing exponents: $\eta$ = 0.033, 0.164 and 0.399 for timing lifts of 0.02, 0.10 and 0.25
- Exploration: $K = 60$ templates until M6, send probability 0.775 at the best slot and 0.075 elsewhere, smallest propensity 0.00125
- Not reported: $p_0$ of the two candidate lists side by side. With every other factor shared, their ratio is the simulated comparison of the two item policies, which M5 pre-registers first

## D025 Simulator data choices after review
- Date: 2026-11-08
- Organic visit rates and the simulator profile use the timestamps of all random exposures, including the evaluation half's (597,219 of 1,178,025 rows), because random exposures mark activity for every user and leaving out half of them would make the two halves' users look unequally active. Only `user_id`, `date` and `time_ms` are read from those rows; no click or other outcome of the evaluation half enters the simulator, and the response model uses the simulator half only
- The opt-out evidence is Zhang et al. (2023) instead of Zhao et al. (2018) named in SPEC Section 6: the full text of Zhao et al. could not be obtained
- The policy profile starts on 04-09 (METHODS Section 1), where SPEC Section 6 and D014 say 04-08: the KuaiRand-Pure standard log has no tab-1 rows on 04-08, so nothing changes

## D026 Delivery policies push from the cascade list
- Date: 2026-11-08
- The cascade (fine-ranker top items) is the pipeline SPEC Section 4 describes, and the exploration logs of METHODS 11.7, which M7 reuses, follow it. The response model gives 6% of its gain to the two-tower score (D024), which orders the two-tower list, so in the simulator that list is favored by construction. Its comparison with the cascade list (L1) is secondary and carries this caveat; every delivery comparison holds the list fixed, so the choice moves the shared level of $p_0$, not which policy wins

## D027 What policies observe
- Date: 2026-11-08
- `Observation` gains the opens and daily visits of earlier days, so policies can use feedback; outcomes of the current day stay hidden. Two policy-side inputs are added to the cached simulator inputs: hourly standard-log counts per user from 04-09 to 04-21, from which the misspecified profile is rebuilt (METHODS 12.5), and a visit rate over the same period. Neither reads the simulator period
- The environment can also return each user's open probability at every slot and the organic visit probability of every day. Only the score diagnostics read them (METHODS 12.3); no policy does
- Templates are drawn from a stream keyed by (run seed, day, slot) that every policy shares, because `Draws` holds no template draws and template effects would otherwise be unpaired

## D028 Per-day budget in the primary analysis
- Date: 2026-11-08
- SPEC Section 4 named a single Lagrange multiplier for the global budget. A multiplier set from the exploration logs does not hold volume at the budget once the policy's own histories differ from the logs, and with a floor at zero only the uplift gate stops sending, so the response threshold and the gate would compare unequal volumes. The primary analysis therefore sends each day to the $\lfloor b n \rfloor$ eligible users with the highest scores, which is the allocator with one multiplier per day; the single multiplier is a secondary variant reported with its realized volume and the days the cap binds. The allocator is still tested against brute force. SPEC Section 4 is updated
- METHODS 12.4 first defined the allocator with the largest maximizer only; with it the smallest feasible multiplier need not exist at a breakpoint, so the definition now uses the smallest maximizer and fills ties up to the largest. Only the allocator's tests use it

## D029 Statistics for the policy comparisons
- Date: 2026-11-08
- 10,000 bootstrap resamples instead of the 1,000 of D012: at 1,000 the first Holm step rests on a handful of tail draws
- Per setting, each of the four primary comparisons is tested at 0.0125 (Bonferroni) instead of Holm, so one comparison's verdict does not depend on another's p-value
- The sweep average is bootstrapped over settings as well as users, so its interval covers the choice of the 64 settings and their fit and world randomness; per-setting intervals are conditional on the setting's fitted models
- Holm runs across the eight sweep-level tests (four comparisons in the main and the annoyance-free sweep). Claim labels are in METHODS 12.7

## D030 Design choices made after a synthetic replica
- Date: 2026-11-08
- Before this pre-registration, the independent review of the draft ran the simulator on synthetic inputs, with visit rates and the mean $p_0$ of pushed items matched to D024. It used no KuaiRand evaluation user and no real policy run. What it showed:
  - Score quality against the true same-day uplift (Spearman): T-learner $\hat\tau$ 0.07, predicted open probability 0.57, the decomposed score $\hat p (1 - \hat\mu_0)$ 0.77. $\hat\tau$ had an SD of 0.021 against a true 0.0045, and the T-learner picked the true best slot on 26% of user-days (open probability 38%, chance 25%)
  - In a 64-point hypercube, one push a day lowered active days below no pushes in 56% of settings; its lift fell inside the 0.07% to 0.5% band in 6%, below in 58% and above in 36%, and inside in none with $\epsilon_v = 0$. Annoyance at one push a day costs about $(1 - e^{-\epsilon_v})\bar v$ of daily active share, 2.8 points at the reference, against about 0.5 points from opens
  - Per-setting standard errors of 0.0014 to 0.012 active days, so differences below about 0.04 days are not detectable per setting with 5,000 users
- Design changed because of it: the decomposed gate G2 became a primary comparison (H2); S, G and G2 choose the slot by predicted open probability and differ only in which user-days they send; the per-day budget (D028); 10,000 resamples and the per-setting levels (D029); the T½ split of H3 and the comparisons against no pushes as secondary results; the band is read by interval
- Not changed: the M4 parameter ranges. H1 and H2 compare equal volumes, so annoyance acts on both sides; the comparisons it moves most (H3, H4 and those against no pushes) are also reported on the annoyance-free sweep and the one-at-a-time grid, which gives the annoyance elasticity at which a daily push breaks even

## D031 Sweep design, users and cost
- Date: 2026-11-08
- Users: 5,000 for evaluation (the quick sample), 10,000 of the others for fitting the policies' models, 2,000 more for score diagnostics; disjoint
- Settings: 64 main (Latin hypercube over 13 parameters), the same 64 with $\epsilon_v = 0$, the same 64 with the opt-out stress value, 61 one-at-a-time points and 10 refits at the reference, 263 in all. Three evaluation run seeds per setting
- Cost measured on synthetic inputs of full size, 4 processes with one thread each: fitting the three LightGBM models takes about 20 s per setting and one model predicts 28 days for 5,000 users in about 3.5 s, so the sweep takes about 2 to 3 hours (3 to 4 with five seeds). This replaces the 40-minute estimate in SPEC Section 10, which assumed no fitted policies
- Design points and seeds are frozen in `results/policy_design.json` and pushed before any policy runs on evaluation users. A later change needs a new decision with its reason, and results before and after the change are both reported

## D032 Random streams keyed by purpose
- Date: 2026-11-08
- NumPy's seed sequence ignores trailing zeros: `default_rng([s, 0, 0])` gives the same numbers as `default_rng(s)`. The exploration policy keyed its coins by `[seed, day, slot]`, so at day 0, slot 0 they were the open draws of `env.draws(seed)` whenever the two seeds were equal. Found while testing the M5 policies, where it raised the open rate of the first slot from 0.13 to 0.38 in a synthetic run. In M4 only the timing run of `sim.build` used equal seeds, and no outcome of it was reported
- Every keyed stream is now `default_rng([purpose, *key, 1])` (`world.stream`): the leading purpose separates uses and the closing 1 keeps trailing zeros from being dropped. Integer seeds used in one run (world copy effects and draws) are kept distinct by the sweep design; profile noise uses a keyed stream. A test checks that the exploration, template, coin, half-day and noise streams and the open, opt-out and visit draws of one seed are uncorrelated at day 0, slot 0
- Checked: rerunning `sim.build` after the fix reproduces `data/processed/sim/inputs.npz` bit for bit and changes only the timings in `results/simulator.json`, so the committed M4 results are not affected

## D033 Changes to the design before it was frozen
- Date: 2026-11-08
- Made after the independent code review and before `results/policy_design.json` was committed or any policy ran on evaluation users
- Evaluation run seeds are 101 to 103 instead of the 1 to R written in METHODS 12.6: seeds 1 and 2 already select the fit and diagnostic users, and every integer seed of the design should have one use (D032)
- The secondary budget $b = 1$ is dropped: with one push a day allowed to every eligible user, S, G and G2 send to the same users at the same slot and are one policy
- F's slot is the largest share of the perturbed profile averaged over users, as implemented; METHODS 12.2 now says so instead of "population share", which 11.1 uses for pooled counts
- Every setting records the code commit it ran with; the runner refuses a modified working tree and will not resume over settings from another commit. `results/policy.json` gets labels and Holm decisions only when all main and annoyance-free settings are present
