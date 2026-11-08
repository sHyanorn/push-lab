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
- Before this pre-registration, the simulator was run on synthetic inputs, with visit rates and the mean $p_0$ of pushed items matched to D024. It used no KuaiRand evaluation user and no real policy run. What it showed:
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
- Made before `results/policy_design.json` was committed or any policy ran on evaluation users
- Evaluation run seeds are 101 to 103 instead of the 1 to R written in METHODS 12.6: seeds 1 and 2 already select the fit and diagnostic users. Within one simulation run no integer seed now drives two random processes (D032); seeds 0 to 2 also serve the user samples, the hypercube and the analysis bootstrap, which run no simulation
- The secondary budget $b = 1$ is dropped: with one push a day allowed to every eligible user, S, G and G2 send to the same users at the same slot and are one policy
- F's slot is the largest share of the perturbed profile averaged over users, as implemented; METHODS 12.2 now says so instead of "population share", which 11.1 uses for pooled counts
- Every setting records the code commit it ran with; the runner refuses a modified working tree and any setting on disk from another commit, with or without `--ids`, and the comparison refuses settings from more than one commit and records the commit in `results/policy.json`. `results/policy.json` gets labels and Holm decisions only when all main and annoyance-free settings are present

## D034 Policy sweep: cost and results
- Date: 2026-11-08
- Run alone at commit 7490c2a with the frozen design: 263 settings in 6,097 s on 4 processes (main-sweep settings about 130 s, others about 80 s), peak resident memory 2.2 GB per worker
- Pre-registered results (METHODS 12.7), Tier C, reported as they came out. Differences are in active days per user over 28 days (about 16.2 without pushes); sweep averages with two-level 95% intervals; wins, losses and neither of 64 settings at level 0.0125; Holm across the eight sweep tests rejects all but main H4. A p-value of 0 means below the resolution 0.0002

| Id | Sweep | Label | Sweep average (95% CI) | Wins / losses / neither |
|---|---|---|---|---|
| H1 G vs S | main | worse on average | -0.015 (-0.021 to -0.009) | 1 / 29 / 34, per-setting verdicts understate fit noise |
| H1 G vs S | annoyance-free | loses in most sampled settings | -0.022 (-0.027 to -0.018) | 0 / 56 / 8, per-setting verdicts understate fit noise |
| H2 G2 vs S | main | beats in most sampled settings | 0.056 (0.048 to 0.064) | 64 / 0 / 0 |
| H2 G2 vs S | annoyance-free | beats in most sampled settings | 0.022 (0.017 to 0.026) | 58 / 0 / 6 |
| H3 G (half volume) vs T | main | depends on the setting | 0.093 (0.052 to 0.136) | 43 / 19 / 2, per-setting verdicts understate fit noise |
| H3 G (half volume) vs T | annoyance-free | loses in most sampled settings | -0.161 (-0.179 to -0.143) | 0 / 64 / 0, per-setting verdicts understate fit noise |
| H4 T vs F | main | no detectable difference | 0.000 (-0.004 to 0.005); median per-setting half-width 0.011 | 6 / 0 / 58 |
| H4 T vs F | annoyance-free | worse on average | -0.005 (-0.010 to -0.001), p = 0.019 against a Holm step of 0.025 | 0 / 6 / 58 |

- H2: G2 assumes an open and an organic visit are independent given the features; this holds in the simulator by construction, so part of any H2 win comes from the simulator's structure
- Stress sweep (opt-out 0.005, outside the evidence), wins / losses / neither: H1 2 / 27 / 35, H2 64 / 0 / 0, H3 42 / 19 / 3, H4 5 / 1 / 58
- Descriptive, one-at-a-time grid with the other parameters at reference values: one push a day (F) gains active days over no pushes only below an annoyance elasticity of 0.016 (95% CI 0.015 to 0.017); at the reference elasticity 0.05 it gains only above a base-rate factor of 0.18 (0.17 to 0.19). The half-volume gate beats T above an annoyance elasticity of 0.013 (0.012 to 0.015). Every other crossing of H1 to H4 is not resolved. Crossings are located by linear interpolation between adjacent grid points (0 and 0.025 for the elasticity, 0.1575 and 0.2 for the base rate); their intervals cover user sampling with the grid's seeds fixed, not the interpolation or model-fit randomness
- Plausibility band, lift of F over N in daily active share across the 64 main settings: 37 below 0.07%, 2 inside, 21 above 0.5%, 4 straddling. At the reference setting the lift is -2.3% (-2.4% to -2.2%). Under the registered reading, one push a day adds less activity than the smallest reported program change in most settings, and conclusions that rely on push benefits are weaker for it
- Secondary, main sweep medians over settings: H3 splits into G minus T½ +0.026 (43 positive, none negative) and T½ minus T +0.052 (38 positive, 22 negative); in the annoyance-free sweep T½ minus T is negative in all 64 (median -0.168). L1: T on the cascade list is below T on the two-tower list in all 64 settings (median -0.012), which the two-tower list's share in the response model favors by construction (D026). Opt-out by day 28: 0.8% to 0.9% under a daily push, 0.3% under the scored policies at half volume
- Score diagnostics, main-sweep medians: Spearman correlation with the true same-day uplift 0.51 for the open model, 0.21 for the T-learner and 0.75 for the decomposed score; the slot of highest predicted open probability is the true best slot on 27% of user-days (chance 25%; the truth includes the template drawn for each slot)
- Fit variability at the reference setting over 10 refits: SD of H1 0.017 against a mean bootstrap SE of 0.005, H2 0.005 against 0.004, H3 0.013 against 0.006 (H4 does not use the models). Per-setting H1 and H3 verdicts are conditional on one fit and understate their uncertainty, so the win and loss counts behind their labels are less certain than their intervals suggest; the sweep averages include fit randomness
- One multiplier (secondary, budget 0.5): S and G2 hit the daily cap on nearly every day (median 28 and 27 of 28); G delivered a median 12.6 pushes per user against the budget of 14, with the cap binding on a median 10 days

## D035 Copy language and placeholders
- Date: 2026-11-08
- Copy is English. KuaiRand comes from a Chinese app, but the repository is English and the simulator does not read the text, so the language changes no result; English filters and copy are also easier for readers of the repository to check
- Placeholders are filled at send time from data before the cutoff $c_2$ or from the simulated history a policy observes (METHODS 13.1), like every other simulator input; fill values are never committed, the committed copy holds placeholders only, and rendered examples in tests and the report use synthetic values
- Popular now uses the item's rank in the trending list at $c_2$ (METHODS Section 2) instead of `video_features_statistic`, which METHODS Section 1 excludes because its window overlaps the evaluation period. Creator update may not claim that a video is new: upload times are not tied to simulated days

## D036 Eligible templates and template numbering
- Date: 2026-11-08
- Creator update, Popular now and Welcome back are eligible only where their placeholder can be filled (METHODS 13.1), so the template bandit faces arms that sleep, as in Yancey and Settles (2020). Eligibility is confounded with the open probability (author affinity is a feature of the response model), which is what the difference score is designed to remove; the secondary policy RAW shows the effect of ignoring it. The eligibility attributes are added to the cached simulator inputs, computed from data before $c_2$
- Each family is asked for 16 variants and the first 10 that pass the filters are kept, in the order returned. Template $k = 6j + f$ is the $j$-th kept variant of family $f$, so $\text{family}(k) = k \bmod 6$ as the simulator has assumed since M4: with 60 templates the copy effects of every world are unchanged and the M5 results stay reproducible. Nothing is regenerated; the template sweep requires 10 per family, which held (D038)
- The M5 policies keep drawing templates uniformly over all $K$ without eligibility; the simulator does not read the text, so this changes none of their results

## D037 Template bandit design
- Date: 2026-11-08
- The recovering difference softmax of Yancey and Settles (2020, Eq. 1 to 4) with their production recency penalty ($\gamma_b = 0.017$, $h_b = 15$ days) and probability threshold ($\theta = 0.005$). The bandit does not know the simulator's novelty parameters, which the sweep varies
- Reward: whether the push was opened, observed after the day ends; scores are updated once a day from all earlier rounds of the run, pooled over users, as in the paper's daily batch job
- Departures from the paper: the prior strength is set each day by the method of moments, since no arm here has the large sample the paper estimated it from; the softmax temperature is 0.005 instead of 0.0025, because each arm sees tens of thousands of rounds instead of millions; arm histories are not truncated, since the 28-day run is shorter than the paper's one to five weeks of history. All three are fixed before any run

## D038 Generated copy
- Date: 2026-11-08
- Six requests, one per family, all ended normally; 5,667 input and 12,898 output tokens in total (about 0.28 USD at the model's list price). The model, request ids and usage are in `data/creative/manifest.json`; the catalog, manifest and `results/copy.json` rebuild byte for byte from the committed responses
- All 96 variants passed all six filters, so the first 10 of each family form the catalog: $K = 60$ in 6 families, and the M5 copy draws are unchanged (D036)
- Read after generation, with the filters left as fixed beforehand, kept templates make claims the data cannot back that the family rules did not cover: the video is short or quick (20, 28 and 50, where nothing is known about its length), new or fresh (8, 16, 22, 26, 35 and 41), and Popular now says most watched "lately" or "recently" (1, 7, 13, 19, 37, 43 and 55) although the rank is fixed at $c_2$ and simulated days are not calendar days; the family brief in the prompt itself says "recently". They stay in the catalog; the blind rating counts how often kept copy reads as misleading
- Single banned words first matched only in the singular, so "rewards" or "prizes" would have passed; plurals are now banned too. No generated variant contains one, and the catalog, manifest and counts are unchanged
- With no rejected variant, the 20 extra items of the rating are drawn from the 36 that passed but were not kept (METHODS 13.3)

## D039 Template comparisons, pre-registered
- Date: 2026-11-08
- Comparisons, metric, decision rule and labels as in METHODS 13.5: B1 (RD against U) and B2 (RD against D0) on opens per user over 28 days, each at 0.025 per setting, Holm across the two sweep averages of the main sweep at 0.05, the labels of 12.7. Secondary results carry intervals and no tests. Results are reported as they come out
- Design: 94 settings frozen in `results/template_design.json`, generated from `results/policy_design.json` (its sha256 is recorded): the 64 main points with their world and perturbation seeds and run seeds 101 to 103; 20 one-at-a-time points for base rate, copy spread, novelty penalty and novelty half-life, the reference among them; the reference with 10 more run-seed triples (201 to 230). METHODS 13.5 first said 17 grid points; the frozen grid has 20, since only copy spread has the reference on its grid
- All 27,285 users, every setting; 10,000 bootstrap resamples of users (seed 0), settings resampled with seed 1, as in M5
- Before this pre-registration, the template policies were run on synthetic inputs of full size (27,000 users, open rate near 1.7%), with no KuaiRand user and no setting of the design. Opens of RD over U: +0.2% at $\sigma = 0$ and +31% at $\sigma = 0.2$ (myopic oracle +0.3% and +39%), averaged over two worlds. On 2,000 users the daily method-of-moments prior followed noise in the first days and RD locked onto weak templates (below U at $\sigma = 0.05$), which is why every setting uses all users. A fixed prior strength was tried and was worse at $\sigma = 0$ (−0.9% against U); the design of D037 is unchanged
- Cost on synthetic inputs of full size: about 43 s per setting on one process, so about 17 minutes on 4 processes; the comparison holds 1.1 GB of bootstrap weights, about 2 s per setting, peak memory 2.5 GB
- The comparisons are equal-volume: every policy sends one push a day at the same slot, so annoyance and opt-out act alike on both sides. The report states the M5 background with them: at the reference setting one push a day lowers active days by 2.3% against no pushes, and a daily push has a net loss above an annoyance elasticity of about 0.016 (D034)
- Blind rating by one rater (METHODS 13.3), counts in `results/copy.json`: of the 60 kept templates, 60 were rated clear, 54 fitting for a short-video push and none misleading; of the 20 that passed but were not kept, 19, 15 and none. None of the templates listed above as making unbackable claims was rated misleading; one rater cannot show that such claims do not mislead readers. The rater may have seen the catalog before rating, so the rating may not have been blind to which items were kept

## D040 Template comparisons: cost and results
- Date: 2026-11-08
- Run alone at commit 365f040 with the frozen design: 94 settings in 1,075 s on 4 processes (about 43 s each), 1,178 s with the comparison; peak resident memory 2.2 GB. Intervals for the secondary summaries were added after the first report by rerunning the comparison on the same runs; every pre-registered number is unchanged
- Pre-registered results (METHODS 13.5), Tier C, reported as they came out. Differences are in opens per user over 28 days (uniform templates: a mean 0.73 over the main settings, 0.37 at the reference); sweep averages over the 64 main settings with two-level 95% intervals; wins, losses and neither at level 0.025; Holm across the two sweep tests rejects B1 only. A p-value of 0 means below the resolution 0.0002. The win and loss counts rest on per-setting intervals that cover user sampling only: at the reference, the SD over 10 seed triples is 17.7 times the mean bootstrap SE for B1 and 19.2 times for B2

| Id | Label | Sweep average (95% CI) | Wins / losses / neither | Seed SD over bootstrap SE |
|---|---|---|---|---|
| B1 RD vs U | beats in most sampled settings | 0.079 (0.053 to 0.109), p = 0 | 57 / 7 / 0 | 17.7 |
| B2 RD vs D0 | depends on the setting | 0.001 (−0.002 to 0.005), p = 0.45 | 28 / 28 / 8 | 19.2 |

- Seed variability (registered secondary): at the reference the SD over 10 seed triples is 0.0080 for B1 against a mean bootstrap SE of 0.00045, and 0.0073 for B2 against 0.00038. The win and loss counts and every grid crossing rest on one seed triple per setting and are far less certain than their intervals show; the sweep averages resample settings with different seeds and include this randomness
- At the reference setting, whose base rate comes from published open rates, there is no detectable gain: B1 averages 0.0015 over the 10 triples (t interval −0.0042 to 0.0072; 4 of 10 negative), 0.4% of U's opens against 8% for the oracle. B2 averages −0.0052 (−0.0104 to 0.0001; 7 of 10 negative), about −1.4% of U's opens: at the reference the evidence leans toward the recency penalty costing opens, without reaching significance
- B2 over the sweep: the penalty changes a choice only where learned scores lie within about $\gamma_b = 0.017$ of each other. Repeat rates over the main sweep (ratio of averages) are 0.78 (0.74 to 0.81) for RD and 0.81 (0.79 to 0.84) for D0, against 0.24 for U and 0.82 for the oracle, so the penalty rotates templates only a little at the scores learned here. The registered rule labels B2 "depends on the setting"; with a 28 to 28 split and the seed variability above, the sweep shows no detectable effect of the recency penalty on average
- Secondary, main sweep (two-level intervals): D0 − U 0.078 (0.052 to 0.108), so nearly all of B1 comes from the learned scores; RAW − RD −0.001 (−0.005 to 0.003), so ignoring eligibility cost nothing detectable; O − U 0.139 (0.107 to 0.174). Relative to U's average opens, RD gains 10.9% (7.6% to 14.5%) and the oracle 19.1% (15.2% to 23.2%); RD reaches 57% (48% to 64%) of the oracle's gain as a ratio of averages (the median of per-setting shares is 41%). Active days follow opens: B1 0.045 days per user (0.030 to 0.062). At day 28 the learned scores have a median Spearman correlation over the main settings of 0.33 (0.28 to 0.38) with the log of the drawn effects (D0 0.34, RAW 0.32) and share a median 3.3 (3.0 to 3.7) of their top 10 templates with the true top 10
- Grid, one seed triple per point: on the copy-spread grid B1 is −0.002 at $\sigma = 0$ and +0.076 at $\sigma = 0.2$, and it grows along the base-rate grid. The registered rule marks crossings of B1 on the copy-spread grid (at 0.017) and of B2 on the base-rate and copy-spread grids as resolved; their intervals cover user sampling only, so given the seed variability they are not read as thresholds. Along the novelty-penalty grid B1 falls steadily, 0.008, 0.006, 0.002, −0.000 and −0.003 as $\gamma$ goes from 0 to 0.05, while B2 stays near −0.010; the points share world and run seeds, so this is a paired trend. A reading added after the results (post hoc): RD repeats templates far more often than U, so the simulator's novelty penalty takes back part of its gain, and from $\gamma \approx 0.04$ RD falls below U at the reference. The half-life moves B1 and B2 by less than 0.001
- Added after the results (post hoc, descriptive): across the main settings B1's gain relative to U correlates with copy spread at 0.76 (0.71 to 0.87, settings resampled), and all 7 settings where RD loses have $\sigma \le 0.03$. Under U the share of pushes per family (median over main settings) is 1.1% Creator update, 6.0% Popular now, 4.7% Welcome back and 29.4% for each always-eligible family
- What this supports: in the simulator, a bandit that learns from opens finds templates with larger drawn effects when templates differ by about 10% or more, or by less when base rates are well above the reference; at the reference setting it gains nothing detectable, and whether the recency penalty helps is not detectable, with the reference leaning toward a small cost. The simulator never reads the text, so nothing here says the generated copy is good or which wording works for real users. The comparisons are equal-volume; the M5 background holds: at the reference one push a day lowers active days by 2.3% against no pushes (D034)

## D041 Off-policy evaluation: tiers and units
- Date: 2026-11-08
- Tier B needs real logs and an estimator first checked against a known answer on real data (SPEC Section 5). The only such logs here are the Open Bandit Dataset, so the project's one Tier B result is the BTS policy value estimated from OBD's random logs. Every estimate on simulator logs and every check against simulator truth is Tier C and is reported apart from it
- The BTS target probabilities follow the Open Bandit Pipeline at the commit pinned for the data (`sources.py`): Monte Carlo draws from the production Beta prior of the Men's campaign, top three items in order. The pipeline itself is not a dependency (D006)
- In the simulator a round is a user-day and its action the vector of four slot send decisions; templates are uniform in the logs and in every target, so their probabilities cancel. The smallest propensity of one slot-and-template action (0.00125) therefore does not bound the weights; the largest weight of any target is 123
- No clipping or switching: every weight is used, and each estimate reports its effective sample size and largest weight

## D042 Off-policy checks, pre-registered
- Date: 2026-11-08
- Full Open Bandit Dataset: the archive is 412,931,917 bytes (sha256 pinned in `sources.py`); the Men's campaign files are 63 MB of it compressed and 1.48 GB extracted, with 27 GB of disk free, so it is used as the secondary OBD check of METHODS 14.2 under the same rule. It is downloaded and prepared only when named (`download --only obd_full`, `prepare --only obd_full`) and never in CI; conversion reads the two Men's files from the archive without extracting the others and takes 10 s with a peak of 4.1 GB. The campaign has 452,949 random impressions with 2,321 clicks and 4,077,727 BTS impressions with 27,495 clicks, against 46 and 69 clicks in the small sample
- Decision rules, estimators, targets and rewards as in METHODS 14.1 to 14.4: on OBD, an estimator is validated if its 95% interval contains the BTS click rate and its median relative error is below 0.5, on the small sample (primary) and the full campaign (secondary); in the simulator, a (setting, target, estimator) is validated if the Wilson interval of coverage over 50 log seeds contains 0.95 and the bias interval contains 0. The primary results are DR and IPS at the reference setting, reported after the no-carryover control; if DR or IPS fails the control for T, T½ or N, the simulator results count as an implementation failure. Every verdict is reported
- Design, fixed in `ope/simulator.py` at the commit that runs it: the 5,000 evaluation users of M5, world seed 10,100, log seeds 1,001 to 1,050, truth run seeds 2,001 to 2,020, the control and reference settings, the policy profile without perturbation; 1,000 bootstrap resamples per estimate (seed equal to the log seed); BTS probabilities from $10^6$ prior draws with seed 0
- Cost on synthetic inputs of full size: about 51 s per log on one process (two outcome models with 2-fold cross-fitting dominate), so about 22 minutes on 4 processes; 0.4 s for the six truth runs of one seed; peak memory 0.6 GB per process

## D043 Off-policy checks: cost and results
- Date: 2026-11-08
- Run alone at commit 2d120df: the OBD checks in 146 s (peak 4.2 GB, the full campaign's outcome model dominating), the simulator check in 1,330 s on 4 processes
- Pre-registered results (METHODS 14.4), reported as they came out
- OBD, BTS click rate estimated from the random logs of the Men's campaign (on-policy value with a 95% interval; estimate with its 95% interval; median relative error over paired resamples):

| Sample | On-policy | DM | IPS | SNIPS | DR |
|---|---|---|---|---|---|
| Small: 10,000 random (46 clicks), 10,000 BTS (69 clicks) | 0.0069 (0.0053 to 0.0085) | 0.0045 (0.0044 to 0.0046), 0.36, not validated | 0.0045 (0.0025 to 0.0069), 0.37, validated | 0.0046 (0.0025 to 0.0069), 0.35, not validated | 0.0047 (0.0026 to 0.0071), 0.34, validated |
| Full: 452,949 random (2,321 clicks), 4,077,727 BTS (27,495 clicks) | 0.00674 (0.00667 to 0.00682) | 0.00593 (0.00591 to 0.00595), 0.12, not validated | 0.00601 (0.00552 to 0.00648), 0.11, not validated | 0.00612 (0.00563 to 0.00662), 0.09, not validated | 0.00608 (0.00560 to 0.00655), 0.10, not validated |

- Reading of OBD: on the small sample IPS and DR pass the registered rule, so their estimates are the Tier B results, but their intervals only just reach the on-policy value and SNIPS misses it by 0.00001. With 50 times the clicks, every estimator underestimates the BTS click rate by about 10% and none is validated. The small-sample pass reflects low power; the Tier B estimates carry this about 10% underestimate as a known limitation
- Simulator, registered verdicts. In the no-carryover control, IPS, SNIPS and DR are validated for T, R, F and T2 and not validated for T½ and N, with positive biases of 0.0009 to 0.0016 in active-day share and coverage of 0.96 to 1.00; DM is biased for every target, as expected. By the registered rule, the failure of IPS and DR on T½ and N in the control makes the simulator results an implementation failure, and the reference results are not interpreted
- Reference results, not interpreted under the rule (bias of DR in active-day share; truth in brackets): T +0.0007 (0.564), T½ −0.0088 (0.571), R +0.0004 (0.564), N −0.0177 (0.577), F +0.0008 (0.564), T2 +0.0189 (0.549); IPS, SNIPS and DR agree within 0.001 for every target
- Added after the results (post hoc). In the control every day is independent and templates are uniform, so each target's value can be computed exactly from the simulator's inputs. All six simulated truths lie 0.0004 to 0.0005 below their exact values (1.6 to 1.9 truth SE each). The registered interval did include each target's own truth SE; because every target's truth came from the same 20 run seeds, the six truth errors were one event rather than six, and that event fell low. Against the exact values, the control biases of IPS, SNIPS and DR contain 0 for all six targets (N: IPS +0.0012, −0.0002 to 0.0025; DR +0.0006, −0.0001 to 0.0014), and DM stays biased. The failure of IPS and DR on T½ and N is consistent with a chance false alarm among the 12 unadjusted tests of the implementation check (two estimators, three targets, bias and coverage), helped by that low truth draw and a small positive deviation of the estimates. Coverage against the exact values cannot be recomputed, since per-seed intervals were not saved
- Post hoc reading of the reference, given that diagnosis: per-day OPE does not see carryover, and its bias is 22 to 37 times its standard error for the targets whose volume differs most from the logs. It underestimates no pushes by 3.1% and overestimates two pushes a day by 3.4% at the reference setting; D046 narrows this reading
- Two readings of the Tier B result, both kept. Registered: the small-sample IPS and DR estimates of the BTS value are the Tier B results, since they pass the rule. Stronger: on the full campaign, a check on real logs against a known answer, IPS and DR estimate the BTS click rate at 0.89 and 0.90 of its on-policy value with intervals that exclude it, a shortfall the small sample has too little power to show. They do not conflict: the registered definition names the estimate, and the second states how accurate such an estimate was found to be. The evidence matrix lists the shortfall first and each small-sample estimate under it with its outcome on both samples
- Post hoc on OBD: the deployed BTS was not the fixed prior (its logged propensities vary from 0.0002 to 0.73 across rows of the small sample and from 0.000001 to 0.73 in the full campaign). Using the empirical (item, position) frequencies of the BTS logs as the target moves the full-campaign IPS estimate from 0.89 to 0.92 (0.84 to 0.98) of the on-policy value, with the interval still excluding 1, so the drift explains part of the underestimate and the rest stays unexplained here

## D044 Amended simulator check, pre-registered
- Date: 2026-11-08
- Made after the first results (D043). The first run's registered verdict, an implementation failure, stands and is reported with the amended one; neither replaces the other
- Changed: how the truth is computed and what is recorded. In the control, every target's truth is its exact value (METHODS 14.3), computed from the simulator's inputs; at the reference, each target's truth averages 200 online runs whose seeds no other target shares ($100{,}000 + 1{,}000\,i + k$). Each log's estimates and intervals are saved under `results/runs/ope/` (not committed), so coverage can be recomputed against any truth. Results go to `results/ope_simulator_amended.json`; `results/ope_simulator.json` keeps the first run
- Unchanged: the estimators, outcome models, targets, rewards, users, world seed, log seeds 1,001 to 1,050, bootstrap, settings and the decision rule of METHODS 14.4, including that a failure of DR or IPS in the control for T, T½ or N makes the run an implementation failure
- The OBD check is rerun only to add each estimate's ratio to the on-policy value with its paired-resample interval; every other OBD number must come out unchanged, which the rerun checks
- Cost: about 22 minutes for the logs as before, plus about 1 minute for the 1,200 truth runs

## D045 Amended simulator check: results
- Date: 2026-11-08
- Run alone at commit 553fad2 with the design of D044: 1,427 s on 4 processes, peak 4.3 GB. The OBD rerun before it reproduced every earlier OBD number and added each estimate's ratio to the on-policy value: on the full campaign IPS 0.891 (0.817 to 0.964) and DR 0.901 (0.828 to 0.973), on the small sample IPS 0.66 (0.34 to 1.08) and DR 0.68 (0.36 to 1.10)
- Pre-registered results (METHODS 14.4, D044), Tier C, reported as they came out; bias of DR in active-day share with its 95% interval, coverage over 50 log seeds:

| Target | Control (exact truth) | Reference (200 independent truth runs) |
|---|---|---|
| T | +0.0002 (−0.0003 to 0.0006), 1.00, validated | +0.0003 (−0.0002 to 0.0008), 1.00, validated |
| T½ | +0.0004 (−0.0001 to 0.0009), 1.00, validated | −0.0091 (−0.0096 to −0.0086), 0.04, not validated |
| R | −0.0000 (−0.0011 to 0.0010), 1.00, validated | −0.0001 (−0.0011 to 0.0008), 1.00, validated |
| N | +0.0006 (−0.0001 to 0.0014), 1.00, validated | −0.0181 (−0.0189 to −0.0173), 0.00, not validated |
| F | −0.0003 (−0.0022 to 0.0015), 0.98, validated | +0.0003 (−0.0017 to 0.0023), 0.98, validated |
| T2 | −0.0015 (−0.0031 to 0.0001), 0.96, validated | +0.0182 (0.0166 to 0.0199), 0.18, not validated |

- In the control, IPS, SNIPS and DR are validated for all six targets and DM for none, so the implementation check passes. The amended run reused log seeds 1,001 to 1,050, so every mean estimate equals the first run's, and the bias half of this check had already been computed against the exact values in D043 before D044 was written; only coverage was new. This pass is therefore weak confirmation; fresh log seeds would be needed for a real one. At the reference, IPS and DR give the same verdicts as each other for every target: validated for T, R and F, not validated for T½, N and T2 (bias and under-coverage). SNIPS agrees; DM is biased everywhere except F
- Reading, at the reference setting only: per-day OPE is accurate for the targets that send about one push a day, as the logs do (T, R, F, even with weights up to 123), and biased for those whose volume differs. DR underestimates half the volume by 1.60% (1.51% to 1.69%) and no pushes by 3.13% (2.99% to 3.27%), and overestimates two pushes a day by 3.32% (3.02% to 3.61%); IPS is within 0.1 points. The expectation stated before the run named N and T2 as biased and T and T½ as close; T½ is biased too. The direction of the bias depends on which carryover dominates and is not a general property of per-day OPE (D046)
- Secondary reward, opens per user-day at the reference: DR is validated for T, R, N and F and not for T½ and T2 (bias +0.0057 opens per user-day for T2)
- Both runs are reported: the first run's registered verdict (D043), an implementation failure caused by a truth estimate shared across targets, stands; the amended run passes its implementation check and gives the results above

## D046 Corrections after the M7 results
- Date: 2026-11-08
- Simulator results are never shown without the sensitivity sweep (CLAUDE.md). The simulator OPE check ran at two settings only, so its results are not yet conclusions: the evidence matrix marks them as awaiting a sweep, and D045's reading is limited to the reference setting
- The direction of the reference bias comes mainly from annoyance: per-day weights do not see that recent volume lowers organic visits, so fewer pushes look worse and more pushes look better. Return carryover acts the other way (an extra open raises later organic visits, which per-day weights also miss). In a check outside the registered runs (IPS, 15 log seeds, 30 truth runs per target), with annoyance 0 and return boost 0.1, both inside the swept ranges, the bias of no pushes was +0.0030 (±0.0030) and of two pushes a day +0.0007 (±0.0041); with fatigue and opt-out also 0, +0.0036 and −0.0030. So "per-day OPE makes extra pushes look better" holds at the reference and is not general
- OBD resampling: the rows of OBD are the positions of one impression (the full random logs have 452,949 rows in about 136,000 impressions), and METHODS 14.1 names impressions as the resampling unit. The code first resampled rows and split folds by row; it now groups rows into impressions (same user features, less than a second apart) for folds, bootstrap and the on-policy interval. Rerun at commit 3f26e5d: the small-sample intervals widen and SNIPS now also passes (IPS, SNIPS and DR pass, DM fails); on the full campaign no estimator passes, with IPS at 0.891 (0.810 to 0.969) and DR at 0.896 (0.816 to 0.968) of the on-policy value. The OBD numbers in D043 and D045 are those of the row-level run; `results/ope_obd.json` now holds the impression-level run and records its commit
- Estimates whose intervals do not overlap are now flagged inconclusive (SPEC Section 5); no OBD estimate is
- Tiers on OBD, replacing the readings of D043: the full-campaign ratios (IPS 0.891, DR 0.896 of the on-policy value) use randomized real data to measure the estimators themselves, so they are Tier A results about the estimators, not policy values. Each ratio mixes estimator error with a target mismatch: the target is BTS with its fixed production prior, and the deployed BTS was not that; post hoc, re-weighting to the BTS logs' own frequencies moves IPS to 0.92 (0.84 to 0.98). No estimator passed on the full campaign, so the small-sample passes are kept only as the record of a registered verdict, and the project has no Tier B result

## D047 Carryover sweep of the simulator check, pre-registered
- Date: 2026-11-08
- Made after the results of D043 and D045, to meet the rule that simulator results are shown only with a sweep and to find where the direction of the per-day bias changes. Design, measures and reading as in METHODS 14.6: 24 settings on one-at-a-time grids of annoyance, fatigue, opt-out, return and novelty; 10 fresh log seeds per setting; 100 independent truth runs per target; flips of the DR and IPS bias of T½, N and T2 by the rule of 12.7
- Estimators, outcome models, targets, users and world seed as in 14.3. Until these results exist, the simulator OPE results of D043 and D045 stay out of the conclusions; afterwards, statements about the direction of the bias are made only with the ranges where the sweep shows it
- Cost: 240 logs at about 51 s each and 144 truth jobs of 100 runs, about 55 minutes on 4 processes

## D048 Carryover sweep: results
- Date: 2026-11-08
- Run alone at commit 0b454e6 with the design of D047: 3,529 s on 4 processes, peak 594 MB per process
- Pre-registered results (METHODS 14.6), Tier C, reported as they came out. Verdicts of 14.4 over the 24 settings: IPS and DR are validated for T, R and F at all 24. For T½, N and T2 they are not validated at 23; the exception is annoyance 0, where IPS is validated for all three and DR for T2 only
- DR bias in active-day share along the annoyance grid, other parameters at the reference, with resample intervals:

| Annoyance | T½ | N | T2 |
|---|---|---|---|
| 0 | +0.0011 (0.0003 to 0.0021) | +0.0030 (0.0015 to 0.0044) | +0.0017 (−0.0013 to 0.0044) |
| 0.025 | −0.0038 (−0.0047 to −0.0028) | −0.0071 (−0.0087 to −0.0054) | +0.0108 (0.0080 to 0.0133) |
| 0.05 | −0.0087 (−0.0097 to −0.0078) | −0.0171 (−0.0189 to −0.0155) | +0.0194 (0.0168 to 0.0217) |
| 0.075 | −0.0134 (−0.0144 to −0.0124) | −0.0266 (−0.0284 to −0.0248) | +0.0273 (0.0247 to 0.0297) |
| 0.1 | −0.0179 (−0.0189 to −0.0169) | −0.0361 (−0.0379 to −0.0343) | +0.0354 (0.0330 to 0.0378) |

- Flips by the rule of 12.7: the DR bias of T½ and N changes sign along annoyance, at 0.0058 (0.0017 to 0.0092) and 0.0074 (0.0043 to 0.0101). For IPS the signs change at the same place but the crossing is not resolved, since its intervals at annoyance 0 contain 0. T2 has no flip: the bias is positive at every point, and its interval contains 0 only at annoyance 0 for DR. IPS at annoyance 0 is +0.0052; its resample interval (0.0003 to 0.0102) excludes 0 and the interval of 14.4 (−0.0001 to 0.0105) does not. Fatigue, opt-out, return and novelty give no flip; over their grids the DR bias stays within −0.0091 to −0.0080 for T½, −0.0178 to −0.0154 for N and +0.0163 to +0.0202 for T2
- Expectation stated before the run: the T2 bias grows with annoyance and fatigue and falls with return boost. All three held; the fatigue and return effects are small (+0.0163 to +0.0202 and +0.0197 to +0.0185)
- Reading, within the swept ranges and one parameter at a time: per-day OPE is accurate for the targets that send about one push a day (T, R, F) at every setting. From an annoyance elasticity of 0.025 upward it underestimates half the volume and no pushes and overestimates two pushes a day, and each bias grows about linearly with annoyance; below about 0.006 (DR) the first two turn slightly positive. Fatigue, opt-out, return and novelty do not change any direction. This replaces the general wording of D046: in this sweep annoyance alone sets the direction; the D046 side check that moved two parameters at once lies outside a one-at-a-time grid
- Post hoc, after the results above: does the bias change which target looks better? For the differences N − T, T½ − T and T2 − T, the per-day estimate (mean over the 10 log seeds) is compared with the true difference (`post_hoc` in `results/ope_carryover.json`, `python -m push_lab.ope.carryover --post-hoc`). The estimates barely move along the annoyance grid (DR: N − T −0.0036 to −0.0049, T2 − T +0.0042 to +0.0049), while the truths change sign between 0 and 0.025 (N − T from −0.0065 to +0.0315, T2 − T from +0.0032 to −0.0307). From annoyance 0.025 upward, per-day DR reverses all three orderings, with both intervals excluding 0: it ranks one push a day above none and two above one where the simulator ranks them the other way. DR reverses each ordering at 23 of the 24 settings, all but annoyance 0; IPS does so at 23 for T2 − T and 22 for the other two (its interval also contains 0 at fatigue 0.8). In short, per-day OPE barely responds to annoyance, so in this simulator it ranks push volumes almost the same way whatever annoyance is
- With the sweep in place, the evidence matrix reports the simulator OPE results of D045 with, for the reference setting, the number of sweep settings where the same verdict holds, the two annoyance crossings, and the post hoc orderings at the reference
