# Methods

Each method lists the problem it solves, the formula, the source, where it lives in the code and how it is tested. Implementations must match this document; any deviation goes into `DECISIONS.md` first. Primary metrics and comparisons are fixed here before results are computed.

## Conventions

- KuaiRand rows are `tab == 1` only (D004). A *click* is `is_click == 1`; a *long view* is `long_view == 1`.
- Dates are the KuaiRand `date` field. Standard logs used for fitting run from 2022-04-09 to 04-21; random exposures from 04-22 to 05-08.
- Every random process takes an explicit seed; the default is 0.
- Confidence intervals: 1,000 bootstrap resamples of evaluation users with replacement, 95% percentile intervals (D012). A comparison is a paired bootstrap of the difference on the same resampled users; its two-sided p-value is twice the smaller of the shares of resampled differences at or below zero and at or above zero, capped at 1. Holm's step-down procedure is applied at family-wise level 0.05.
- Tier A means the number is measured on randomly exposed items (SPEC Section 5).

---

## 1. Splits and cutoffs

**Problem.** Rankers are fitted on logs produced by the production recommender and must be evaluated on data that this recommender did not choose, without any feature seeing the future.

**Cutoff rule.** Everything that produces a feature for decisions at cutoff $c$ (two-tower model, popularity, trending, user and item statistics) is fitted on standard logs with $\text{date} < c$. Two cutoffs are used:

| Cutoff | Fitted on | Used for |
|---|---|---|
| $c_1$ = 04-15 | standard logs 04-09 to 04-14 | features of the fine ranker's training rows |
| $c_2$ = 04-22 | standard logs 04-09 to 04-21 | retrieval, coarse and fine ranking at evaluation |

The window expands rather than slides: activity in the Pure logs is concentrated on 04-10 to 04-12, so a fixed 7-day window would give the two cutoffs very different amounts of data. Count features are divided by the number of days in the window so that they are on the same scale at both cutoffs. This does not remove all drift: the fine ranker learns from features built on a 6-day window ($c_1$) and is applied to features built on a 13-day window ($c_2$), so smoothed rates are based on more impressions at $c_2$ and are spread further from the window-wide mean. The evaluation reports the 5th, 25th, 50th, 75th and 95th percentiles of the main features (user and item impressions per day, user and item smoothed click rates, two-tower score) at both cutoffs as a diagnostic, without a test.

**Fine-ranker labels.** Standard-log impressions from 04-15 to 04-19 train the fine ranker; 04-20 and 04-21 are its validation set for early stopping. Their features come from cutoff $c_1$.

**Random exposures.** The users in the random-exposure log are sorted by id and permuted with `numpy.random.default_rng(0)`; the first half is the *evaluation half*, the rest the *simulator half* (used from M4 on). No model is fitted on either half. Duplicate (user, video) pairs within the random log are kept as separate impressions for AUC and NDCG and counted once for recall.

**Excluded data.** `video_features_statistic` is not used: per the KuaiRand README its fields are daily averages over one month, a window that overlaps the evaluation period.

**Undated user attributes.** The user table (`user_active_degree`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, follow, fan and friend counts, `register_days`, `register_days_range`, `onehot_feat*`) is a single snapshot with no date. It is constant within a user, but both models let it interact with item features (tree splits in the fine ranker, the inner product of the two towers), so it can change the order of items within a user and with it per-user AUC and NDCG as well as Recall@K. It is used as released, and this is a limitation of every ranking result. Refitting the fine ranker without these fields raises its fused GAUC (click) on the evaluation half, so in the fine ranker they do not inflate the reported number (D058).

**Code.** `data/splits.py`. **Tests.** No feature table built at cutoff $c$ contains a row with $\text{date} \ge c$; the two halves are disjoint, cover all random-exposure users and are identical across runs.

---

## 2. Non-personal sources

- *Popularity*: items ranked by clicks in the window before the cutoff.
- *Trending*: items ranked by clicks in the last 3 days before the cutoff; ties broken by all-window clicks, then item id.

Both serve as recall baselines; trending is also a candidate source (Section 4).

**Code.** `recall/popularity.py`.

---

## 3. Two-tower model

**Problem.** Score every user against every item cheaply enough to retrieve a personal candidate set from the whole catalog.

**Model.** A user tower $f$ and an item tower $g$ map features to 32-dimensional vectors, each followed by L2 normalization; the score is the inner product $s(u, i) = f(x_u)^\top g(x_i)$.

- User inputs: user id, `user_active_degree`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`, `register_days_range`, `onehot_feat0` to `onehot_feat17`; each an embedding, concatenated, then a two-layer MLP (64 units, ReLU).
- Item inputs: video id, author id, first tag, `upload_type`, `video_type`, duration bucket (deciles of `video_duration`); same structure.
- Ids not seen in the training window share one out-of-vocabulary embedding, so cold items are scored from their side features.

**Loss: in-batch softmax with log-Q correction** (Yi et al., 2019). Training examples are clicked (user, item) pairs from the window. For a batch of $B$ pairs $(u_b, i_b)$ and temperature $\tau$:

$$
\tilde s_{bj} = \frac{s(u_b, i_j)}{\tau} - \log q_{i_j}, \qquad
\mathcal L = -\frac{1}{B} \sum_{b=1}^{B} \left[ \tilde s_{bb} - \log \sum_{j=1}^{B} m_{bj}\, e^{\tilde s_{bj}} \right]
$$

- $q_i = n_i / \sum_k n_k$, with $n_i$ the number of clicks on item $i$ in the window: the probability that item $i$ is the positive of a uniformly drawn training pair, hence the rate at which it appears as an in-batch negative. It is computed exactly from the window rather than estimated by streaming.
- Without the correction, popular items are over-penalized as negatives because they appear in many batches.
- $m_{bj} = 0$ when $j \ne b$ and $i_j = i_b$ (another row in the batch has the same item, which is not a negative); otherwise 1.
- The correction applies in training only; retrieval ranks by $s(u, i)$.

**Fixed hyperparameters** (no tuning, to keep the comparisons pre-registered): $\tau = 0.05$, Adam with learning rate $10^{-3}$, batch 1,024, 5 epochs, seed 0, 4 threads.

**Index.** Item vectors go into a faiss HNSW index with inner-product metric (`M = 32`, `efConstruction = 200`, `efSearch = 1024`; raised from 128, see D018). Exact search (`IndexFlatIP`) is the reference.

**Reference.** Yi, X., Yang, J., Hong, L., Cheng, D. Z., Heldt, L., Kumthekar, A., Zhao, Z., Wei, L., Chi, E. (2019). Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. *RecSys 2019*. https://doi.org/10.1145/3298689.3346996

**Code.** `recall/two_tower.py`, `recall/index.py`. **Tests.**

- The loss equals a hand-computed value on a 3 × 3 example.
- With all $q_i$ equal, the loss equals plain in-batch softmax.
- A duplicated item in a batch is masked.
- HNSW overlap with exact search at $K = 200$ is at least 0.99 on synthetic unit vectors; the figure for the real evaluation users is reported in `results/ranking.json`.

Every reported retrieval number uses exact search; HNSW is the scalable path and is checked against it (D018).

---

## 4. Candidate pool and cascade

**Pool.** For each user, the union of the top 500 items by two-tower score and the top 100 trending items, deduplicated. Items the user already interacted with are *not* removed, so evaluation positives are not dropped by construction.

**Coarse ranking.** The two-tower score orders the pool and keeps the top $N$.

**Fine ranking.** The fine ranker (Section 5) scores the $N$ survivors; the top-1 is the item to push.

**Truncation study.** $N \in \{50, 100, 200, 500\}$ and *pool* (no truncation), against a *catalog* reference where the fine ranker scores all 7,583 items. Per evaluation user:

- *Survival* (Tier A): the share of the user's clicked random-exposure items that are in the cascade's top-$N$ set. This is the only truncation measure tied to real outcomes.
- *Top-1 agreement* (model-internal): whether the cascade's top-1 equals the catalog reference's top-1.
- *Score regret* (model-internal): the catalog top-1's fused score minus the cascade top-1's fused score.
- *Latency*: wall time of fine scoring per user (D056).

Model-internal measures say how much truncation changes the model's choice, not whether the change matters to users, and are labeled so in the report.

**Code.** `rank/cascade.py` (D056). **Tests.** With $N$ equal to the catalog size, the cascade returns the catalog reference's top-1; survival is monotone non-decreasing in $N$.

---

## 5. Fine ranker

**Model.** Two LightGBM binary classifiers, one for clicks and one for long views, on the same features. The fused score is

$$
r(u, i) = w\, \hat p_{\text{click}}(u, i) + (1 - w)\, \hat p_{\text{long}}(u, i), \qquad w = 0.5 .
$$

$w = 0.5$ is fixed in advance; $w \in \{0, 0.25, 0.5, 0.75, 1\}$ is reported as a secondary sensitivity table.

**Features**, all computed from the window before the cutoff:

- *User*: the categorical fields of Section 3, log of 1 + follow, fan and friend counts, `register_days`, impressions per day, smoothed click and long-view rates (D056).
- *Item*: `video_duration`, `upload_type`, `video_type`, first tag, days since upload at the cutoff, impressions per day, smoothed click and long-view rates.
- *User × item*: the user's smoothed click rate on the item's author and on its first tag, the two-tower score at the same cutoff, the item's popularity rank.
- Smoothed rate: $(k + \alpha \mu) / (n + \alpha)$ with $k$ successes in $n$ impressions, $\mu$ the window-wide rate and $\alpha = 20$.
- Not used: raw author id (high cardinality; the affinity features carry it) and `video_features_statistic` (Section 1).

**Fixed hyperparameters:** `num_leaves` 63, learning rate 0.05, `min_data_in_leaf` 100, feature and bagging fractions 0.8, at most 2,000 rounds with early stopping after 50 rounds without improvement in validation log loss, seed 0, 4 threads.

**Training bias.** The fine ranker trains on impressions the production recommender chose, with unknown propensities, so no correction is applied in training. Evaluation on random exposures is what keeps the reported numbers unbiased.

**Code.** `rank/features.py`, `rank/fine.py`. **Tests.**

- Features at cutoff $c$ do not change when rows with $\text{date} \ge c$ are added.
- The smoothed rate equals $\mu$ when $n = 0$.
- Training is deterministic for a fixed seed.

---

## 6. Ranking metrics

All are computed on the evaluation half of the random exposures and are Tier A.

- **Recall@K.** $P_u$ is the set of distinct items user $u$ clicked among their random exposures and $R_u^K$ the top $K$ retrieved:
  $$\text{Recall@}K = \frac{1}{|U|} \sum_{u \in U} \frac{|P_u \cap R_u^K|}{|P_u|}, \qquad U = \{u : |P_u| \ge 1\}.$$
  $K \in \{50, 100, 200, 500\}$. The interpretation rests on how the random exposures were collected. Gao et al. (2022, Section 3.2) describe it as: "Each time when the recommender system recommends a video list to a user, we decide whether to insert a random item with a fixed probability. If the answer is yes, then we intervene in the recommendation list by randomly selecting one video from this list and replacing it with a random item uniformly sampled from the 7,583 items." The insertion probability is not disclosed, and users with fewer than 10 random exposures were removed. Taking that description as given, the item in each random exposure is independent of the user's preferences, so $P_u$ is a uniform sample of the catalog items $u$ would click if shown and the ratio estimates the share of those items in the top $K$. Two departures would bias it: a pool that differs from the 7,583 catalog items (the Pure random log covers 7,537 of them, consistent with uniform sampling over about 1.18M exposures), and clicks that depend on the item's position in the feed, which the dataset does not record for random exposures. Both are reported as limitations; neither can be checked further with the released data.
- **GAUC.** The ROC AUC of the score over each user's random exposures, averaged with equal weight over users who have both clicked and non-clicked impressions. The impression-weighted average is reported alongside. Computed for the click label and the long-view label.
- **NDCG@5.** Each user's random exposures ranked by score, gain 1 for a click, ideal DCG from the best ordering; averaged over users with at least one click. Tied scores keep the order of the log, which matters mainly for the popularity scorer, whose unseen items share one rank.

**Reference.** Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., He, X. (2022). KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. *CIKM 2022*. https://arxiv.org/abs/2208.08696

**Primary comparisons** (Holm correction across the two):

| Id | Comparison | Metric |
|---|---|---|
| C1 | Two-tower against popularity | Recall@200 |
| C2 | Fused fine ranker against two-tower score alone | GAUC (click) |

Everything else (other $K$, long-view GAUC, NDCG@5, the $w$ table, truncation) is secondary and reported without significance claims.

**Code.** `rank/metrics.py`. **Tests.** Each metric against hand-computed examples, including ties (AUC counts a tie as 0.5) and users with a single class (excluded from GAUC).

---

## 7. Uplift data and contrasts

**Problem.** Show that the uplift methods rank people by their response to a treatment, using trials where the treatment was randomized (Tier A). These trials are e-mail and advertising, not push; they validate the methods, not the simulator's gate (D013).

| | Hillstrom | Criteo Uplift v2.1 |
|---|---|---|
| Treatment $t = 1$ | women's e-mail | `treatment == 1` (assigned to see ads; intention to treat, `exposure` is not used) |
| Control $t = 0$ | no e-mail (the men's e-mail arm is dropped, D004) | `treatment == 0` |
| Primary outcome $y$ | `visit` | `visit` |
| Secondary outcome | `conversion` | `conversion` |
| Features $x$ | `recency`, `history`, `mens`, `womens`, `newbie`, one-hot `zip_code` and `channel`; `history_segment` is dropped as a binned copy of `history` | `f0` to `f11` |
| Training rows | a 50% split stratified on $(t, y)$, `random_state = 0` (about 21,350 rows) | the seeded 10% `train` sample (1,398,727 rows) |
| Test rows | the other 50% (about 21,350) | every row outside the sample (12,580,865) |
| Treated share | 0.50 | 0.85 |

`visit` is primary on both: conversions are rare (0.57% and 0.88% by arm on Hillstrom, 0.29% on Criteo), and Diemert et al. (2021) recommend modeling Criteo uplift on visits for that reason. Rows are customers (Hillstrom) and users (Criteo); each appears once. The propensity is the treated share of the training rows, $\hat e$, a constant because assignment was randomized.

---

## 8. Uplift learners

Each learner outputs $\hat\tau(x)$, an estimate of $\tau(x) = P(y = 1 \mid x, t = 1) - P(y = 1 \mid x, t = 0)$, fitted on the training rows only.

**Base model.** Every LightGBM model uses the same fixed hyperparameters, with no tuning and no early stopping: 200 rounds, learning rate 0.05, `num_leaves` 31, `min_data_in_leaf` 100, feature and bagging fractions 0.8 (bagging every round), seed 0, 4 threads. Classifiers use the binary objective, regressors L2.

- **S-learner.** One classifier $\hat\mu(x, t)$ with $t$ as a feature; $\hat\tau(x) = \hat\mu(x, 1) - \hat\mu(x, 0)$. Trees may never split on $t$, which gives $\hat\tau = 0$ for many rows; this is a known weakness and is kept.
- **T-learner.** Classifiers $\hat\mu_1$ on treated rows and $\hat\mu_0$ on control rows; $\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$.
- **X-learner** (Künzel et al. 2019). Stage 1 is the T-learner. Imputed effects: $D_i^1 = y_i - \hat\mu_0(x_i)$ for treated rows and $D_i^0 = \hat\mu_1(x_i) - y_i$ for control rows. Stage 2 fits regressors $\hat\tau_1$ on $(x_i, D_i^1)$ and $\hat\tau_0$ on $(x_i, D_i^0)$. Combination: $\hat\tau(x) = \hat e\, \hat\tau_0(x) + (1 - \hat e)\, \hat\tau_1(x)$, the weighting the paper recommends with $g(x)$ equal to the propensity.
- **Causal forest.** EconML `CausalForestDML` (Athey, Tibshirani and Wager 2019; Chernozhukov et al. 2018): outcome nuisance a LightGBM regressor as above, treatment nuisance `DummyClassifier(strategy="prior")` (the known constant propensity), `discrete_treatment=True`, 2-fold cross-fitting, 100 trees, other forest settings at EconML 0.17 defaults (honest, `min_samples_leaf` 5, `max_samples` 0.45), `random_state` 0, 4 jobs. On Criteo it is fitted on 200,000 rows drawn from the training sample with `numpy.random.default_rng(0)`, since a fit on all 1.4M rows does not fit the time budget (SPEC Section 10: about 1 minute for 200k rows); the meta-learners use all 1.4M, and the report states this asymmetry.
- **Response model** (baseline, not an uplift method). $\hat\mu_1(x)$ from the T-learner: the predicted outcome if treated, which is what targeting by propensity to respond uses.
- **Random** (baseline). Scores drawn from $U(0, 1)$ with `numpy.random.default_rng(0)`.

**References.**
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS* 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116
- Athey, S., Tibshirani, J., Wager, S. (2019). Generalized random forests. *Annals of Statistics* 47(2), 1148–1178. https://doi.org/10.1214/18-AOS1709
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal* 21(1), C1–C68. https://doi.org/10.1111/ectj.12097

**Code.** `uplift/learners.py`, `uplift/forest.py`. **Tests** (synthetic data with a known $\tau(x)$, no real data):

- Each learner's $\hat\tau$ correlates with the true $\tau$ above a stated threshold, and its mean is within a stated tolerance of the true average effect.
- With $\tau = 0$ everywhere, the T-learner's Qini coefficient has a 95% bootstrap interval that contains 0.
- Fitting is deterministic for a fixed seed.

---

## 9. Uplift curves and metrics

**Curves.** Sort the $n$ test rows by score, highest first. For the top $k$ rows let $N_T(k), N_C(k)$ be the treated and control counts and $Y_T(k), Y_C(k)$ their responders.

- *Qini curve* (Radcliffe 2007): $Q(k) = Y_T(k) - Y_C(k)\, N_T(k) / N_C(k)$, incremental responders in units of the treated group.
- *Uplift curve* (Devriendt et al. 2022): $U(k) = \left( Y_T(k)/N_T(k) - Y_C(k)/N_C(k) \right) k$, incremental responders if the top $k$ were treated and nobody else.
- A ratio with an empty denominator is 0. Rows with tied scores enter together: the curve is evaluated only at the end of each tied group and joined by straight lines, as scikit-uplift 0.5.1 does for both curves. For the counts this is the expected path under a random order within the group; for the ratios inside $Q$ and $U$ it is an approximation.
- Both are *joint* curves: treated and control rows are ranked together. Diemert et al. (2021) report the *separate, relative* AUUC (each arm ranked on its own, response rates differenced), which Devriendt et al. (2022) found more robust to unequal arm sizes; Criteo numbers here are therefore not directly comparable with theirs, which also use a different split.

**Areas.** With curve points $(k_j, C_j)$, $k_0 = C_0 = 0$, the area is the trapezoid sum $A = \sum_j (k_j - k_{j-1})(C_j + C_{j-1})/2$, and the random-targeting line from $(0, 0)$ to $(n, C(n))$ has area $n\, C(n)/2$. The *perfect* curve orders rows by an oracle score that uses the outcomes: for Qini, $y t - y (1 - t)$ (treated responders first, control responders last, negative effects included); for the uplift curve, the scikit-uplift construction. Then

$$
\text{Qini coefficient} = \frac{A_Q - n\,Q(n)/2}{A_Q^\ast - n\,Q(n)/2}, \qquad
\text{AUUC} = \frac{A_U - n\,U(n)/2}{A_U^\ast - n\,U(n)/2},
$$

which equal scikit-uplift's `qini_auc_score(negative_effect=True)` and `uplift_auc_score`. Both are 0 for random targeting in expectation, and the denominator is the same for every method on the same test rows, so paired differences rank methods exactly as the unnormalized areas do.

**Binned curves on Criteo.** With 12.6M test rows, the model's curve is evaluated at 200 score quantiles: rows are grouped into bins of about equal size by score, all rows with the same score in the same bin, and the bins are treated as tied groups. A curve point every 0.5% of the test set changes the area by much less than its sampling error. The perfect curve uses only the outcome and treatment counts and stays exact. Hillstrom curves are exact. With 85% of rows treated, a bin can hold few control rows; `results/uplift.json` reports, per method, the minimum and the 5th, 50th and 95th percentiles of control rows per bin and the number of bins with none (where a control rate would be 0/0 and the bin's control responders count as 0).

**Bootstrap.** Rows are resampled (D012: 1,000 resamples, 95% percentile intervals, seed 0), and every statistic is recomputed on each resample, including the perfect-curve areas in the denominators, which depend on the resampled counts of the four $(t, y)$ classes. Each metric depends on the data only through the counts in cells (score group, $t$, $y$), where a group is a distinct score on Hillstrom and a score bin on Criteo, so drawing $n$ rows with replacement is the same as drawing the cell counts from a multinomial with the observed cell shares; this is how both datasets are resampled. For a paired comparison of methods $a$ and $b$ the cells are (group under $a$, group under $b$, $t$, $y$). Groups are fixed at their full-sample values: on Hillstrom this is exact, since a group is a function of the score; on Criteo the bin edges are not recomputed per resample, which at 12.6M rows moves the result by far less than the sampling error. The intervals cover test-set sampling only: each model is fitted once on a fixed training set with a fixed seed. A test checks on small synthetic data that the cell bootstrap and row resampling give the same intervals within Monte Carlo error.

**Cross-check.** Hillstrom Qini coefficients and AUUC for every method are compared with scikit-uplift 0.5.1 (development dependency only) on the same scores; the largest absolute difference is reported in `results/uplift.json` and must be below $10^{-9}$ (D056). The unit tests run the same comparison on synthetic data with ties.

**References.**
- Radcliffe, N. J. (2007). Using control groups to target on predicted lift: Building and assessing uplift models. *Direct Marketing Analytics Journal*, 14–21. Direct Marketing Association Analytics Council.
- Devriendt, F., Van Belle, J., Guns, T., Verbeke, W. (2022). Learning to rank for uplift modeling. *IEEE Transactions on Knowledge and Data Engineering* 34(10), 4888–4904. https://doi.org/10.1109/TKDE.2020.3048510
- Diemert, E., Betlei, A., Renaudin, C., Amini, M.-R., Gregoir, T., Rahier, T. (2021). A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling. arXiv:2111.10106 (not peer reviewed).

**Code.** `uplift/curves.py`. **Tests.** Curves and both coefficients against hand-computed examples, against scikit-uplift on synthetic data with ties, and binned against exact on data with few distinct scores (equal when every bin is one tied group).

---

## 10. Targeting under a budget and primary comparisons

**Budget.** A budget $b$ is the share of the test population that may be treated; the top $bn$ rows by score are treated and the rest are not. The value of a method at $b$ is $U(bn)$, linear between curve points, so a tied group or bin that straddles the boundary is treated in proportion. It is reported as incremental responders per 1,000 test rows and as the share of the treat-everyone effect, $U(bn) / U(n)$; the share is left out when the interval of $U(n)$ contains 0, since it is then undefined. The interval of $U(n)$ comes from one bootstrap over the $(t, y)$ cells per dataset and outcome, shared by every method, since $U(n)$ does not depend on the scores (D055). Budgets: $b \in \{0.1, 0.2, 0.3, 0.5\}$. Methods: the four learners, the response model and random. This table is secondary and carries no significance claims.

**Primary comparisons**, on the primary outcome (`visit`) and the Qini coefficient, all with the T-learner as the uplift method, fixed before any model is fitted on these data. Holm correction across the four at family-wise level 0.05:

| Id | Dataset | Comparison | Test |
|---|---|---|---|
| U1 | Hillstrom | T-learner Qini coefficient against 0 (random targeting) | bootstrap of the coefficient |
| U2 | Hillstrom | T-learner against response model | paired bootstrap of the difference |
| U3 | Criteo | T-learner Qini coefficient against 0 | bootstrap of the coefficient |
| U4 | Criteo | T-learner against response model | paired bootstrap of the difference |

The p-values follow the Conventions (twice the smaller tail share at zero). U1 and U3 test the T-learner's own coefficient against 0, its expected value under random targeting; they do not use the random baseline's scores, which are a single draw and whose coefficient carries its own sampling noise. A claim that uplift ranking beats random or response ranking on a dataset needs the comparison rejected under Holm with a positive difference. Everything else (the other learners, AUUC, `conversion`, the budget table) is secondary and reported without significance claims.

**Results** go to `results/uplift.json`: setup (row counts, treated shares, test-set sizes), every metric with its interval and tier A, the four comparisons with p-values and Holm decisions, the budget table, Qini curves at 101 points with 95% bands for the report, the scikit-uplift cross-check and fit times. One diagnostic was added after U4 came out (D022) and is labeled post hoc: on Criteo visits, the treated and control response rates and their difference per score decile of the T-learner and of the response model.

**Code.** `uplift/evaluate.py` (`python -m push_lab.uplift.evaluate`). Criteo is read by column and scored in one pass over 1M-row batches (D015); treatment and outcomes are read from the same batches as the features.

---

## 11. Push simulator

**Problem.** No public data links push sends to later behavior (SPEC Section 12), so delivery policies are compared in a simulator. Every component below is an assumption with a stated source and a swept range that includes no effect; any result that depends on the simulator is Tier C and is reported only with the sweep (M5). M4 builds the simulator and checks its fitted inputs; it reports no policy outcomes.

**Population and time.** All 27,285 KuaiRand-Pure users, 28 simulated days, four send slots a day at 08:00, 12:00, 17:00 and 21:00 (local time, UTC+8). Simulated days are not calendar days; every user starts with no push history. A 5,000-user sample drawn with `numpy.random.default_rng(0)` serves quick runs.

**Decisions.** At each (user, day, slot) a policy either skips or sends one item with one copy template $k$. Users who opted out receive nothing; a send to them is dropped and logged as not delivered.

### 11.1 Inputs fitted on KuaiRand

- **Response model** $p_0(u, i)$: the probability that user $u$ clicks item $i$ when it is shown at random in the feed. A LightGBM click classifier with the fine ranker's features at cutoff $c_2$ (Section 5) and its fixed hyperparameters, fitted on the random exposures of the *simulator half* (Section 1, tab 1 only). The simulator half's users are split 80/20 by user with `numpy.random.default_rng(0)`; the 20% drive early stopping and the reported validation (log loss, AUC, GAUC, and a reliability table in 10 bins of predicted probability). No row of the evaluation half is used. These checks are Tier A because they are measured on random exposures of users held out from fitting; the same users set the stopping round, which makes them slightly optimistic (D058).
- **Candidate items.** For each user, the top 112 items of the M2 cascade at $c_2$ (pool, coarse top 200 by two-tower score, fused fine score), and separately the top 112 by two-tower score alone; 112 is one item per send slot of the horizon, so a list never runs out. The second list exists because C2 found the two-tower score ranks random exposures better (D020). $p_0$ is computed for the union. A policy pushes the highest-ranked item of its chosen list that it has not pushed to that user before.
- **Organic visit rate** $v_u$: the share of the 17 days from 04-22 to 05-08 on which the user has any tab-1 row in the standard or random log, smoothed as $(n_u + 2\bar v)/(17 + 2)$ with $\bar v$ the population mean. Random exposures are inserted while a user browses, so they mark activity as well as standard impressions do.
- **Activity profiles.** A push sent at slot $s$ is assumed to be opened, if at all, within the following three hours: windows $[8, 11)$, $[12, 15)$, $[17, 20)$ and $[21, 24)$. The hour comes from `time_ms` converted to UTC+8; `hourmin` is a coarse bucket that differs from `time_ms` by up to an hour (KuaiRand documents it as "HHSS"; the values are whole hours). The time zone is an assumption: KuaiRand does not state it. The `date` field equals the calendar date of `time_ms` in UTC+8 for 99.3% of rows, against 96.0% for UTC+7 and 95.6% for UTC+9, and activity is lowest at 04:00 in UTC+8 (D024); a test checks the date agreement. The *simulator profile* counts each user's tab-1 rows in each window from 04-22 to 05-08 (standard and random logs); the *policy profile* uses standard logs from 04-09 to 04-21 (D014). Shares are smoothed toward the population shares $\pi_s$: $a_{u,s} = (n_{u,s} + 10\,\pi_s) / (n_u + 10)$.

### 11.2 Open probability

A push of item $i$ with template $k$ delivered to user $u$ at slot $s$ of day $d$ is opened with probability

$$
p = \min\!\Big(1,\; r \cdot p_0(u, i) \cdot m_{u,s} \cdot c_k \cdot e^{-\phi F^-_{u}} \cdot \big(1 - \gamma\, 0.5^{\Delta_{u,k}/h}\big)\Big)
$$

- **Base rate** $r$: push opens are far rarer than in-feed clicks on random items (17.7%, D020). Airship's 2024 benchmarks (industry, not peer reviewed) put median direct open rates at 3.4% (Android) and 3.1% (iOS), 10th to 90th percentile about 0.7% to 10.7%; Kuaishou push CTRs shown in Bie et al. (2026) are about 0.8% to 1.2%. The items the policies push have a mean $p_0$ of about 0.28 (D024; not in D024, measured in D054: 0.277), so $r \in [0.03, 0.2]$ gives open rates of about 0.8% to 5.6% before the other factors, and the reference $r = 0.06$ about 1.7%.
- **Send time** $m_{u,s} = a_{u,s}^\eta / \tfrac14 \sum_{s'} a_{u,s'}^\eta$, so sending at a uniformly random slot gives multiplier 1 on average. $\eta \ge 0$ is set by the swept *timing lift* $L = \mathbb E_u[\max_s m_{u,s}] - 1$, the gain from always sending at each user's true best slot, solved by bisection on the users being simulated, so the fit, evaluation and diagnostic users of one setting get slightly different $\eta$ (D058). Evidence: personalized timing raised CTR by 1.22% at Kuaishou (Ji et al. 2024) and engagement by a risk ratio of 1.025 to 1.118 by time and day in a micro-randomized trial (Bidargaddi et al. 2018); vendor claims reach about 23% (not peer reviewed). $L \in [0, 0.25]$.
- **Copy effect** $c_k = \exp(\sigma z_k - \sigma^2/2)$ with $z_k = \sqrt{0.5}\, z_{\text{family}(k)} + \sqrt{0.5}\, z'_k$ and standard normal $z$ drawn once per world from a separate seed, so $\mathbb E[c_k] = 1$. Duolingo template scores imply a spread of about 2% (Yancey and Settles 2020, Fig. 3); LLM-written copy raised Kuaishou CTR by 4% to 14% (Bie et al. 2026); a personalized e-mail subject raised opens by 20% (Sahni et al. 2018). $\sigma \in [0, 0.2]$. The effect is drawn at random for each world and is not derived from the text (SPEC Section 7): the simulator cannot tell which copy is well written, and the template bandit (M6) is tested only on whether it finds the templates with the largest drawn effects. The committed copy has $K = 60$ templates in 6 families (13.1).
- **Volume fatigue** $e^{-\phi F^-_u}$, where $F^-_u = \sum_j 0.5^{(t - t_j)/h_f}$ sums over the user's earlier deliveries $j$, with times in days (slot hours as fractions of a day). $\phi = \epsilon_f / F_{\text{ref}}$, where $F_{\text{ref}} = 1/(2^{1/h_f} - 1)$ is $F^-$ just before a push under one push a day, so $\epsilon_f$ is the elasticity of the open rate with respect to recent volume at that volume. Online tests found volume elasticities of the open rate of about $-0.7$ to $-1.3$: Twitter, +16.8% sends and −12.0% open rate (O'Brien et al. 2022, Table 1, preprint); LinkedIn, +7.7% volume and −7.8% CTR (Prabhakar et al. 2022, Table 1), and −3.5% volume and +4.5% CTR (Yuan et al. 2022). These also include weaker marginal sends, so they bound pure fatigue from above. $\epsilon_f \in [0, 0.8]$; $h_f \in [3, 14]$ days is an assumption, as no source reports a memory length.
- **Template novelty** $1 - \gamma\, 0.5^{\Delta_{u,k}/h}$, where $\Delta_{u,k}$ is the number of days since template $k$ was last delivered to $u$ (the term is 1 if never). This is the recency penalty of Yancey and Settles (2020, Eq. 3), applied as a relative change, with their production values $\gamma = 0.017$ and $h = 15$ days as the reference. $\gamma \in [0, 0.05]$, $h \in [7, 30]$ days.

### 11.3 Opt-out

Each delivered push makes the user opt out with probability $\lambda = \kappa_o \cdot F^+_u / (F_{\text{ref}} + 1)$, where $F^+_u$ includes the current push, so $\lambda = \kappa_o$ under one push a day and grows with recent volume. Opted-out users receive no further pushes; their organic visits continue. Evidence: at Tencent, the share of new users with push enabled fell from 97% to 91% over 14 days at 15 pushes a day, at most about 0.03% per push, with part of the fall seen at every volume (Zhang et al. 2023, Fig. 3); switch-off rates did not change detectably in Twitter's and Kuaishou's tests. $\kappa_o \in [0, 0.001]$, with $0.005$ as a stress value outside the evidence.

### 11.4 Visits

User $u$ visits on day $d$ if an organic draw succeeds or any push that day is opened (an open is an app visit). The organic probability is

$$
v_{u,d} = \min\!\Big(1,\; v_u\, e^{-\delta F_{u,d}} + \kappa_r R_{u,d}\Big), \qquad R_{u,d} = \sum_{j:\, d_j < d} 0.5^{(d - d_j - 1)/h_r},
$$

with $F_{u,d}$ the decayed volume at the start of the day and the sum over the days $d_j$ of earlier opens.

- **Return effect** $\kappa_r R$: an opened push raises the chance of visiting on following days, decaying with half-life $h_r$. KuaiSim raises a user's return probability with immediate feedback (Zhao et al. 2023, Algorithm 1) and Accordion adds an exponentially decaying intensity after each interaction (McInerney et al. 2021, Eq. 1–2); neither reports fitted magnitudes for push. A micro-randomized trial found a risk ratio of 1.039 for engagement within 24 hours of a send (Bidargaddi et al. 2018). $\kappa_r \in [0, 0.1]$, $h_r \in [0.5, 3]$ days, both assumptions.
- **Annoyance** $e^{-\delta F}$ with $\delta = \epsilon_v / F_{\text{ref}}$: recent volume lowers organic visits, with elasticity $-\epsilon_v$ at one push a day. LinkedIn saw WAU fall 0.69% with 7.7% more volume (Prabhakar et al. 2022), while Twitter saw DAU rise 0.41% with 16.8% more sends (O'Brien et al. 2022); both net effects include visits caused by pushes. $\epsilon_v \in [0, 0.1]$. The elasticity is exact just before a push; at the start of the day, when the annoyance term is evaluated, the decayed volume is a little above $F_{\text{ref}}$ and the elasticity 3% to 9% larger. SPEC Section 6 does not list this component; it is added because without it no volume can lower activity below the organic level (D023).
- **Plausibility band.** Reported program-level DAU lifts from notification changes are 0.07% to 0.5% (Duolingo, Twitter, LinkedIn, Kuaishou). M5 reports the share of sweep settings whose simulated lift of a once-a-day policy over no pushes falls in, below and above that band; the band is a check, not a target.

### 11.5 Parameters

| Parameter | Symbol | Range | Reference |
|---|---|---|---|
| Base-rate factor | $r$ | 0.03 to 0.2 | 0.06 |
| Timing lift | $L$ | 0 to 0.25 | 0.10 |
| Copy spread | $\sigma$ | 0 to 0.2 | 0.05 |
| Fatigue elasticity, half-life (days) | $\epsilon_f$, $h_f$ | 0 to 0.8, 3 to 14 | 0.4, 7 |
| Novelty penalty, half-life (days) | $\gamma$, $h$ | 0 to 0.05, 7 to 30 | 0.017, 15 |
| Opt-out per push at one a day | $\kappa_o$ | 0 to 0.001 (stress 0.005) | 0.0003 |
| Return boost, half-life (days) | $\kappa_r$, $h_r$ | 0 to 0.1, 0.5 to 3 | 0.03, 1 |
| Annoyance elasticity | $\epsilon_v$ | 0 to 0.1 | 0.05 |

The reference setting serves tests, the exploration logs used to build M5 and the examples in the report; the sweep design (Latin hypercube over these ranges, with profile misspecification) is pre-registered in M5.

### 11.6 Randomness and implementation

The simulator is vectorized over users and steps through 28 × 4 slots. Two seeds define a run: a *world seed* for the copy effects and a *run seed* for uniform draws for every (user, day, slot) open and opt-out event and every (user, day) organic visit. The draws are generated before the run and shared by every policy run with the same seed, so policies are compared on common random numbers and their differences are paired. Policies draw their own coins and templates from streams keyed by a purpose, the seed, the day and the slot (D032).

### 11.7 Exploration policy

The logging policy for the uplift gate (D013) and off-policy evaluation (M7) is epsilon-greedy around a default of one push a day at the user's best slot under the *policy* profile:

$$
P(\text{send} \mid u, s) = (1 - \varepsilon)\, \mathbb 1[s = \hat s_u] + \varepsilon\, q, \qquad P(k \mid \text{send}) = 1/K,
$$

with $\varepsilon = 0.3$ and $q = 1/4$, so the send probability is 0.775 at $\hat s_u$ and 0.075 at the other slots, and the expected volume is one push a day. The smallest propensity of a send, $0.075/K$, is recorded with $K$ in `results/simulator.json` for off-policy evaluation (M7). Items follow the cascade list. Each logged decision records the user, day, slot, action, template, item, the propensity of the action taken, whether the push was delivered and opened, and whether the user opted out; daily visits are logged per user.

**References.**
- Yancey, K. P., Settles, B. (2020). A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications. *KDD 2020*. https://doi.org/10.1145/3394486.3403351
- O'Brien, C., Wu, H., Zhai, S., Guo, D., Shi, W., Hunt, J. J. (2022). Should I send this notification? Optimizing push notifications decision making by modeling the future. arXiv:2202.08812 (not peer reviewed).
- Prabhakar, P., Yuan, Y., Yang, G., Sun, W., Muralidharan, A. (2022). Multi-objective Optimization of Notifications Using Offline Reinforcement Learning. *KDD 2022*. https://doi.org/10.1145/3534678.3539193
- Yuan, Y., Muralidharan, A., Nandy, P., Cheng, M., Prabhakar, P. (2022). Offline Reinforcement Learning for Mobile Notifications. *CIKM 2022*. https://doi.org/10.1145/3511808.3557083
- Zhang, Y., Zhao, M., Li, C., Tou, W., Chen, H., Niu, D., Yin, C., He, Y., Guo, F. (2023). Online Volume Optimization for Notifications via Long Short-Term Value Modeling. *PAKDD 2023*. https://doi.org/10.1007/978-3-031-33380-4_2
- Ji, H., Yang, H., Li, L., Zhang, S., Zhang, C., Li, X., Ou, W. (2024). TIM: Temporal Interaction Model in Notification System. *ICMR 2024*. https://doi.org/10.1145/3652583.3657614
- Bie, S., Cao, J., Luo, Z., Zou, Y., Liang, L., Zhang, L., Chen, L., Liu, Z., Li, X., et al. (2026). PushGen: Push Notifications Generation with LLM. *WSDM 2026*. https://doi.org/10.1145/3773966.3779373
- Bidargaddi, N., Almirall, D., Murphy, S., Nahum-Shani, I., Kovalcik, M., Pituch, T., Maaieh, H., Strecher, V. (2018). To Prompt or Not to Prompt? A Microrandomized Trial of Time-Varying Push Notifications to Increase Proximal Engagement With a Mobile Health App. *JMIR mHealth and uHealth* 6(11), e10123. https://doi.org/10.2196/10123
- Sahni, N. S., Wheeler, S. C., Chintagunta, P. (2018). Personalization in Email Marketing: The Role of Noninformative Advertising Content. *Marketing Science* 37(2), 236–258. https://doi.org/10.1287/mksc.2017.1066
- Zhao, K., Liu, S., Cai, Q., Zhao, X., Liu, Z., Zheng, D., Jiang, P., Gai, K. (2023). KuaiSim: A Comprehensive Simulator for Recommender Systems. *NeurIPS 2023 Datasets and Benchmarks*. https://doi.org/10.52202/075280-1945
- McInerney, J., Elahi, E., Basilico, J., Raimond, Y., Jebara, T. (2021). Accordion: A Trainable Simulator for Long-Term Interactive Systems. *RecSys 2021*. https://doi.org/10.1145/3460231.3474259
- Airship (2025). Mobile Engagement Benchmarks (industry report, not peer reviewed).

**Code.** `sim/inputs.py` (response model, candidates, visit rates, profiles), `sim/world.py` (parameters and derived constants), `sim/env.py` (environment), `sim/explore.py` (exploration policy), `sim/build.py` (`python -m push_lab.sim.build`, writes `results/simulator.json` with the input checks; per-user inputs are cached under `data/processed/sim/` and not committed).

**Tests** (SPEC Section 11, on synthetic inputs):
- With no sends, fatigue stays 0, nobody opts out and visits equal the organic draws.
- With $\epsilon_f = \epsilon_v = \gamma = \kappa_o = 0$ and the item fixed per (user, day, slot), adding sends never lowers opens or visits on the same random numbers. (With the default item rule an extra send moves later pushes down the user's list, which can lower $p_0$, so the invariant is tested with fixed items.)
- With $\sigma = \gamma = 0$, the choice of template does not change any outcome.
- Fixed seeds give identical outputs; policies run with the same run seed see the same draws.
- $\eta$ reproduces the target timing lift; $\phi$, $\delta$ and $\lambda$ give the stated elasticities and hazard at one push a day.
- The exploration policy's logged propensities match the formula, and its empirical send and template frequencies match them within sampling error.
- Response-model features use no data on or after $c_2$ (Section 5 tests).
- On the real data, the `date` field agrees with the UTC+8 date of `time_ms` more often than with any other offset, and hourly activity in UTC+8 is lowest between 00:00 and 06:00.

---

## 12. Delivery policies and their comparison

**Problem.** Decide for each user and day whether to push and at which slot under a global volume budget, and test whether an uplift gate keeps more users active than sending to everyone or sending by predicted open rate. Every result here is Tier C and is reported only with the sweep (12.6).

### 12.1 What a policy may read

| Input | Source | Never |
|---|---|---|
| Policy profile, shifted and noised (12.5) | standard logs 04-09 to 04-21, hourly counts per user | `profile_sim` |
| Policy visit rate $\hat v_u$ | share of the 13 days 04-09 to 04-21 with a tab-1 standard-log row, smoothed as in 11.1 | `visit_rate` |
| Score of the next item | `lists["cascade"].score[u, j]`, $j$ the number of the policy's own deliveries to $u$ | `p0` |
| Own history | sends, deliveries (send to a user not opted out), templates | |
| Feedback | opens and daily visits of days before $d$; opted-out flags | outcomes of day $d$ or later |

The environment passes feedback in `Observation` (D027). Every policy plans day $d$ at its first slot from data before $d$ and sends at most one push per user per day. Templates are uniform over $K$ and drawn from a stream keyed by (run seed, day, slot), so every policy gives a user the same template at the same (day, slot) and template effects stay paired.

**Item list** (D026). All policies push from the cascade list (11.1), as the exploration logs do (11.7). The two-tower list appears only in the secondary comparison L1. The response model gives 6% of its gain to the two-tower score (D024), which orders the two-tower list, so L1 leans toward that list by construction; the delivery comparisons hold the list fixed.

### 12.2 Policies

| Id | Policy | Reads |
|---|---|---|
| N | No pushes | nothing |
| R | Each slot independently with probability 1/4 (one a day on average, no daily cap); coins from their own stream | nothing |
| F | One push a day at the slot with the largest share of the setting's perturbed policy profile averaged over all users (21:00 at $\Delta = \xi = 0$) | mean of the perturbed profile |
| T | One push a day at the user's best slot under the perturbed policy profile | policy profile |
| T½ | T's slot, for a seeded random half of users each day (secondary, separates volume from targeting in H3) | policy profile |
| S | Response threshold: user-day score $\max_s \hat p(x_{u,d,s})$ | 12.1 |
| G | Uplift gate, T-learner: user-day score $\hat\tau(x_{u,d,s^*})$ | 12.1 |
| G2 | Uplift gate, decomposed: user-day score $\hat p(x_{u,d,s^*})\,(1 - \hat\mu_0(x_{u,d,s^*}))$ | 12.1 |

S, G and G2 all send at $s^*_{u,d} = \arg\max_s \hat p(x_{u,d,s})$: the share of organic visits a push can add is the same at every slot of a day, so the slot is chosen by open probability and the three differ only in which user-days they send. G2 assumes an open and an organic visit are independent given $x$, which holds in the simulator by construction; it is labeled as using that assumption.

### 12.3 Gate and response models

**Logs.** For each setting, the exploration policy (11.7, with the setting's perturbed profile) runs over the fit users (12.6) with its own run seed. One row per (user, day, slot) at which the user had not opted out; treatment $t = 1$ if a push was delivered; outcome $y = 1$ if the user visited on day $d$. The send probability depends only on whether $s$ is the user's best slot, which is a feature, so treatment is randomized given $x$.

**Features** $x_{u,d,s}$: slot; perturbed profile share at $s$; whether $s$ is the best slot; $\hat v_u$; score of the next item; day $d$; share of earlier simulated days with a visit; visit on $d - 1$; opens so far; days since the last open and since the last delivery (capped at 28); decayed own delivered volume with a fixed half-life of 7 days (the policy does not know $h_f$).

**Models.** LightGBM with the base model of Section 8 (200 rounds; one thread per process):
- $\hat\mu_1, \hat\mu_0$: same-day visit on treated and on control rows; $\hat\tau = \hat\mu_1 - \hat\mu_0$ (the T-learner of U1 to U4).
- $\hat p$: open on treated rows.

The scores target the same-day visit; return, fatigue, annoyance and opt-out act on later days and enter only through the features and the budget. This myopia is a stated limitation. Two biases act on S, G and G2 alike: the logging policy's other pushes on the same day, and histories the logs never contain (for example ten days without a delivery).

**Score diagnostics** (secondary): on 2,000 users drawn from the 12,285 users outside the fit and evaluation sets, an exploration run with the setting's parameters gives the true same-day uplift of each (user, day, slot), $p_{u,d,s}\,(1 - v_{u,d})$ with $p$ from 11.2 and $v$ from 11.4; the report gives the Spearman correlation of each score with it and the share of user-days on which $s^*$ is the true best slot. The true best slot includes the effect of the template drawn for each slot, which no policy can predict, so this share understates slot accuracy when the copy spread is large.

### 12.4 Budget and the Lagrange multiplier

**Allocator.** Units $u$ have marginal gains $g_{u,1}, g_{u,2}, \dots$ and the budget is $B$ sends. For $\lambda \ge 0$ let $x^-_u(\lambda)$ and $x^+_u(\lambda)$ be the smallest and the largest maximizer of $\sum_{j \le x} (g_{u,j} - \lambda)$; both fall as $\lambda$ grows. The allocator takes the smallest $\lambda \ge 0$ with $\sum_u x^-_u(\lambda) \le B$ (it is 0 or the average gain of a block of consecutive sends), sends $x^-_u(\lambda)$, and, if $\lambda > 0$, spends leftover budget raising units in index order to $x^+_u(\lambda)$ when the whole step fits. With non-increasing gains the allocated total gain equals the brute-force optimum (totals are compared, since ties allow several optimal sets); a test documents an instance with non-monotone gains where it falls short (D009).

**Primary use: per-day budget** (D028). Units are the eligible users of day $d$ (not opted out), each with one gain, its user-day score. Each day S, G and G2 send to the $\lfloor b\,n \rfloor$ eligible users with the highest scores (ties by user index), with no floor at zero, so the three send exactly the same volume. This is the allocator with one multiplier per day. Primary budget $b = 0.5$ pushes per user per day.

**Secondary use: one multiplier.** $\lambda$ is set once per setting as the $(1 - b)$ quantile of the user-day scores on the fit users' exploration logs, floored at 0, with the daily cap $\lfloor b\,n \rfloor$ on top; realized volume and the number of days the cap binds are reported. Total deliveries never exceed $28 \lfloor b\,n \rfloor$ in either use.

### 12.5 Profile misspecification

The policy profile is rebuilt from hourly counts $c_{u,h}$ (04-09 to 04-21, UTC+8) with two perturbations per setting:
- *Shift* $\Delta \in [-3, 3]$ hours: $c'_{u,h} = (1 - f)\, c_{u, h - \lfloor\Delta\rfloor} + f\, c_{u, h - \lfloor\Delta\rfloor - 1}$ with $f = \Delta - \lfloor\Delta\rfloor$, hours modulo 24, then windowing and smoothing as in 11.1. Three hours moves most of a window into the next one.
- *Noise* $\xi \in [0, 1]$: $a'_{u,s} \propto a_{u,s}\, e^{\xi z_{u,s}}$ with standard normal $z$ from a stream keyed by the setting's perturbation seed (D032). At $\xi = 1$ the best slot is close to random.

At $\Delta = \xi = 0$ the profile equals `profile_policy`, which already differs from the simulator profile (best slots agree for 38.8% of users, D024). The exploration policy and every policy of a setting use the same perturbed profile.

### 12.6 Sweep design

- *Users*: policies are evaluated on the 5,000-user quick sample (11); models are fitted on exploration logs of 10,000 of the other 22,285 users, drawn with `default_rng(1)`; score diagnostics use 2,000 of the remaining 12,285, drawn with `default_rng(2)`. The three sets are disjoint.
- *Main sweep*: 64 settings from a scrambled Latin hypercube over the 11 ranges of 11.5 and $\Delta$, $\xi$ (12.5), uniform on each range.
- *Annoyance-free sweep*: the same 64 settings with $\epsilon_v = 0$ (D023).
- *Stress sweep*: the same 64 settings with $\kappa_o = 0.005$, outside the evidence; reported apart and never pooled.
- *One-at-a-time grid*: from the reference setting with $\Delta = \xi = 0$, each of the 13 parameters at 5 evenly spaced points of its range, endpoints included, others at their reference values; 61 distinct settings.
- *Fit variability*: at the reference setting, models refitted on 10 exploration run seeds.
- *Seeds*: per setting a world seed, an exploration run seed, a perturbation seed and evaluation run seeds 101, 102 and 103 ($R = 3$; D033), shared by every policy of the setting so differences are paired on common random numbers. $R$ was set from timing on synthetic inputs of full size: about 2 to 3 hours on 4 processes with $R = 3$, about 3 to 4 with $R = 5$ (D031).
- *Freezing*: every design point and seed is written to `results/policy_design.json` by the design code and committed and pushed before any policy runs on evaluation users.

### 12.7 Metrics, comparisons and decision rule

**Primary metric**: active days per user over the 28 days, averaged over the $R$ run seeds.

**Primary comparisons** (difference in mean active days, first minus second):

| Id | Comparison | Question |
|---|---|---|
| H1 | G against S, $b = 0.5$ | Does the T-learner gate beat the response threshold at equal volume? |
| H2 | G2 against S, $b = 0.5$ | Does the decomposed gate beat it? |
| H3 | G at $b = 0.5$ against T | Does the gate keep more users active with half the volume? |
| H4 | T against F | Does the user's own best slot beat one fixed slot? |

**Secondary** (intervals, no tests): G − N, T − N, F − N; H3 split as (G − T½) + (T½ − T); G2 − T; L1, T on the cascade list against T on the two-tower list; S, G, G2 with one multiplier; opens per user in week 1 and in total; active days in week 4; delivered pushes per user; share opted out by day 28; active days per delivered push; score diagnostics (12.3); fit variability, the SD of each $\Delta_h$ over the 10 refits against its bootstrap SE (D056).

**Bootstrap** (D029). 10,000 resamples of the 5,000 users (seed 0); the same resampled user sets serve every setting. Per setting $j$ and comparison $h$: $\Delta_{h,j}$ is the mean per-user difference, with its percentile interval and the two-sided p-value of the Conventions; this interval is conditional on the setting's fitted models. Sweep average $\bar\Delta_h$, the unweighted mean over the 64 settings: each resample also draws 64 settings with replacement, so its interval covers the choice of settings and, through their different seeds, model-fit and world randomness.

**Decision rule.**
1. *Per setting*: each of H1 to H4 at level 0.0125 (Bonferroni over the four, so one comparison's verdict does not depend on another's p-value). A win for the first policy if rejected with a positive difference, a loss if negative, otherwise neither. The 64 verdicts share users, draws and resamples and are correlated.
2. *Sweep level*: Holm across the eight tests of $\bar\Delta_h$ (four comparisons in the main and in the annoyance-free sweep) at family-wise level 0.05.
3. *Labels*, per comparison and sweep, applied in this order: *beats in most sampled settings* if $\bar\Delta_h$ is rejected positive and there are at least 48 wins of 64; otherwise *depends on the setting* if there are at least 4 wins and at least 4 losses; otherwise *better on average* if $\bar\Delta_h$ is rejected positive; mirror labels for negative; otherwise *no detectable difference*, reported with the median interval half-width. Wins, losses and neither are always reported, and losing settings are listed with their parameters. "Share of settings" means share of 64 points drawn uniformly over the parameter box, not a probability that the real world falls in any region.

The stress sweep gets the same per-setting verdicts and no label. Wherever H2 is reported, the result carries the sentence that G2's assumption holds in the simulator by construction, so part of any H2 win comes from the simulator's structure.

**Flip thresholds** (descriptive), from the one-at-a-time grid: for each parameter and comparison, a flip is reported only where adjacent grid points have intervals that exclude 0 with opposite signs, located by linear interpolation of the estimates, with an interval from the same bootstrap (D061); where estimates cross 0 without that, "not resolved"; otherwise "no flip in range". The same rule applied to F − N along the $\epsilon_v$ grid gives the annoyance elasticity below which one push a day has a net gain in active days; it is reported first among the descriptive results.

**Plausibility band** (11.4): for each main-sweep setting, the relative lift of F over N in mean daily active share, $(\bar a_F - \bar a_N)/\bar a_N$, classified by its interval as below 0.07%, inside 0.07% to 0.5%, above 0.5%, or straddling a bound. The band comes from changes to running notification programs, not from pushing against not pushing, so it is a loose check. Reading fixed now: if most settings lie below the band, the report states that under these ranges one push a day adds less activity than the smallest reported program change, and that conclusions which rely on push benefits are weaker for it; ranges are not changed after results are seen, and settings are never filtered by the band.

**Code.** `policy/` (`profile.py`, `features.py`, `gate.py`, `budget.py`, `rules.py`, `design.py`, `sweep.py`, `compare.py`); `python -m push_lab.policy.design` writes `results/policy_design.json`; `python -m push_lab.policy.sweep` writes per-setting aggregates to `results/runs/` (ignored) and `results/policy.json`. **Tests.** Allocator against brute force on small instances with non-increasing gains, and the documented non-monotone case; deliveries never exceed the cap; S, G and G2 deliver the same volume per day; $\Delta = \xi = 0$ reproduces `profile_policy` and a 24-hour shift is the identity; a policy cannot read outcomes of the current day; two policies give a user the same template at the same (day, slot); Holm, the per-setting verdicts and the labels on hand-made inputs; the two-level bootstrap on synthetic data with a known mean; fixed seeds give identical results.

---

## 13. Copy and template selection

**Problem.** Choose which copy template goes with each push. The simulator gives each template a latent effect on the open probability (11.2) and penalizes templates the user saw recently; the question is whether a bandit that learns from opens finds the templates with the largest drawn effects and rotates them so the novelty penalty stays small. Every result here is Tier C.

**What this can and cannot show.** The simulator never reads the text. A template's effect is a random draw per world, unrelated to its wording (11.2, SPEC Section 7). The comparisons below therefore test whether the selection policy finds and rotates the templates with the largest drawn effects; they say nothing about whether the generated copy is good or which wording works better for real users. This sentence goes with every copy result in the report.

### 13.1 Families and placeholders

KuaiRand has no titles or captions, so copy names no video or creator. A template may use only placeholders that the policy can fill at send time from data before the decision (Section 1 cutoff rule) or from the simulated history it observes (12.1).

| Id | Family | Placeholder | Filled from | Values | Eligible when |
|---|---|---|---|---|---|
| 0 | Creator update | `{creator_videos}` | clicks by the user on videos of the item's author, standard logs 04-09 to 04-21 | "1 video", "2 videos", ..., "20+ videos" | the count is at least 1 |
| 1 | Popular now | `{top_n}` | the item's rank in the trending list at $c_2$ (Section 2) | "top 10", "top 50", "top 100" | rank at most 100 |
| 2 | Personal pick | none | | | always |
| 3 | Short or long watch | `{duration}` | `video_duration` (static item attribute) | "under 30 seconds", "under a minute", "1 to 3 minutes", "3 to 10 minutes", "over 10 minutes" | always |
| 4 | Welcome back | `{days_away}` | simulated visits the policy observed | "2 days", ..., "6 days", "a week or more" | no visit on days $d-1$ and $d-2$ ($d \ge 2$) |
| 5 | Time of day | `{time_of_day}` | send slot | "this morning", "at lunch", "this evening", "tonight" | always |

Each family's placeholder is required in its templates (Personal pick has none). Copy is English: the repository is English and the simulator does not read the text, so the language changes no result (D035). Template $k = 6j + f$ is the $j$-th kept variant of family $f$, so with 10 templates per family $\text{family}(k) = k \bmod 6$, the mapping the simulator has used since M4, and the M5 copy draws are unchanged (D036).

The eligibility attributes are computed per user and cascade-list position by `sim.build` from data before $c_2$ and cached with the other inputs. No value is committed; the committed copy holds placeholders only.

### 13.2 Generation

One request per family, six in all, through the Messages API with structured output (a JSON schema of 16 variants, each a `title` and a `body`). The prompt states the app (a short-video app), the push context, the family and its placeholder with every value, the length limits, the charset and the banned phrasings of 13.3. The model is passed by `--model` or an environment variable and is not written in code or docs. Requests use adaptive thinking, effort `high`, `max_tokens` 16,000 and no sampling parameters.

`data/creative/` holds the full text of every prompt, the variants as returned, the kept catalog and `manifest.json`: per request the model the API returns, the request and message ids, `stop_reason`, token usage and the UTC time; for the run the SDK version, effort, `max_tokens` and the sha256 of the output schema; per variant the sha256 of `title + "\n" + body` and its filter verdicts. Generation runs once; everything else reads the committed files and needs no key. Tests and CI never call the API; the parser is tested against a hand-made response in the API's format.

### 13.3 Filters

Fixed before generation and applied in this order; a variant is dropped at its first failure, and the counts per rule and family go to `results/copy.json`.

1. *Schema*: the response parses against the schema; `title` and `body` are non-empty strings.
2. *Charset*: printable ASCII only, no line breaks, no emoji.
3. *Placeholders*: in title and body separately, braces only as `{name}` with a name of the family; the family's placeholder appears exactly once in title and body together; Personal pick has none.
4. *Length*: with every placeholder replaced by its longest value, title 10 to 50 characters and body 20 to 100.
5. *Banned claims* (case-insensitive, whole words and their plurals):
   - any digit outside a placeholder (no invented numbers);
   - false urgency: hurry, last chance, don't miss, do not miss, before it's gone, ending soon, ends soon, expires, limited, act now, right now, urgent, only today, countdown;
   - inducement: free, win, winner, prize, reward, bonus, coins, gift, cash, guaranteed, unlock;
   - clickbait or unverifiable: you won't believe, shocking, secret, everyone, millions, viral, breaking, exclusive, #;
   - guilt: you missed, miss you, we miss, don't leave, come back now;
   - family claims the data cannot back: Creator update may not say new, just, latest, posted or uploaded (upload times are not tied to simulated days); Popular now may not say today or this hour;
   - more than one `!` in title and body together, or a word of three or more letters in capitals.
6. *Near-duplicates*: the text `title | body`, lowercased, each placeholder replaced by one symbol and whitespace collapsed, is compared by character 4-gram Jaccard with every earlier variant that passed all rules, in all families (families in id order, variants in returned order); at 0.5 or above it is dropped.

Every run also applies the filters to hand-made violations of the rules (`filters.VIOLATIONS`) and records how many were rejected as expected, so a run where every variant passes can be told apart from filters that do not work.

The first 10 survivors of each family form the catalog, and nothing is regenerated. Had a family kept fewer than 10, the numbering $6j + f$ and the M5 copy draws would not hold and the template sweep would need a new design; every family kept 10 (D038).

**Checks outside the simulator** (descriptive, no tests):
- Pass rates per rule and family.
- A blind rating by one rater of the 60 kept templates and 20 variants drawn with `default_rng(0)` from those that passed every filter but were not kept (no variant was rejected, D038), shuffled together with `default_rng(0)`, with placeholders filled from synthetic values and no sign of the filter verdict. Three yes or no questions per item: *clear* (easy to understand at a glance), *fits* (reads like a push from a short-video app), *misleading* (misleads or pressures the reader: false urgency, invented facts, rewards, guilt). Only counts are committed: per question, the number of yes among the kept and among the other variants. A template rated misleading is a miss of the filters. No variant was rejected, so the agreement between the rater and the filters cannot be estimated: the prompt listed the limits and banned phrasings, and every variant kept to them (D038).

### 13.4 Template policies

A *round* $t$ is a delivered push to user $u$ at (day $d$, slot $s$); $A_t$ is the set of templates eligible there (13.1); the reward $r_t$ is whether the push was opened, which the policy observes after day $d$ (12.1). Each policy records the probability $b_t(a)$ it gave every $a \in A_t$.

- **U, uniform**: $b_t(a) = 1/|A_t|$.
- **RD, recovering difference softmax** (Yancey and Settles 2020, Eq. 1 to 4). At the first slot of each day, scores are recomputed from all rounds of earlier days of the run, pooled over users. For arm $a$, over the rounds with $a \in A_t$ and $\theta \le b_t(a) \le 1 - \theta$:
  $$\bar\mu_a^+ = \frac{\sum_{a_t = a} w_t^+ r_t}{\sum_{a_t = a} w_t^+},\; w_t^+ = b_t(a)^{-1}; \qquad \bar\mu_a^- = \frac{\sum_{a_t \ne a} w_t^- r_t}{\sum_{a_t \ne a} w_t^-},\; w_t^- = (1 - b_t(a))^{-1}$$
  with Kish effective sizes $n_a^\pm = (\sum w)^2 / \sum w^2$ and $\mu_a$ the plain mean reward of those rounds. Both means are shrunk toward $\mu_a$: $\hat\mu_a^\pm = (\bar\mu_a^\pm n_a^\pm + \mu_a m)/(n_a^\pm + m)$, and
  $$\hat s_a = \frac{\hat\mu_a^+ - \hat\mu_a^-}{\hat\mu_a^-}, \qquad s^*_{a,t} = \hat s_a - \gamma_b\, 0.5^{\Delta_{u,a}/h_b}, \qquad b_t(a) = \frac{e^{s^*_{a,t}/\tau}}{\sum_{a' \in A_t} e^{s^*_{a',t}/\tau}},$$
  where $\Delta_{u,a}$ is the time in days since $a$ was last delivered to $u$ (penalty 0 if never). An arm without rounds scores 0.
- **D0, difference softmax without the penalty**: RD with $\gamma_b = 0$.
- **RAW** (secondary): RD with $\hat s_a = (\hat\mu_a^+ - \bar\mu)/\bar\mu$, $\bar\mu$ the mean reward of all rounds, $\bar\mu_a^+$ shrunk toward $\bar\mu$. It ignores who was eligible and shows the confounding the difference score removes (Creator update is eligible for users with a known author affinity, which the response model rewards).
- **O, myopic oracle** (reference, not a policy anyone could run): the eligible template with the largest $c_a (1 - \gamma\, 0.5^{\Delta_{u,a}/h})$ under the setting's true $c$, $\gamma$ and $h$; ties to the lowest id.

**Fixed values**, not tuned: $\gamma_b = 0.017$ and $h_b = 15$ days (the paper's production values; the bandit does not know the simulator's $\gamma$ and $h$, which the sweep varies), $\tau = 0.005$, $\theta = 0.005$ (the paper's threshold). The paper used $\tau = 0.0025$ with arm scores from millions of rounds; here each arm sees tens of thousands, so $\tau$ is doubled for more exploration (its Table 7: about 35% of rounds off the top arm).

**Prior strength** (deviation from the paper, D037). The paper sets $m$ once from arms with large samples; no arm here has one. Each day $m$ is set by the method of moments over the arms with $n_a^+ > 0$: $v = \max\big(0,\ \overline{(\bar\mu_a^+ - \mu_a)^2} - \overline{\mu_a(1 - \mu_a)/n_a^+}\big)$ and $m = \bar\mu(1 - \bar\mu)/v - 1$; if $v = 0$, every $\hat s_a = 0$ (D056). With no copy effect the scores therefore stay near 0 and RD behaves like U with a recency penalty.

**Pairing.** Every template policy runs with the same delivery rule T (12.2: one push a day at the user's best slot under the setting's perturbed profile) and the same run draws, so sends, items and opt-outs are identical across them; eligibility for Welcome back can diverge through visits. Each policy picks its template by inverse CDF from one uniform per (user, day, slot) drawn from a stream keyed by (run seed, day, slot) (D032), with $A_t$ in ascending id, so U, D0, RD and RAW share their random numbers. The template stream of the M5 policies is unchanged.

### 13.5 Sweep and comparisons

**Users**: all 27,285. No model is fitted outside the run, so no user is held out, and the bandit learns from about 760,000 rounds per run (one push a day per user for 28 days). Intervals bootstrap users conditional on each run's learning path; the seed-variability check below measures what that leaves out.

**Settings** (frozen in `results/template_design.json` and pushed before any run):
- *Main*: the 64 main-sweep points of 12.6 with their world and perturbation seeds; annoyance and opt-out act on the same volume in every arm.
- *Grid*: the one-at-a-time points of 12.6 for base rate, copy spread, novelty penalty and novelty half-life (20 distinct settings, the reference among them).
- *Seed variability*: the reference setting with 10 further triples of run seeds (D056).
- Evaluation run seeds 101, 102, 103, as in M5.

**Primary metric**: opens per user over the 28 days, averaged over the run seeds. Templates act only on the open probability, so active days are secondary.

**Primary comparisons** (first minus second):

| Id | Comparison | Question |
|---|---|---|
| B1 | RD against U | Does the recovering bandit beat uniform templates? |
| B2 | RD against D0 | Does the recency penalty help, given the same learned scores? |

**Secondary** (intervals, no tests): D0 − U; RAW − RD; O − U and the share $(\text{RD} - \text{U})/(\text{O} - \text{U})$ where the interval of O − U excludes 0; active days and opens per delivered push for every pair; repeat rate (share of rounds whose template reached the same user within the previous 15 days); at day 28, the Spearman correlation of $\hat s_a$ with $\log c_a$ and the overlap of the 10 highest-scored templates with the 10 largest $c_a$; seed variability, the SD of each comparison over the 10 seed triples against its mean bootstrap SE (D056).

**Decision rule** (as 12.7). Per setting, B1 and B2 each at level 0.025 (Bonferroni over two) with 10,000 bootstrap resamples of users; sweep averages over the 64 main settings with the two-level bootstrap of 12.7; Holm across the two sweep tests at 0.05; the labels of 12.7 with the same thresholds (48 of 64; 4 and 4). Flip thresholds along the four grid parameters by the rule of 12.7.

**Expected sizes**, stated before any run. With about 60 arms, rotating among the strongest ones is worth roughly $1.5\sigma$ to $2\sigma$ of the open rate (8% to 10% at the reference $\sigma = 0.05$); by day 28 each arm has a few hundred opens, a relative standard error of about 5%, close to the spread itself at the reference, so B1 is expected to grow with $\sigma$ and base rate and may be undetectable at small $\sigma$. At $\sigma = 0$ any gain of RD over U comes from novelty alone, at most about $0.3\gamma$ of opens, which is not detectable per setting.

**References.**
- Yancey, K. P., Settles, B. (2020). A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications. *KDD 2020*. https://doi.org/10.1145/3394486.3403351
- Kish, L. (1965). *Survey Sampling*. Wiley.

**Code.** `creative/` (`families.py`, `filters.py`, `generate.py`: `python -m push_lab.creative.generate --model <id>`), `policy/templates.py` (U, RD, D0, RAW, O), `policy/template_design.py`, `policy/template_sweep.py`, writing `results/copy.json` and `results/templates.json`. **Tests.** Each filter rule on hand-made accepts and rejects; every committed template passes the filters and fills from fixture data within the length limits; the generation script against a hand-made response in the API's format; the RD estimator against a hand-computed two-arm example, and with uniform logging $\bar\mu_a^-$ equals the mean over the other eligible arms; with $\sigma = \gamma = 0$ every template policy gives the same opens on the same draws; the logged probabilities match the softmax and empirical choices match them within sampling error; U, D0, RD and RAW share the uniform of each (user, day, slot); a policy cannot read the current day's opens; fixed seeds give identical results.

---

## 14. Off-policy evaluation

**Problem.** Estimate the value of a policy that was never run from logs of another policy with known propensities, and find out where such estimates can be trusted. An estimate gets Tier B only when it is computed on real logs with an estimator first checked against a known answer on real data (SPEC Section 5). The only real logs with two policies and known propensities here are the Open Bandit Dataset (OBD); every estimate on simulator logs, and every check against simulator truth, is Tier C.

### 14.1 Estimators

Logged rounds $i = 1, \dots, n$ with context $x_i$, action $a_i$ drawn by the logging policy with probability $b_i = b(a_i \mid x_i)$, and reward $r_i$. A target policy $\pi$ gives each action a probability $\pi(a \mid x_i)$; the weight is $w_i = \pi(a_i \mid x_i) / b_i$. With an outcome model $\hat q(x, a)$ and $\hat q(x, \pi) = \sum_a \pi(a \mid x)\, \hat q(x, a)$:

$$
\hat V_{\text{DM}} = \frac1n \sum_i \hat q(x_i, \pi), \quad
\hat V_{\text{IPS}} = \frac1n \sum_i w_i r_i, \quad
\hat V_{\text{SNIPS}} = \frac{\sum_i w_i r_i}{\sum_i w_i}, \quad
\hat V_{\text{DR}} = \hat V_{\text{DM}} + \frac1n \sum_i w_i \big(r_i - \hat q(x_i, a_i)\big).
$$

- Assumptions: the logged propensities are the true ones; overlap, $b(a \mid x) > 0$ wherever $\pi(a \mid x) > 0$; rewards of a round depend only on its context and action (no carryover). IPS is unbiased under these; SNIPS trades a small bias for lower variance; DM is unbiased only if $\hat q$ is right; DR is unbiased if either the propensities or $\hat q$ is right (Dudík et al. 2011).
- $\hat q$ is a LightGBM classifier with the base model of Section 8, fitted with 2-fold cross-fitting by user (each fold's rows scored by the model fitted on the other fold), so no round is scored by a model that saw it (D056).
- No clipping or switching: every weight is used as is. Each estimate is reported with the effective sample size $(\sum w)^2 / \sum w^2$ and the largest weight.
- Intervals: bootstrap over the units that make rounds dependent (users in the simulator, impressions in OBD; OBD has no impression id, so rows in time order with the same user features less than a second apart form one impression, which can merge consecutive impressions of a user and only coarsens the units), 1,000 resamples, 95% percentile, with $\hat q$ held fixed. Holding $\hat q$ fixed leaves its fitting noise out of each interval, so a single interval understates the variance of DM and DR; coverage over many log seeds (14.3) exposes this, and under-coverage of DM or DR is read first as this cause. The four estimators are reported side by side; when two of their intervals do not overlap the estimate is flagged inconclusive (SPEC Section 5). The simulator checks (14.3, 14.6) do not carry the flag (D046). On OBD impressions turned out to be dependent themselves, so these intervals are too narrow; 14.2 adds post hoc intervals over ten-minute blocks (D063).

**Code.** `ope/estimators.py`. **Tests.** On synthetic bandit data with a known policy value: IPS and DR within their bootstrap intervals of the truth over many seeds; with $\pi = b$, IPS equals the on-policy mean and every weight is 1; DR equals DM when $\hat q$ is exact; SNIPS is invariant to scaling all weights; ESS equals $n$ for equal weights.

### 14.2 Validation on the Open Bandit Dataset (real data)

- Data: the committed small sample, Men's campaign, 10,000 rows logged by uniform random (`random_men`, propensity $1/34$ per item and position) and 10,000 by Bernoulli Thompson sampling (`bts_men`). Each row is an item shown at one of three positions with a click label; the rows of one impression share user features and fall within a second of each other. Folds of the outcome model split impressions.
- Target: the BTS policy. Its probability of each (item, position) is computed as in the Open Bandit Pipeline at the pinned commit (Saito et al. 2021): $10^6$ draws from the production Beta prior of the Men's campaign, each ranking the 34 items by their draws and placing the top three at positions 1 to 3. The policy is context-free, so the probabilities are the same in every round. The prior values are copied from the pinned file, whose sha256 is in the code; CI does not download it, so a test checks only their sums and one entry.
- Known answer: the on-policy value, the click rate of the BTS logs, with a bootstrap interval.
- Estimates: DM, IPS, SNIPS and DR of the BTS value from the random logs; $\hat q$ uses the user features, user-item affinities, item features and position.
- Validation measure: for each estimator, the relative error $|\hat V - V| / V$ over 1,000 resamples of each log, drawn independently and matched by index (its median and 95% range, and the RMSE; D060), and whether the estimate's 95% interval contains $V$. The click rate of the BTS logs is resampled over impressions.
- Limitation stated with every small-sample OBD number: the sample has 46 clicks in the random logs and 69 in the BTS logs, so the check has low power: it can show a gross failure, not a small bias.
- Secondary, the full Men's campaign (SPEC optional item 6): the same check on all its random and BTS impressions, downloaded once locally and never in CI; the same rule applies with a larger sample (D042).
- Post hoc (D043, D046): IPS with the BTS logs' own (item, position) frequencies as the target, as a ratio to $V$, and that ratio minus the IPS ratio under the fixed prior. Both use the same 1,000 resamples of the random logs, so the difference of the two IPS estimates is paired; the resamples of $V$ are independent of them, and the empirical frequencies are held fixed across resamples (D060).
- An estimator that passes is *validated on OBD* (14.4), and its OBD estimate of the BTS value is the project's Tier B result. (D046: no estimator passed on the full campaign, so the project has no Tier B result.)
- Post hoc (D063): the full-campaign check with ten-minute blocks as resampling units. A row at time $t$ is in block $\lfloor (t - t_0) / 600\,\mathrm{s} \rfloor$, $t_0$ = 2019-11-24T00:00:00Z, which gives 1,008 blocks. A resample draws 1,008 blocks with replacement and takes the rows of both logs in each; 10,000 resamples, with multiplicities from `numpy.random.default_rng([20261110, 0, i])` for resample $i$. Estimates, the on-policy value and, for the post hoc target, the BTS frequencies are recomputed in every resample, with $\hat q$ held fixed. The rule of 14.4 is applied to these intervals as a record only, also under 100 other random streams; the registered verdicts and the tier reading do not change.

**Code.** `ope/obd.py` (`python -m push_lab.ope.obd`, writes `results/ope_obd.json`); post hoc, `ope/obd_blocks.py` (`python -m push_lab.ope.obd_blocks`, writes `results/ope_obd_blocks.json`).

### 14.3 Validation in the simulator

**Logs.** The exploration policy of 11.7 on the 5,000 evaluation users, at the setting's parameters. A round is a user-day; its action is the vector of the four slot send decisions, $a \in \{0, 1\}^4$, with probability $b(a \mid u, d) = \prod_s p_s^{a_s} (1 - p_s)^{1 - a_s}$, $p_s = 0.775$ at the user's best slot under the policy profile and $0.075$ elsewhere. Templates are uniform in the logs and in every target, so their probabilities cancel and are not part of the action. The logs record the vector, its probability, the per-slot send probabilities, and the user-day reward. The context of a user-day is built from the logged history at the first slot of the day: the day-level features of 12.3 (policy visit rate, score of the next item, day, share of earlier days with a visit, visit the day before, opens so far, days since the last open and the last delivery, decayed volume), the four profile shares, the best slot, and whether the user opted out before the day.

**Reward.** Primary: whether the user visited on day $d$ (active day). Secondary: opens on day $d$.

**Targets** (all with uniform templates), from near to far from the logging policy:

| Id | Target | $\pi$ | Largest weight |
|---|---|---|---|
| T | one push at the best slot | deterministic | 1.6 |
| T½ | T on a seeded half of user-days, else nothing | stochastic | 2.8 |
| R | each slot with probability 1/4 | stochastic | 123 |
| N | no pushes | deterministic | 5.6 |
| F | one push at the fixed slot of 12.2 | deterministic | 69 (users whose best slot differs) |
| T2 | pushes at the two best slots | deterministic | 20 |

The smallest propensity of one slot-and-template action, $0.075/60 = 0.00125$, does not enter, since templates cancel. The smallest probability of a user-day vector is $0.225 \times 0.075^3 \approx 0.0001$ (the three other slots without the best one); only R can choose it, with weight 123, the largest of any target.

**Truth.** Each target runs online on the same 5,000 users at the same setting; its value is the mean reward per user-day. The first run (D043) averaged 20 run seeds shared by every target. The amended run (D044) uses the exact value in the control, where days are independent and templates uniform, so a day with send vector $a$ is visited with probability $1 - (1 - v_u) \prod_{s: a_s = 1} (1 - \bar p_{u,s})$, $\bar p_{u,s}$ the open probability at slot $s$ averaged over templates; at the reference it averages 200 run seeds per target, different for every target, with a t interval (D056). Each log's estimates are kept so that coverage can be recomputed.

**Settings and seeds.**
- *No-carryover control*: the reference with $\epsilon_f = \epsilon_v = \kappa_o = \kappa_r = \gamma = 0$, and each user's $p_0$ held at the mean of the first 28 items of the list, so neither behaviour nor the list position carries over and a day's reward depends only on that day's context and action; here every estimator but DM should be unbiased, which checks the implementation.
- *Reference*: the reference setting (11.5) with $\Delta = \xi = 0$. Fatigue, annoyance, opt-out and return carry over across days, so a target's own history differs from the logged one; per-day OPE cannot see this, and the size of the resulting bias is what the check measures.
- 50 log seeds per setting (run seeds 1,001 to 1,050), world seed 10,100 as in the grid of 12.6; truth run seeds 2,001 to 2,020 in the first run, $100{,}000 + 1{,}000\,i + k$ for target $i$ and $k < 200$ in the amended run.

**Validation measures**, per setting, target and estimator, over the 50 log seeds: bias $= \overline{\hat V} - V$ with a 95% interval whose variance adds that of the estimates over log seeds and that of the truth over run seeds, RMSE, and coverage, the share of seeds whose 95% bootstrap interval contains $V$, with a Wilson interval.

### 14.4 Pre-registered decision rule

- *Validated in the simulator* (per setting, target and estimator): the Wilson 95% interval of coverage contains 0.95, and the bias interval contains 0. Otherwise *not validated*, with the reason (under-coverage, bias or both) (D056).
- *Primary results*: the validation verdicts of DR and IPS for each target at the reference setting. The control setting is reported first, as an implementation check; if DR or IPS is not validated there for T, T½ or N, the simulator results are reported as an implementation failure and not interpreted.
- *Validated on OBD*: the estimate's 95% interval contains the on-policy value and the median relative error over resamples is below 0.5. The threshold is loose because of the sample size.
- Expectation stated before any run: in the control setting all estimators but DM validated for every target; at the reference, IPS and DR biased for N and T2, whose volumes differ most from the logs, and close for T and T½.
- No Holm correction: these are validation verdicts, not comparisons between policies; every verdict is reported, none is dropped.

**References.**
- Dudík, M., Langford, J., Li, L. (2011). Doubly Robust Policy Evaluation and Learning. *ICML 2011*. https://arxiv.org/abs/1103.4601
- Swaminathan, A., Joachims, T. (2015). The Self-Normalized Estimator for Counterfactual Learning. *NeurIPS 2015*.
- Saito, Y., Aihara, S., Matsutani, M., Narita, Y. (2021). Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation. *NeurIPS 2021 Datasets and Benchmarks*. https://arxiv.org/abs/2008.07146

**Code.** `ope/estimators.py`, `ope/obd.py`, `ope/simulator.py` (`python -m push_lab.ope.simulator`, writes `results/ope_simulator.json`), `report/evidence.py` (`results/evidence.json`) (D056). **Tests.** The estimator tests of 14.1; the BTS probabilities sum to 1 per position and match a direct Monte Carlo of the prior; logged user-day probabilities match the product formula and empirical frequencies; in a synthetic no-carryover world, IPS and DR cover the online truth at about the nominal rate; fixed seeds give identical results.

### 14.5 Evidence matrix

`results/evidence.json` lists every headline claim of the project with its tier, the results file and key it comes from, the estimate and its interval, and what the tier cannot support (SPEC Section 5). It is built by code from the results files, so its numbers cannot drift from them; the HTML report (M8) renders it.

### 14.6 Sensitivity sweep of the simulator check

Added after the results of 14.3 (D046, D047): simulator results are reported only with a sweep, and the direction of the per-day bias depends on which carryover dominates. (D048: in the sweep, annoyance alone sets the direction.)

- *Grid*: one parameter at a time over annoyance $\epsilon_v$, fatigue elasticity $\epsilon_f$, opt-out $\kappa_o$, return boost $\kappa_r$ and novelty penalty $\gamma$, at 5 evenly spaced points of each range of 11.5, the other parameters at the reference and $\Delta = \xi = 0$; 24 distinct settings, since the annoyance and fatigue midpoints are both the reference.
- *Seeds*: 10 log seeds per setting (3,001 to 3,010), not used by any earlier run; 100 independent truth runs per target (the first 100 seeds of each target in the amended scheme of 14.3); world seed 10,100; the 5,000 evaluation users.
- *Measures*: for every setting, target and estimator (IPS and DR, visit reward), the bias with its interval and the verdict of 14.4; coverage over 10 seeds has a coarse resolution and is reported without being read. Bias resamples draw the log seeds and the truth runs with replacement, 10,000 times.
- *Flips* (descriptive): along each grid, for the DR and IPS bias of T½, N and T2, by the rule of 12.7: a crossing counts only where adjacent points have intervals that exclude 0 with opposite signs.
- *Reading*, fixed before the run: a direction of the per-day bias is stated only with the grid ranges where it holds, for example "per-day OPE overestimates two pushes a day above an annoyance elasticity of x". Expectation: the bias of T2 grows with annoyance and fatigue and falls with return boost.

**Code.** `ope/carryover.py` (`python -m push_lab.ope.carryover`, writes `results/ope_carryover.json`; each log's estimates go to `results/runs/ope_carryover/`).

