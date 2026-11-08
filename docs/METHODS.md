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

**Undated user attributes.** The user table (`user_active_degree`, `is_lowactive_period`, `register_days`, `onehot_feat*`) is a single snapshot with no date. It is constant within a user, so per-user AUC and NDCG are unaffected, but it shapes the two-tower user vectors and therefore Recall@K. It is used as released, and this is a limitation of every result that involves the user tower.

**Code.** `rank/splits.py`. **Tests.** No feature table built at cutoff $c$ contains a row with $\text{date} \ge c$; the two halves are disjoint, cover all random-exposure users and are identical across runs.

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
- *Latency*: wall time of fine scoring per user.

Model-internal measures say how much truncation changes the model's choice, not whether the change matters to users, and are labeled so in the report.

**Code.** `rank/cascade.py`. **Tests.** With $N$ equal to the catalog size, the cascade returns the catalog reference's top-1; survival is monotone non-decreasing in $N$.

---

## 5. Fine ranker

**Model.** Two LightGBM binary classifiers, one for clicks and one for long views, on the same features. The fused score is

$$
r(u, i) = w\, \hat p_{\text{click}}(u, i) + (1 - w)\, \hat p_{\text{long}}(u, i), \qquad w = 0.5 .
$$

$w = 0.5$ is fixed in advance; $w \in \{0, 0.25, 0.5, 0.75, 1\}$ is reported as a secondary sensitivity table.

**Features**, all computed from the window before the cutoff:

- *User*: the categorical fields of Section 3, log of 1 + follow, fan and friend counts, `register_days`, impressions per day, smoothed click and long-view rates.
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
- A ratio with an empty denominator is 0. Rows with tied scores enter together: the curve is evaluated only at the end of each tied group and joined by straight lines, which is the expected curve under a random order within the group. This is how scikit-uplift 0.5.1 computes both curves.
- Both are *joint* curves: treated and control rows are ranked together. Diemert et al. (2021) report the *separate, relative* AUUC (each arm ranked on its own, response rates differenced), which Devriendt et al. (2022) found more robust to unequal arm sizes; Criteo numbers here are therefore not directly comparable with theirs, which also use a different split.

**Areas.** With curve points $(k_j, C_j)$, $k_0 = C_0 = 0$, the area is the trapezoid sum $A = \sum_j (k_j - k_{j-1})(C_j + C_{j-1})/2$, and the random-targeting line from $(0, 0)$ to $(n, C(n))$ has area $n\, C(n)/2$. The *perfect* curve orders rows by an oracle score that uses the outcomes: for Qini, $y t - y (1 - t)$ (treated responders first, control responders last, negative effects included); for the uplift curve, the scikit-uplift construction. Then

$$
\text{Qini coefficient} = \frac{A_Q - n\,Q(n)/2}{A_Q^\ast - n\,Q(n)/2}, \qquad
\text{AUUC} = \frac{A_U - n\,U(n)/2}{A_U^\ast - n\,U(n)/2},
$$

which equal scikit-uplift's `qini_auc_score(negative_effect=True)` and `uplift_auc_score`. Both are 0 for random targeting in expectation, and the denominator is the same for every method on the same test rows, so paired differences rank methods exactly as the unnormalized areas do.

**Binned curves on Criteo.** With 12.6M test rows, the model's curve is evaluated at 200 score quantiles: rows are grouped into bins of about equal size by score, all rows with the same score in the same bin, and the bins are treated as tied groups. A curve point every 0.5% of the test set changes the area by much less than its sampling error. The perfect curve uses only the outcome and treatment counts and stays exact. Hillstrom curves are exact. With 85% of rows treated, a bin can hold few control rows; `results/uplift.json` reports, per method, the minimum and the 5th, 50th and 95th percentiles of control rows per bin and the number of bins with none (where a control rate would be 0/0 and the bin's control responders count as 0).

**Bootstrap.** Rows are resampled (D012: 1,000 resamples, 95% percentile intervals, seed 0), and every statistic is recomputed on each resample, including the perfect-curve areas in the denominators, which depend on the resampled counts of the four $(t, y)$ classes. On Hillstrom the rows themselves are resampled. On Criteo, each metric depends on the data only through the counts in cells (score bin, $t$, $y$), so drawing 12.6M rows with replacement is the same as drawing the cell counts from a multinomial with the observed cell shares; for a paired comparison of methods $a$ and $b$ the cells are (bin under $a$, bin under $b$, $t$, $y$). The intervals cover test-set sampling only: each model is fitted once on a fixed training set with a fixed seed. A test checks on small synthetic data that the cell bootstrap and row resampling give the same intervals within Monte Carlo error.

**Cross-check.** Hillstrom Qini coefficients and AUUC for every method are compared with scikit-uplift 0.5.1 (development dependency only) on the same scores; the largest absolute difference is reported in `results/uplift.json` and must be below $10^{-9}$. The unit tests run the same comparison on synthetic data with ties.

**References.**
- Radcliffe, N. J. (2007). Using control groups to target on predicted lift: Building and assessing uplift models. *Direct Marketing Analytics Journal*, 14–21. Direct Marketing Association Analytics Council.
- Devriendt, F., Van Belle, J., Guns, T., Verbeke, W. (2022). Learning to rank for uplift modeling. *IEEE Transactions on Knowledge and Data Engineering* 34(10), 4888–4904. https://doi.org/10.1109/TKDE.2020.3048510
- Diemert, E., Betlei, A., Renaudin, C., Amini, M.-R., Gregoir, T., Rahier, T. (2021). A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling. arXiv:2111.10106 (not peer reviewed).

**Code.** `uplift/curves.py`. **Tests.** Curves and both coefficients against hand-computed examples, against scikit-uplift on synthetic data with ties, and binned against exact on data with few distinct scores (equal when every bin is one tied group).

---

## 10. Targeting under a budget and primary comparisons

**Budget.** A budget $b$ is the share of the test population that may be treated; the top $\lceil b n \rceil$ rows by score are treated and the rest are not. The value of a method at $b$ is $U(bn)$ (linear between curve points), reported as incremental responders per 1,000 test rows and as the share of the treat-everyone effect, $U(bn) / U(n)$. Budgets: $b \in \{0.1, 0.2, 0.3, 0.5\}$. Methods: the four learners, the response model and random. This table is secondary and carries no significance claims.

**Primary comparisons**, on the primary outcome (`visit`) and the Qini coefficient, all with the T-learner as the uplift method, fixed before any model is fitted on these data. Holm correction across the four at family-wise level 0.05:

| Id | Dataset | Comparison | Test |
|---|---|---|---|
| U1 | Hillstrom | T-learner Qini coefficient against 0 (random targeting) | bootstrap of the coefficient |
| U2 | Hillstrom | T-learner against response model | paired bootstrap of the difference |
| U3 | Criteo | T-learner Qini coefficient against 0 | bootstrap of the coefficient |
| U4 | Criteo | T-learner against response model | paired bootstrap of the difference |

The p-values follow the Conventions (twice the smaller tail share at zero). U1 and U3 test the T-learner's own coefficient against 0, its expected value under random targeting; they do not use the random baseline's scores, which are a single draw and whose coefficient carries its own sampling noise. A claim that uplift ranking beats random or response ranking on a dataset needs the comparison rejected under Holm with a positive difference. Everything else (the other learners, AUUC, `conversion`, the budget table) is secondary and reported without significance claims.

**Results** go to `results/uplift.json`: setup (row counts, treated shares, test-set sizes), every metric with its interval and tier A, the four comparisons with p-values and Holm decisions, the budget table, Qini curves at 101 points for the report, the scikit-uplift cross-check and fit times.

**Code.** `uplift/evaluate.py` (`python -m push_lab.uplift.evaluate`). Criteo is read by column and scored in row groups (D015).
