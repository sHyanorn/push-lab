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

**Index.** Item vectors go into a faiss HNSW index with inner-product metric (`M = 32`, `efConstruction = 200`, `efSearch = 128`). Exact search (`IndexFlatIP`) is the reference.

**Reference.** Yi, X., Yang, J., Hong, L., Cheng, D. Z., Heldt, L., Kumthekar, A., Zhao, Z., Wei, L., Chi, E. (2019). Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. *RecSys 2019*. https://doi.org/10.1145/3298689.3346996

**Code.** `recall/two_tower.py`, `recall/index.py`. **Tests.**

- The loss equals a hand-computed value on a 3 × 3 example.
- With all $q_i$ equal, the loss equals plain in-batch softmax.
- A duplicated item in a batch is masked.
- HNSW overlap with exact search at $K = 200$ is at least 0.99 on the evaluation users.

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
- **NDCG@5.** Each user's random exposures ranked by score, gain 1 for a click, ideal DCG from the best ordering; averaged over users with at least one click.

**Reference.** Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., He, X. (2022). KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. *CIKM 2022*. https://arxiv.org/abs/2208.08696

**Primary comparisons** (Holm correction across the two):

| Id | Comparison | Metric |
|---|---|---|
| C1 | Two-tower against popularity | Recall@200 |
| C2 | Fused fine ranker against two-tower score alone | GAUC (click) |

Everything else (other $K$, long-view GAUC, NDCG@5, the $w$ table, truncation) is secondary and reported without significance claims.

**Code.** `rank/metrics.py`. **Tests.** Each metric against hand-computed examples, including ties (AUC counts a tie as 0.5) and users with a single class (excluded from GAUC).
