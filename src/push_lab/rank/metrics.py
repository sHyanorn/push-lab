"""Per-user ranking metrics, user bootstrap and Holm correction (METHODS Section 6).

Metrics return one value per user (NaN where the user is not eligible), so that every
estimate and comparison can be bootstrapped over the same resampled users.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0


def recall_at_k(retrieved: IntArray, positives: list[set[int]], k: int) -> FloatArray:
    """Share of each user's positives within their top ``k``; NaN without positives."""
    out = np.full(len(positives), np.nan)
    for u, (row, pos) in enumerate(zip(retrieved, positives, strict=True)):
        if pos:
            out[u] = len(pos & set(row[:k].tolist())) / len(pos)
    return out


def _grouped(
    users: IntArray, order: IntArray, scores: FloatArray, labels: FloatArray
) -> pd.DataFrame:
    frame = pd.DataFrame({"user": users, "score": scores, "label": labels})
    frame["pos"] = pd.Categorical(frame["user"], categories=order)
    return frame


def user_auc(
    users: IntArray, scores: FloatArray, labels: FloatArray, order: IntArray
) -> FloatArray:
    """ROC AUC per user (ties count 0.5), aligned with ``order``; NaN for single-class users."""
    frame = _grouped(users, order, scores, labels)
    frame["rank"] = frame.groupby("pos", observed=True)["score"].rank(method="average")
    g = frame.groupby("pos", observed=False)
    n = g["label"].size().to_numpy(dtype=float)
    n_pos = g["label"].sum().to_numpy(dtype=float)
    rank_sum = frame[frame["label"] == 1].groupby("pos", observed=False)["rank"].sum()
    n_neg = n - n_pos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum.to_numpy() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    out: FloatArray = auc.astype(np.float64)
    return out


def impressions(users: IntArray, order: IntArray) -> FloatArray:
    counts = pd.Series(users).value_counts().reindex(order).fillna(0)
    out: FloatArray = np.asarray(counts, dtype=np.float64)
    return out


def user_ndcg(
    users: IntArray, scores: FloatArray, labels: FloatArray, order: IntArray, k: int = 5
) -> FloatArray:
    """NDCG@k per user with binary gains; NaN for users without a positive."""
    frame = _grouped(users, order, scores, labels)
    frame = frame.sort_values(["pos", "score"], ascending=[True, False], kind="stable")
    frame["position"] = frame.groupby("pos", observed=True).cumcount()
    discount = 1.0 / np.log2(frame["position"].to_numpy() + 2.0)
    frame["dcg"] = np.where(frame["position"] < k, frame["label"] * discount, 0.0)
    g = frame.groupby("pos", observed=False)
    dcg = g["dcg"].sum().to_numpy()
    n_pos = g["label"].sum().to_numpy()
    ideal = np.array([np.sum(1.0 / np.log2(np.arange(min(int(p), k)) + 2.0)) for p in n_pos])
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(n_pos > 0, dcg / ideal, np.nan)
    return out


def resample_indices(
    n_users: int, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> IntArray:
    """Bootstrap resamples of user positions, shared by every estimate and comparison."""
    rng = np.random.default_rng(seed)
    idx: IntArray = rng.integers(0, n_users, size=(samples, n_users))
    return idx


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float
    users: int

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high, "users": self.users}


def _nanmean_rows(values: FloatArray, idx: IntArray) -> FloatArray:
    # A resample can contain no eligible user; its mean is NaN and is dropped by the caller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out: FloatArray = np.nanmean(values[idx], axis=1)
    return out


def estimate(values: FloatArray, idx: IntArray, weights: FloatArray | None = None) -> Estimate:
    """Mean over eligible users with a 95% percentile interval."""
    valid = ~np.isnan(values)
    if weights is None:
        point = float(np.mean(values[valid]))
        boot = _nanmean_rows(values, idx)
    else:
        w = np.where(valid, weights, 0.0)
        v = np.where(valid, values, 0.0)
        point = float(np.sum(v * w) / np.sum(w))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.sum(v[idx] * w[idx], axis=1) / np.sum(w[idx], axis=1)
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return Estimate(point, float(low), float(high), int(valid.sum()))


@dataclass(frozen=True)
class Comparison:
    difference: Estimate
    p_value: float


def compare(a: FloatArray, b: FloatArray, idx: IntArray) -> Comparison:
    """Paired comparison of mean(a) - mean(b) over users eligible for both."""
    both = ~np.isnan(a) & ~np.isnan(b)
    diff = np.where(both, a - b, np.nan)
    est = estimate(diff, idx)
    boot = _nanmean_rows(diff, idx)
    boot = boot[~np.isnan(boot)]
    p = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return Comparison(est, p)


def holm(p_values: dict[str, float], level: float = 0.05) -> dict[str, bool]:
    """Holm's step-down procedure; True where the hypothesis is rejected."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected: dict[str, bool] = {}
    still = True
    for rank, (name, p) in enumerate(ordered):
        still = still and p <= level / (m - rank)
        rejected[name] = still
    return rejected
