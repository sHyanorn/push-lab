"""Off-policy estimators DM, IPS, SNIPS and DR with a cluster bootstrap (METHODS 14.1)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

ESTIMATORS = ("DM", "IPS", "SNIPS", "DR")
SAMPLES = 1_000
CHUNK = 100


@dataclass(frozen=True)
class Logged:
    """One target policy evaluated on logged rounds.

    ``weight`` is pi(a_i | x_i) / b(a_i | x_i); ``q_logged`` is q(x_i, a_i) and ``q_target``
    is q(x_i, pi), both from a model that did not see round i; ``cluster`` groups rounds that
    are resampled together (users, or one per round)."""

    reward: FloatArray
    weight: FloatArray
    q_logged: FloatArray
    q_target: FloatArray
    cluster: IntArray

    def sums(self) -> FloatArray:
        """Per cluster [count, sum w r, sum w, sum q_target, sum w (r - q_logged)]."""
        _, idx = np.unique(self.cluster, return_inverse=True)
        k = int(idx.max()) + 1
        columns = (
            np.ones_like(self.reward),
            self.weight * self.reward,
            self.weight,
            self.q_target,
            self.weight * (self.reward - self.q_logged),
        )
        return np.stack([np.bincount(idx, weights=c, minlength=k) for c in columns], axis=1)


def from_sums(totals: FloatArray) -> FloatArray:
    """Estimates in the order of ``ESTIMATORS`` from summed columns [..., 5]."""
    n, wr, w, qt, correction = (totals[..., i] for i in range(5))
    dm = qt / n
    with np.errstate(divide="ignore", invalid="ignore"):
        snips = np.where(w > 0, wr / w, np.nan)
    out: FloatArray = np.stack([dm, wr / n, snips, dm + correction / n], axis=-1)
    return out


def point(logged: Logged) -> dict[str, float]:
    values = from_sums(logged.sums().sum(axis=0))
    return dict(zip(ESTIMATORS, map(float, values), strict=True))


def effective_size(weight: FloatArray) -> float:
    """Kish's effective sample size (sum w)^2 / sum w^2."""
    return float(weight.sum() ** 2 / (weight**2).sum()) if (weight**2).sum() > 0 else 0.0


def bootstrap(logged: Logged, samples: int = SAMPLES, seed: int = 0) -> FloatArray:
    """Estimates [sample, estimator] over resamples of clusters with replacement; the outcome
    model is held fixed."""
    sums = logged.sums()
    k = len(sums)
    rng = np.random.default_rng(seed)
    out = np.empty((samples, len(ESTIMATORS)))
    for start in range(0, samples, CHUNK):
        rows = min(CHUNK, samples - start)
        idx = rng.integers(0, k, size=(rows, k))
        counts = np.stack([np.bincount(r, minlength=k) for r in idx]).astype(np.float64)
        out[start : start + rows] = from_sums(counts @ sums)
    return out


def evaluate(logged: Logged, samples: int = SAMPLES, seed: int = 0) -> dict[str, object]:
    """Every estimator with its 95% percentile interval, the effective sample size and the
    largest weight."""
    est = point(logged)
    boot = bootstrap(logged, samples, seed)
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {
        "estimates": {
            name: {"value": est[name], "ci_low": float(lo), "ci_high": float(hi)}
            for name, lo, hi in zip(ESTIMATORS, low, high, strict=True)
        },
        "effective_size": effective_size(logged.weight),
        "rounds": len(logged.weight),
        "max_weight": float(logged.weight.max()),
    }
