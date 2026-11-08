"""Qini and uplift curves, their normalized areas and the cell bootstrap (METHODS Section 9).

Every statistic is a function of counts per score group, ordered from the highest score: the
treated and control rows in the group and their responders. A row bootstrap only reweights
those counts, so the same code serves exact curves, binned curves and their resamples.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CRITEO_BINS = 200


@dataclass(frozen=True)
class Counts:
    """Per-group counts in score order, highest score first."""

    n_t: FloatArray
    y_t: FloatArray
    n_c: FloatArray
    y_c: FloatArray

    @property
    def n(self) -> float:
        return float(self.n_t.sum() + self.n_c.sum())


def exact_groups(score: FloatArray) -> IntArray:
    """Group id per row: one group per distinct score, 0 for the highest."""
    unique, inverse = np.unique(score, return_inverse=True)
    groups: IntArray = (len(unique) - 1 - inverse).astype(np.int64)
    return groups


def quantile_groups(score: FloatArray, bins: int = CRITEO_BINS) -> IntArray:
    """Group id per row from score quantiles, 0 for the highest; equal scores share a group."""
    negated = -score
    edges = np.unique(np.quantile(negated, np.linspace(0, 1, bins + 1)[1:-1]))
    groups: IntArray = np.searchsorted(edges, negated, side="left").astype(np.int64)
    return groups


def counts(
    groups: IntArray,
    t: npt.NDArray[np.integer],
    y: npt.NDArray[np.integer],
    weights: FloatArray | None = None,
    n_groups: int | None = None,
) -> Counts:
    """Aggregate rows (or weighted cells) into per-group counts."""
    w = np.ones(len(groups)) if weights is None else weights
    size = int(groups.max()) + 1 if n_groups is None else n_groups
    treated = t == 1
    wt, wc = np.where(treated, w, 0.0), np.where(treated, 0.0, w)
    return Counts(
        n_t=np.bincount(groups, weights=wt, minlength=size),
        y_t=np.bincount(groups, weights=wt * y, minlength=size),
        n_c=np.bincount(groups, weights=wc, minlength=size),
        y_c=np.bincount(groups, weights=wc * y, minlength=size),
    )


def _ratio(a: FloatArray, b: FloatArray) -> FloatArray:
    out: FloatArray = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return out


def _cumulative(c: Counts) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    n_t, y_t = np.cumsum(c.n_t), np.cumsum(c.y_t)
    n_c, y_c = np.cumsum(c.n_c), np.cumsum(c.y_c)
    return n_t + n_c, n_t, y_t, n_c, y_c


def _with_origin(x: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    return np.r_[0.0, x], np.r_[0.0, v]


def qini_curve(c: Counts) -> tuple[FloatArray, FloatArray]:
    """Points (k, Q(k)) with Q(k) = Y_T(k) - Y_C(k) N_T(k) / N_C(k)."""
    k, n_t, y_t, n_c, y_c = _cumulative(c)
    return _with_origin(k, y_t - y_c * _ratio(n_t, n_c))


def uplift_curve(c: Counts) -> tuple[FloatArray, FloatArray]:
    """Points (k, U(k)) with U(k) = (Y_T(k) / N_T(k) - Y_C(k) / N_C(k)) k."""
    k, n_t, y_t, n_c, y_c = _cumulative(c)
    return _with_origin(k, (_ratio(y_t, n_t) - _ratio(y_c, n_c)) * k)


def area(x: FloatArray, v: FloatArray) -> float:
    return float(np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2))


def _classes(c: Counts) -> tuple[float, float, float, float]:
    """Totals of treated responders, treated non-responders, control responders and
    control non-responders."""
    tr, cr = float(c.y_t.sum()), float(c.y_c.sum())
    return tr, float(c.n_t.sum()) - tr, cr, float(c.n_c.sum()) - cr


def _ordered(groups: list[tuple[float, float, float, float]]) -> Counts:
    a = np.array(groups, dtype=np.float64)
    return Counts(n_t=a[:, 0], y_t=a[:, 1], n_c=a[:, 2], y_c=a[:, 3])


def perfect_qini_counts(c: Counts) -> Counts:
    """Groups under the oracle score y t - y (1 - t): treated responders first, control
    responders last, everyone else tied in between."""
    tr, tn, cr, cn = _classes(c)
    return _ordered([(tr, tr, 0, 0), (tn, 0, cn, 0), (0, 0, cr, cr)])


def perfect_uplift_counts(c: Counts) -> Counts:
    """Groups under scikit-uplift's oracle 2 [y = t] + (y if CR > TN else t)."""
    tr, tn, cr, cn = _classes(c)
    tail = [(0.0, 0.0, cr, cr), (tn, 0.0, 0.0, 0.0)]
    if cr <= tn:
        tail.reverse()
    return _ordered([(tr, tr, 0, 0), (0, 0, cn, 0), *tail])


def _normalized(curve: FloatArray, perfect: FloatArray, x: FloatArray, xp: FloatArray) -> float:
    n, end = float(x[-1]), float(curve[-1])
    random = n * end / 2
    denominator = area(xp, perfect) - random
    return (area(x, curve) - random) / denominator if denominator != 0 else 0.0


def qini_coefficient(c: Counts) -> float:
    """Area between the Qini curve and the random line over the same for the perfect curve."""
    x, q = qini_curve(c)
    xp, qp = qini_curve(perfect_qini_counts(c))
    return _normalized(q, qp, x, xp)


def auuc(c: Counts) -> float:
    """Normalized area under the uplift curve, as scikit-uplift's ``uplift_auc_score``."""
    x, u = uplift_curve(c)
    xp, up = uplift_curve(perfect_uplift_counts(c))
    return _normalized(u, up, x, xp)


def uplift_at(c: Counts, budgets: FloatArray) -> FloatArray:
    """U(b n), linear between curve points, for each budget share b."""
    x, u = uplift_curve(c)
    out: FloatArray = np.interp(budgets * x[-1], x, u)
    return out


def qini_on_grid(c: Counts, points: int = 101) -> tuple[FloatArray, FloatArray]:
    """Qini curve at evenly spaced shares of the rows, divided by the number of treated rows."""
    x, q = qini_curve(c)
    share = np.linspace(0.0, 1.0, points)
    values: FloatArray = np.interp(share * x[-1], x, q) / c.n_t.sum()
    return share, values


@dataclass(frozen=True)
class Cells:
    """Distinct (group under each method, t, y) combinations with their row counts."""

    groups: list[IntArray]
    t: IntArray
    y: IntArray
    count: FloatArray


def cells(groups: list[IntArray], t: npt.NDArray[np.integer], y: npt.NDArray[np.integer]) -> Cells:
    """Collapse rows into cells; a statistic of the rows is the same statistic of the cells."""
    columns = np.column_stack([*groups, t.astype(np.int64), y.astype(np.int64)])
    unique, count = np.unique(columns, axis=0, return_counts=True)
    return Cells(
        groups=[unique[:, i] for i in range(len(groups))],
        t=unique[:, -2],
        y=unique[:, -1],
        count=count.astype(np.float64),
    )


def resampled_weights(count: FloatArray, samples: int, seed: int) -> Iterator[FloatArray]:
    """Cell counts of row bootstrap resamples: each resample draws as many rows as there are,
    with replacement, which is a multinomial draw over cells with the observed shares."""
    rng = np.random.default_rng(seed)
    total = int(count.sum())
    shares = count / count.sum()
    for _ in range(samples):
        yield rng.multinomial(total, shares).astype(np.float64)
