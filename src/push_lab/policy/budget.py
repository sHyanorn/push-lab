"""Budget allocation by marginal gain with a Lagrange multiplier (METHODS Section 12.4).

The policies call ``top_k`` only, the allocator with one multiplier per day (D028). ``allocate``
is kept for the tests that check it against ``brute_force``.
"""

import itertools

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


TOL = 1e-9


def _counts(gains: FloatArray, lam: float, largest: bool) -> IntArray:
    """Per unit, the smallest or largest x maximizing the sum of (g_j - lam), j <= x."""
    cum = np.concatenate([np.zeros((len(gains), 1)), np.cumsum(gains - lam, axis=1)], axis=1)
    reached = cum >= cum.max(axis=1, keepdims=True) - TOL
    first = np.argmax(reached[:, ::-1] if largest else reached, axis=1)
    out: IntArray = (cum.shape[1] - 1 - first if largest else first).astype(np.int64)
    return out


def _breakpoints(gains: FloatArray) -> FloatArray:
    """0 and every positive average gain of a block of consecutive sends: the values of lam
    at which a unit's maximizer can change."""
    cum = np.concatenate([np.zeros((len(gains), 1)), np.cumsum(gains, axis=1)], axis=1)
    m = gains.shape[1]
    averages = [(cum[:, b] - cum[:, a]) / (b - a) for a in range(m) for b in range(a + 1, m + 1)]
    values = np.concatenate([np.zeros(1), *averages])
    out: FloatArray = np.unique(values[values >= 0])
    return out


def allocate(gains: FloatArray, budget: int) -> tuple[IntArray, float]:
    """Sends per unit and the multiplier (METHODS 12.4).

    ``gains[u, j]`` is the gain of unit u's (j+1)-th send. lam is the smallest value >= 0
    at which the smallest maximizers fit the budget; if lam > 0, leftover budget raises
    units, in index order, to their largest maximizer at lam when the whole step fits. With
    non-increasing gains the total gain equals the brute-force optimum; otherwise it can fall
    short (D009).
    """
    if budget < 0:
        raise ValueError("the budget must be non-negative")
    points = _breakpoints(gains)
    # The smallest maximizer does not grow with lam, so bisect over the breakpoints
    low, high = -1, len(points) - 1
    if _counts(gains, float(points[high]), largest=False).sum() > budget:
        raise ValueError("no multiplier fits the budget")
    while high - low > 1:
        mid = (low + high) // 2
        fits = _counts(gains, float(points[mid]), largest=False).sum() <= budget
        low, high = (low, mid) if fits else (mid, high)
    lam = float(points[high])
    x = _counts(gains, lam, largest=False)
    top = _counts(gains, lam, largest=True)
    left = budget - int(x.sum()) if lam > 0 else 0
    for u in np.flatnonzero(top > x):
        step = int(top[u] - x[u])
        if step <= left:
            x[u] = top[u]
            left -= step
    return x, lam


def total_gain(gains: FloatArray, x: IntArray) -> float:
    return float(sum(gains[u, : x[u]].sum() for u in range(len(gains))))


def brute_force(gains: FloatArray, budget: int) -> float:
    """Best total gain over every per-unit count that fits the budget (small inputs only)."""
    best = 0.0
    ranges = [range(gains.shape[1] + 1)] * len(gains)
    for x in itertools.product(*ranges):
        if sum(x) <= budget:
            best = max(best, total_gain(gains, np.array(x, dtype=np.int64)))
    return best


def top_k(scores: FloatArray, eligible: npt.NDArray[np.bool_], k: int) -> npt.NDArray[np.bool_]:
    """The k eligible units with the highest scores, ties by index; all if fewer."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    order = order[eligible[order]][:k]
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[order] = True
    return chosen
