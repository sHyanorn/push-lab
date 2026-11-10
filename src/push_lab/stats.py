"""Summaries of bootstrap resamples shared by the evaluation steps (METHODS Conventions)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from push_lab.runtime import rounded

FloatArray = npt.NDArray[np.float64]


def p_value(boot: FloatArray) -> float:
    """Twice the smaller tail share at zero, capped at 1; NaN resamples are left out.

    Only the ranking and uplift resamples can hold NaN (a resample without an eligible user).
    The policy and template resamples are means of finite simulator counts over a fixed number
    of users, so leaving NaN out never changes them.
    """
    boot = boot[~np.isnan(boot)]
    return min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))


def holm(p_values: dict[str, float], level: float = 0.05) -> dict[str, bool]:
    """Holm's step-down procedure; True where the hypothesis is rejected."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    rejected: dict[str, bool] = {}
    still = True
    for rank, (name, p) in enumerate(ordered):
        still = still and p <= level / (len(ordered) - rank)
        rejected[name] = still
    return rejected


def interval(point: float, boot: FloatArray) -> dict[str, float]:
    """A point value with the 95% percentile interval of its resamples, rounded."""
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"value": rounded(point), "ci_low": rounded(low), "ci_high": rounded(high)}


def flips(x: FloatArray, points: FloatArray, boot: FloatArray) -> list[dict[str, Any]]:
    """Crossings of zero between adjacent grid points ``x`` (sorted), with ``boot``
    [sample, point]. A crossing is resolved only if both intervals exclude 0."""
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    out: list[dict[str, Any]] = []
    for i in range(len(x) - 1):
        a, b = points[i], points[i + 1]
        if np.sign(a) == np.sign(b):
            continue
        resolved = (low[i] > 0 and high[i + 1] < 0) or (high[i] < 0 and low[i + 1] > 0)
        entry: dict[str, Any] = {
            "between": [rounded(x[i]), rounded(x[i + 1])],
            "resolved": bool(resolved),
        }
        if resolved:
            at = x[i] + (x[i + 1] - x[i]) * a / (a - b)
            ba, bb = boot[:, i], boot[:, i + 1]
            crossing = np.sign(ba) != np.sign(bb)
            xs = x[i] + (x[i + 1] - x[i]) * ba[crossing] / (ba[crossing] - bb[crossing])
            lo, hi = np.percentile(xs, [2.5, 97.5])
            entry |= {
                "at": rounded(at),
                "ci_low": rounded(lo),
                "ci_high": rounded(hi),
                "resamples_crossing": rounded(crossing.mean()),
            }
        out.append(entry)
    return out


def flip_result(crossings: list[dict[str, Any]]) -> str:
    if not crossings:
        return "no flip in range"
    return "flip" if any(c["resolved"] for c in crossings) else "not resolved"
