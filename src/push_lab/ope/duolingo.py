"""Duolingo notification logs: logging check, day-level intervals, DL1, DL2 and DL3 (METHODS 15).

``python -m push_lab.ope.duolingo`` reads the converted logs of ``data/processed/duolingo`` and
writes ``results/duolingo.json``. The results are Tier A, limited to the value of one
notification under a template policy given the logged histories (D067).

The rows of one user cannot be linked, so every interval is a delta-method interval whose
standard error is built from day-level standard errors (``DayMoments``), each inflated for the
rows a user has within the day; decisions use the conservative sum over days.
"""

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.stats import norm

from push_lab.data import duolingo, fixtures, paths
from push_lab.policy.templates import ArmStats
from push_lab.runtime import Deadline, code_commit, one_thread_per_process, write_json
from push_lab.sim.world import DUOLINGO_STREAM, stream
from push_lab.uplift import learners

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

K = len(duolingo.TEMPLATES)
Z = 1.959964
# Logging check: uniform unless a template's ratio lies wholly outside this band
UNIFORM = (0.995, 1.005)
# Fixed values of the paper (Yancey and Settles 2020) and of METHODS 13.4
GAMMA, HALF_LIFE = 0.017, 15.0
TAUS = (0.02, 0.01, 0.005, 0.0025)
RD_TAU, RD_TAU_PAPER = 0.005, 0.0025
PAPER_BASELINE = 0.1295
# Paper lifts: printed percentages, or printed levels over the baseline (Tables 5 to 7)
PAPER_LIFT = {
    "Template": 0.012,
    "Template+UI": 0.018,
    "Reuse Last Template": -0.005,
    "Template+UI with recency penalty": 0.019,
    **{
        f"Template+UI with recency penalty, softmax {t:g}": v / PAPER_BASELINE - 1
        for t, v in zip(TAUS, (0.1306, 0.1311, 0.1316, 0.1319), strict=True)
    },
}
DL1 = tuple(PAPER_LIFT)
DL3 = ("U", "RD", "D0", f"RD, tau {RD_TAU_PAPER:g}")
# DL3a and DL3b
PAIRS = (("RD", "U"), ("RD", "D0"))
GRID_GAMMA = (0.0, 0.005, 0.01, 0.017, 0.03, 0.05, 0.1)
GRID_HALF_LIFE = (1.0, 3.0, 7.0, 15.0, 30.0)
TRAIN_SHARE, TEST_SHARE = 0.05, 0.10
# DL2 recency bins (days before, each closed on the right; the first includes 0)
EDGES = (1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 21.0, 30.0)
H_BOUNDS = (0.25, 365.0)
# Reading of gamma and h when L-BFGS-B reports no success (D076)
NOT_CONVERGED = "fit did not converge"
RESULTS_FILE = "duolingo.json"
SMOKE = Path("runs") / "smoke" / "duolingo"
SMOKE_ROWS = 20_000
# Rows per slice of a row group in the workers, and per chunk of the DL2 fits
SLICE = 250_000
CHUNK = 500_000


# ---------------------------------------------------------------------------------------------
# Day-level intervals


def inflation(j_mean: float) -> float:
    """Variance inflation of a day's mean, 1 + 2 mean(j_t): exact when each user's rows in the
    day are perfectly correlated with equal variance, whatever their number."""
    return 1 + 2 * j_mean


@dataclass
class DayMoments:
    """Per-day row counts, sums of z, sums of z z^T and sums of j_t for a vector z of per-row
    terms; an estimate is a smooth function of the means of z."""

    width: int
    days: dict[int, list[Any]] = field(default_factory=dict)

    def add(self, day: IntArray, z: FloatArray, j: IntArray) -> None:
        order = np.argsort(day, kind="stable")
        day, z, j = day[order], z[order], j[order]
        cuts = np.flatnonzero(np.diff(day)) + 1
        for start, end in zip(np.r_[0, cuts], np.r_[cuts, len(day)], strict=True):
            if end == start:
                continue
            seg = z[start:end]
            entry = self.days.setdefault(
                int(day[start]), [0, np.zeros(self.width), np.zeros((self.width,) * 2), 0]
            )
            entry[0] += end - start
            entry[1] += seg.sum(axis=0)
            entry[2] += seg.T @ seg
            entry[3] += int(j[start:end].sum())

    def merge(self, other: DayMoments) -> None:
        for d, (n, s, ss, js) in other.days.items():
            entry = self.days.setdefault(
                d, [0, np.zeros(self.width), np.zeros((self.width,) * 2), 0]
            )
            entry[0] += n
            entry[1] += s
            entry[2] += ss
            entry[3] += js

    @property
    def rows(self) -> int:
        return int(sum(e[0] for e in self.days.values()))

    def mean(self) -> FloatArray:
        out: FloatArray = sum(e[1] for e in self.days.values()) / self.rows
        return out

    def standard_errors(self, g: FloatArray) -> dict[str, Any]:
        """Conservative and independent-days standard errors of a statistic whose influence is
        g . (z - mean z), with each day's standard error inflated for same-day rows."""
        total = self.rows
        weights, ses, j_means = [], [], []
        for n, s, ss, js in (self.days[d] for d in sorted(self.days)):
            mean = s / n
            cov = (ss - n * np.outer(mean, mean)) / (n - 1) if n > 1 else np.zeros_like(ss)
            j_mean = js / n
            var = max(float(g @ cov @ g), 0.0) * inflation(j_mean)
            weights.append(n / total)
            ses.append(np.sqrt(var / n))
            j_means.append(j_mean)
        w, se = np.array(weights), np.array(ses)
        return {
            "conservative": float(w @ se),
            "independent": float(np.sqrt(np.sum((w * se) ** 2))),
            "days": len(w),
            "j_mean_max": float(max(j_means)),
            "single_row_days": int(sum(e[0] == 1 for e in self.days.values())),
        }

    def estimate(self, f: Callable[[FloatArray], float]) -> dict[str, Any]:
        """f(mean z) with both 95% intervals (delta method, numerical gradient). The step is
        relative to each mean, so the derivative of a ratio of small means keeps its accuracy
        (D078)."""
        mu = self.mean()
        g = np.zeros(self.width)
        for i in range(self.width):
            h = 1e-6 * abs(mu[i]) if mu[i] else 1e-6
            up, down = mu.copy(), mu.copy()
            up[i] += h
            down[i] -= h
            g[i] = (f(up) - f(down)) / (2 * h)
        return interval(f(mu), self.standard_errors(g))


def interval(value: float, se: dict[str, Any]) -> dict[str, Any]:
    """The value with both 95% intervals, the number of days, the largest mean j_t of a day
    and the number of days with a single row, whose standard error counts as 0."""
    out: dict[str, Any] = {"value": value}
    out |= {k: se[k] for k in ("days", "j_mean_max", "single_row_days") if k in se}
    for kind in ("conservative", "independent"):
        out[kind] = {
            "se": se[kind],
            "ci_low": value - Z * se[kind],
            "ci_high": value + Z * se[kind],
        }
    return out


# ---------------------------------------------------------------------------------------------
# Rounds of one row group


@dataclass(frozen=True)
class Rounds:
    """Converted rows: day, time of day, language code, eligible [row, template], set size,
    selected template, reward, recency [row, template] (inf if absent), j_t, days since the
    previous notification and history length."""

    day: IntArray
    time: FloatArray
    language: IntArray
    eligible: BoolArray
    size: IntArray
    selected: IntArray
    reward: FloatArray
    recency: FloatArray
    j: IntArray
    since: FloatArray
    length: IntArray

    def __len__(self) -> int:
        return len(self.day)

    def take(self, rows: IntArray | BoolArray | slice) -> Rounds:
        return Rounds(*(getattr(self, f)[rows] for f in self.__dataclass_fields__))


def read_rounds(path: Path, group: int, languages: Sequence[str]) -> Rounds:
    table = pq.ParquetFile(path).read_row_group(group)
    when = table["datetime"].to_numpy()
    # A null language is the value "None", as the conversion report counts it
    names: Any = pc.cast(table["ui_language"], pa.string())
    language = pc.index_in(pc.fill_null(names, "None"), value_set=pa.array(list(languages)))
    if language.null_count:
        raise ValueError("a language missing from the conversion report")
    mask = table["eligible"].to_numpy().astype(np.int64)
    eligible = ((mask[:, None] >> np.arange(K)) & 1).astype(bool)
    return Rounds(
        day=np.floor(when).astype(np.int64),
        time=when - np.floor(when),
        language=np.asarray(language.to_numpy(zero_copy_only=False), dtype=np.int64),
        eligible=eligible,
        size=eligible.sum(axis=1),
        selected=table["selected"].to_numpy().astype(np.int64),
        reward=table["completed"].to_numpy().astype(np.float64),
        recency=np.column_stack([table[c].to_numpy() for c in duolingo.RECENCY]).astype(np.float64),
        j=table["j"].to_numpy().astype(np.int64),
        since=table["since_previous"].to_numpy().astype(np.float64),
        length=table["history_length"].to_numpy().astype(np.int64),
    )


def logged(r: Rounds, values: FloatArray) -> FloatArray:
    """The value of the logged template in each row of ``values`` [row, template]."""
    out: FloatArray = values[np.arange(len(r)), r.selected]
    return out


# ---------------------------------------------------------------------------------------------
# Logging check


def logging_moments(r: Rounds) -> dict[str, DayMoments]:
    """Per template a, over all rows and per eligible-set size: moments of (1[a_t = a],
    1[a in A_t] / |A_t|), whose mean ratio is the selection ratio."""
    out: dict[str, DayMoments] = {}
    chosen = r.selected[:, None] == np.arange(K)
    expected = r.eligible / r.size[:, None]
    for size in [None, *np.unique(r.size)]:
        rows = np.ones(len(r), dtype=bool) if size is None else r.size == size
        for a in range(K):
            key = f"{duolingo.TEMPLATES[a]}|{'all' if size is None else int(size)}"
            z = np.column_stack([chosen[rows, a], expected[rows, a]]).astype(np.float64)
            out.setdefault(key, DayMoments(2)).add(r.day[rows], z, r.j[rows])
    return out


def logging_check(moments: dict[str, DayMoments]) -> dict[str, Any]:
    """Selection ratios with their intervals; the logs are taken as uniform unless the
    conservative interval of some template's ratio over all rows lies wholly outside 0.995 to
    1.005 (METHODS 15)."""
    ratios: dict[str, Any] = {}
    departures = []
    for key in sorted(moments):
        m = moments[key]
        if m.mean()[1] == 0:
            continue
        ratios[key] = m.estimate(lambda mu: float(mu[0] / mu[1]))
        if key.endswith("|all"):
            c = ratios[key]["conservative"]
            if c["ci_low"] > UNIFORM[1] or c["ci_high"] < UNIFORM[0]:
                departures.append(key.split("|")[0])
    return {"ratios": ratios, "departures": departures, "uniform": not departures}


# ---------------------------------------------------------------------------------------------
# Arm scores


def arm_sums(r: Rounds, arm: IntArray, arms: int) -> FloatArray:
    """Sums of 13.4 per arm over rows with |A_t| > 1 (theta = 0.005 excludes only b_t = 1):
    [w, w r, w^2] of rounds that chose the arm (w = 1/b_t), the same of rounds where it was
    eligible and not chosen (w = 1/(1 - b_t)), and [rounds, rewards] where eligible. ``arm``
    [row, template] numbers the arm of each template in each row."""
    rows, cols = np.nonzero(r.eligible & (r.size[:, None] > 1))
    b = 1.0 / r.size[rows]
    picked = cols == r.selected[rows]
    w = np.where(picked, 1 / b, 1 / (1 - b))
    y = r.reward[rows]
    index = arm[rows, cols]
    out = np.zeros((8, arms))
    for i, side in enumerate((picked, ~picked)):
        for k, v in enumerate((w, w * y, w**2)):
            out[3 * i + k] = np.bincount(index[side], weights=v[side], minlength=arms)
    out[6] = np.bincount(index, minlength=arms)
    out[7] = np.bincount(index, weights=y, minlength=arms)
    return out


def raw_scores(sums: FloatArray) -> FloatArray:
    """Relative difference scores without shrinkage (m = 0, the paper's offline setup); an arm
    with no round on either side, or a zero mean on the minus side, scores 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = sums[1] / sums[0]
        minus = sums[4] / sums[3]
        ok = (sums[0] > 0) & (sums[3] > 0) & (minus > 0)
        out: FloatArray = np.where(ok, (plus - minus) / minus, 0.0)
    return out


def shrunk_scores(sums: FloatArray, rounds: float, rewards: float) -> tuple[FloatArray, float]:
    """RD scores of 13.4 with the prior strength by its method of moments, computed by the
    template policies' own code; ``rounds`` and ``rewards`` count every training round."""
    stats = ArmStats(sums.shape[1])
    stats.plus, stats.minus, stats.eligible = sums[0:3], sums[3:6], sums[6:8]
    stats.rounds, stats.rewards = rounds, rewards
    # The prior strength as ArmStats.scores computes it, for the report
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_plus = np.where(sums[0] > 0, sums[1] / sums[0], 0.0)
        n_plus = np.where(sums[2] > 0, sums[0] ** 2 / sums[2], 0.0)
        center = np.where(sums[6] > 0, sums[7] / sums[6], 0.0)
    m = ArmStats.prior_strength(mu_plus, n_plus, center, rewards / rounds)
    return stats.scores(), float(m)


# ---------------------------------------------------------------------------------------------
# Policies


def penalty(recency: FloatArray, gamma: float, half_life: float) -> FloatArray:
    """gamma 0.5^(d / h), 0 for a template not in the history (d = inf)."""
    out: FloatArray = gamma * np.power(0.5, recency / half_life)
    return out


def argmax_policy(r: Rounds, score: FloatArray) -> FloatArray:
    """The eligible template with the largest score, ties to the earliest letter."""
    best = np.where(r.eligible, score, -np.inf).argmax(axis=1)
    out = np.zeros(r.eligible.shape)
    out[np.arange(len(r)), best] = 1.0
    return out


def softmax_policy(r: Rounds, score: FloatArray, tau: float) -> FloatArray:
    z = np.where(r.eligible, score / tau, -np.inf)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    out: FloatArray = e / e.sum(axis=1, keepdims=True)
    return out


def uniform_policy(r: Rounds) -> FloatArray:
    out: FloatArray = r.eligible / r.size[:, None]
    return out


def reuse_last(r: Rounds) -> FloatArray:
    """The eligible template with the smallest recency, uniform over ties and over every
    eligible template when none is in the history."""
    rec = np.where(r.eligible, r.recency, np.nan)
    low = np.nanmin(rec, axis=1)
    tied = r.eligible & (rec == low[:, None]) | (r.eligible & np.isinf(low)[:, None])
    out: FloatArray = tied / tied.sum(axis=1, keepdims=True)
    return out


@dataclass(frozen=True)
class Scores:
    template: FloatArray  # [template], DL1
    ui: FloatArray  # [language * K + template], DL1
    rd: FloatArray  # [template], DL3 with shrinkage

    def by_language(self, r: Rounds) -> FloatArray:
        out: FloatArray = self.ui.reshape(-1, K)[r.language]
        return out


def policies(r: Rounds, s: Scores) -> dict[str, FloatArray]:
    """Every DL1 and DL3 policy as a distribution over the eligible templates [row, template]."""
    ui = s.by_language(r)
    penalized = ui - penalty(r.recency, GAMMA, HALF_LIFE)
    rd = s.rd[None, :] - penalty(r.recency, GAMMA, HALF_LIFE)
    out = {
        "Baseline": uniform_policy(r),
        "Template": argmax_policy(r, np.broadcast_to(s.template, r.recency.shape)),
        "Template+UI": argmax_policy(r, ui),
        "Reuse Last Template": reuse_last(r),
        "Template+UI with recency penalty": argmax_policy(r, penalized),
        **{
            f"Template+UI with recency penalty, softmax {t:g}": softmax_policy(r, penalized, t)
            for t in TAUS
        },
        "RD": softmax_policy(r, rd, RD_TAU),
        "D0": softmax_policy(r, np.broadcast_to(s.rd, r.recency.shape), RD_TAU),
        f"RD, tau {RD_TAU_PAPER:g}": softmax_policy(r, rd, RD_TAU_PAPER),
    }
    out["U"] = out["Baseline"]
    return out


def grid_choices(r: Rounds, s: Scores) -> dict[tuple[float, float], FloatArray]:
    """pi(a_t) of the argmax of Template+UI scores minus gamma 0.5^(d/h) over the DL2 grid."""
    ui = s.by_language(r)
    out = {}
    for gamma in GRID_GAMMA:
        for h in GRID_HALF_LIFE:
            pi = argmax_policy(r, ui - penalty(r.recency, gamma, h))
            out[(gamma, h)] = logged(r, pi)
    return out


# ---------------------------------------------------------------------------------------------
# Outcome model for DM and DR


def features(r: Rounds, template: IntArray) -> FloatArray:
    """Language, template, |A_t|, the template's recency, time of day, days since the previous
    notification and history length; an absent recency or no previous notification is missing."""
    rec = r.recency[np.arange(len(r)), template]
    return np.column_stack(
        [
            r.language,
            template,
            r.size,
            np.where(np.isinf(rec), np.nan, rec),
            r.time,
            np.where(np.isinf(r.since), np.nan, r.since),
            r.length,
        ]
    ).astype(np.float64)


def fit_outcome_model(x: FloatArray, y: FloatArray, threads: int) -> lgb.Booster:
    """LightGBM classifier with the base model of Section 8; language and template are
    categorical."""
    params = learners.PARAMS | {"objective": "binary", "num_threads": threads}
    data = lgb.Dataset(x, y, categorical_feature=[0, 1])
    return lgb.train(params, data, num_boost_round=learners.ROUNDS)


def outcome_matrix(r: Rounds, model: lgb.Booster) -> FloatArray:
    """q(x, a) for every eligible template [row, template], 0 elsewhere."""
    rows, cols = np.nonzero(r.eligible)
    out = np.zeros(r.eligible.shape)
    out[rows, cols] = model.predict(features(r.take(rows), cols))
    return out


# ---------------------------------------------------------------------------------------------
# DL2: conditional likelihood of the completed rounds


@dataclass(frozen=True)
class Completed:
    """Completed rounds with |A_t| > 1: eligible [row, template], selected, recency, day, j."""

    eligible: BoolArray
    selected: IntArray
    recency: FloatArray
    day: IntArray
    j: IntArray

    @classmethod
    def of(cls, r: Rounds) -> Completed:
        keep = (r.reward == 1) & (r.size > 1)
        return cls(
            r.eligible[keep],
            r.selected[keep],
            r.recency[keep].astype(np.float32),
            r.day[keep],
            r.j[keep],
        )

    @classmethod
    def concat(cls, parts: Sequence[Completed]) -> Completed:
        return cls(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in cls.__dataclass_fields__)
        )

    def __len__(self) -> int:
        return len(self.selected)

    def chunks(self) -> Iterator[Completed]:
        for start in range(0, len(self), CHUNK):
            yield Completed(
                *(getattr(self, f)[start : start + CHUNK] for f in self.__dataclass_fields__)
            )


class Model:
    """u[a, t] = c_a + template term; the choice probability of a completed round is the
    softmax of u over its eligible templates, since the template was drawn uniformly. Only
    templates that occur have a c_a, and the earliest of them, A if it occurs, has c = 0 (D077)."""

    free: list[int]
    names: list[str]
    bounds: list[tuple[float | None, float | None]]
    start: FloatArray
    held: tuple[str, ...] = ()

    def terms(self, theta: FloatArray, c: Completed) -> tuple[FloatArray, list[FloatArray]]:
        """The template term [row, template] and its derivatives in the non-c parameters."""
        raise NotImplementedError

    def chunk(
        self, theta: FloatArray, c: Completed, scores: bool = False
    ) -> tuple[float, FloatArray, FloatArray | None]:
        """Log-likelihood sum, gradient sum and, with ``scores``, the per-round scores [row,
        parameter]. Ineligible templates get probability 0, so their terms never enter."""
        term, derivatives = self.terms(theta, c)
        u = term
        u[:, self.free] += theta[: len(self.free)]
        u[~c.eligible] = -np.inf
        top = u.max(axis=1, keepdims=True)
        p = np.exp(u - top)
        total = p.sum(axis=1, keepdims=True)
        p /= total
        rows = np.arange(len(c))
        loglik = float(np.sum(u[rows, c.selected] - top[:, 0] - np.log(total[:, 0])))
        # Scores: the chosen template's derivative minus the probability-weighted mean
        diff = -p
        diff[rows, c.selected] += 1.0
        if not scores:
            parts = [diff[:, self.free].sum(axis=0)]
            parts += [np.array([np.sum(diff * d)]) for d in derivatives]
            return loglik, np.concatenate(parts), None
        s = np.column_stack([diff[:, self.free]] + [(diff * d).sum(axis=1) for d in derivatives])
        return loglik, s.sum(axis=0), s

    def gradient(
        self, theta: FloatArray, c: Completed, deadline: Deadline | None = None
    ) -> tuple[float, FloatArray]:
        """Mean log-likelihood and its gradient, after checking the stop line."""
        if deadline is not None:
            deadline.check()
        ll, g = 0.0, np.zeros(len(theta))
        for part in c.chunks():
            a, b, _ = self.chunk(theta, part)
            ll += a
            g += b
        return ll / len(c), g / len(c)


class PaperForm(Model):
    """Term -gamma 0.5^(d/h), 0 for a template not in the history; theta = (c of every
    occurring template after the first, gamma, log h)."""

    def __init__(self, templates: Sequence[int] = range(K)) -> None:
        self.free = list(templates)[1:]
        self.names = [f"c_{duolingo.TEMPLATES[a]}" for a in self.free] + ["gamma", "log_h"]
        self.held = ("log_h",)
        free: list[tuple[float | None, float | None]] = [(None, None)] * (len(self.free) + 1)
        self.bounds = [*free, (float(np.log(H_BOUNDS[0])), float(np.log(H_BOUNDS[1])))]
        self.start = np.r_[np.zeros(len(self.free)), GAMMA, np.log(HALF_LIFE)]

    def terms(self, theta: FloatArray, c: Completed) -> tuple[FloatArray, list[FloatArray]]:
        """-gamma g with g = 0.5^(d/h), and its derivatives -g in gamma and -gamma g ln 2 d/h in
        log h; g = 0 where d = inf."""
        gamma, rate = theta[-2], np.log(2) / np.exp(theta[-1])
        rec = c.recency.astype(np.float64)
        g = np.exp(-rate * rec)
        dg = g * rate * np.where(g > 0, rec, 0.0)
        return -gamma * g, [-g, -gamma * dg]


def recency_bin(recency: FloatArray) -> IntArray:
    """Bin of each recency, len(EDGES) for a template not in the history."""
    out: IntArray = np.where(
        np.isinf(recency), len(EDGES), np.digitize(recency, EDGES, right=True)
    ).astype(np.int64)
    return out


class Curve(Model):
    """Term beta_k for the recency bin k of the template, 0 when not in the history; bins in
    which no eligible template falls are left out."""

    def __init__(self, bins: Sequence[int], templates: Sequence[int] = range(K)) -> None:
        self.bins = list(bins)
        self.free = list(templates)[1:]
        labels = ["[0, 1.5]"] + [f"({a:g}, {b:g}]" for a, b in pairwise(EDGES)]
        self.labels = [labels[k] for k in self.bins]
        self.names = [f"c_{duolingo.TEMPLATES[a]}" for a in self.free]
        self.names += [f"beta {x}" for x in self.labels]
        self.bounds = [(None, None)] * len(self.names)
        self.start = np.zeros(len(self.names))

    def terms(self, theta: FloatArray, c: Completed) -> tuple[FloatArray, list[FloatArray]]:
        k = recency_bin(c.recency.astype(np.float64))
        beta = np.zeros(len(EDGES) + 1)
        beta[self.bins] = theta[len(self.free) :]
        return beta[k], [(k == b).astype(np.float64) for b in self.bins]


def fit(model: Model, c: Completed, deadline: Deadline | None = None) -> dict[str, Any]:
    """Maximum likelihood (L-BFGS-B), with the intervals of ``intervals``; stops at the line
    of ``deadline`` at any evaluation of the likelihood."""
    result = minimize(
        lambda t: tuple(-x for x in model.gradient(t, c, deadline)),
        model.start,
        jac=True,
        method="L-BFGS-B",
        bounds=model.bounds,
        # The mean log-likelihood is flat in gamma and h: the default tolerances stop early
        options={"maxiter": 2_000, "ftol": 1e-15, "gtol": 1e-10},
    )
    theta = result.x
    at_bound = [
        name
        for (low, high), name, t in zip(model.bounds, model.names, theta, strict=True)
        if (low is not None and abs(t - low) < 1e-6) or (high is not None and abs(t - high) < 1e-6)
    ]
    return {
        "parameters": intervals(model, theta, c, deadline),
        "rounds": len(c),
        "converged": bool(result.success),
        "message": str(result.message),
        "at_bound": at_bound,
    }


def intervals(
    model: Model, theta: FloatArray, c: Completed, deadline: Deadline | None = None
) -> dict[str, Any]:
    """Each round's influence is -H^-1 s_t, H the mean Hessian by central differences of the
    gradient; intervals from the day-level standard errors of the influence. Where the model
    holds a parameter (``held``), each other parameter's standard error is the larger of the
    one from the full Hessian and the one with the held parameter fixed: away from a stationary
    point of h, as at its bound, the full Hessian can shrink the standard error of gamma toward
    0 when gamma is near 0, where h is not identified (D070)."""
    p = len(theta)
    hessian = np.zeros((p, p))
    for i in range(p):
        h = 1e-5 * max(1.0, abs(theta[i]))
        up, down = theta.copy(), theta.copy()
        up[i] += h
        down[i] -= h
        hessian[:, i] = (
            model.gradient(up, c, deadline)[1] - model.gradient(down, c, deadline)[1]
        ) / (2 * h)
    hessian = (hessian + hessian.T) / 2
    influence = -np.linalg.inv(hessian)
    free = [i for i, name in enumerate(model.names) if name not in model.held]
    fixed = np.zeros((p, p))
    fixed[np.ix_(free, free)] = -np.linalg.inv(hessian[np.ix_(free, free)])
    moments = DayMoments(p)
    for part in c.chunks():
        s = model.chunk(theta, part, scores=True)[2]
        assert s is not None
        moments.add(part.day, s, part.j)
    params = {}
    for i, name in enumerate(model.names):
        se = moments.standard_errors(influence[i])
        extra = {}
        if name not in model.held and model.held:
            held = moments.standard_errors(fixed[i])
            extra = {
                "full_hessian": {k: se[k] for k in ("conservative", "independent")},
                "held_fixed": {k: held[k] for k in ("conservative", "independent")},
            }
            for kind in ("conservative", "independent"):
                se[kind] = max(se[kind], held[kind])
        params[name] = interval(float(theta[i]), se) | extra
    return params


def read_gamma(gamma: dict[str, Any]) -> str:
    """The registered reading of gamma from its conservative interval."""
    c = gamma["conservative"]
    if c["ci_low"] <= 0 <= c["ci_high"]:
        return "decay not detected"
    if c["ci_high"] < 0:
        return "a recent showing raises completion"
    if (c["ci_high"] - c["ci_low"]) / 2 > GAMMA / 2:
        return "gamma not determined"
    inside = c["ci_low"] <= GAMMA <= c["ci_high"]
    return "consistent with these data" if inside else "not consistent"


def read_half_life(fitted: dict[str, Any]) -> dict[str, Any]:
    """h on the scale of days: its interval from the one on log h, and the registered reading
    when the fit converged and gamma lies above 0."""
    log_h = fitted["parameters"]["log_h"]
    out: dict[str, Any] = {"value": float(np.exp(log_h["value"]))}
    for kind in ("conservative", "independent"):
        out[kind] = {
            "ci_low": float(np.exp(log_h[kind]["ci_low"])),
            "ci_high": float(np.exp(log_h[kind]["ci_high"])),
        }
    c = out["conservative"]
    if not fitted["converged"]:
        out["reading"] = NOT_CONVERGED
    elif read_gamma(fitted["parameters"]["gamma"]) in (
        "decay not detected",
        "a recent showing raises completion",
    ):
        out["reading"] = "not read"
    elif "log_h" in fitted["at_bound"] or c["ci_high"] > 4 * c["ci_low"]:
        out["reading"] = "h not determined"
    else:
        inside = c["ci_low"] <= HALF_LIFE <= c["ci_high"]
        out["reading"] = "consistent" if inside else "not consistent"
    return out


def dl2(c: Completed, deadline: Deadline | None = None) -> dict[str, Any]:
    """The paper's form with its readings, none if the fit did not converge (D076), then the
    descriptive curve; the curve stops if a recency lies outside the bins (D070). A template
    never eligible in these rounds has no c_a (D077)."""
    seen = np.zeros(K, dtype=bool)
    for part in c.chunks():
        seen |= part.eligible.any(axis=0)
    templates = np.flatnonzero(seen).tolist()
    paper = fit(PaperForm(templates), c, deadline)
    gamma = paper["parameters"]["gamma"]
    out: dict[str, Any] = {
        "templates_left_out": [duolingo.TEMPLATES[a] for a in range(K) if not seen[a]],
        "paper_form": paper,
        "gamma_reading": read_gamma(gamma) if paper["converged"] else NOT_CONVERGED,
        "half_life": read_half_life(paper),
    }
    outside, per_bin = 0, np.zeros(len(EDGES) + 1, dtype=np.int64)
    for part in c.chunks():
        rec = part.recency[part.eligible].astype(np.float64)
        outside += int(np.sum(np.isfinite(rec) & ((rec < 0) | (rec > EDGES[-1]))))
        per_bin += np.bincount(recency_bin(rec), minlength=len(EDGES) + 1)
    if outside:
        out["curve"] = {"stopped": f"{outside} eligible recencies outside [0, 30] days"}
        return out
    curve = Curve([b for b in range(len(EDGES)) if per_bin[b]], templates)
    fitted = fit(curve, c, deadline)
    for name in curve.names:
        if name.startswith("beta"):
            p = fitted["parameters"][name]
            p["relative change"] = {
                "value": float(np.expm1(p["value"])),
                **{
                    kind: {
                        "ci_low": float(np.expm1(p[kind]["ci_low"])),
                        "ci_high": float(np.expm1(p[kind]["ci_high"])),
                    }
                    for kind in ("conservative", "independent")
                },
            }
    out["curve"] = fitted | {"eligible_per_bin": per_bin.tolist()}
    return out


# ---------------------------------------------------------------------------------------------
# Passes over the row groups


Job = tuple[int, Path, int, list[str]]


def _merge_moments(parts: Sequence[dict[str, DayMoments]]) -> dict[str, DayMoments]:
    out: dict[str, DayMoments] = {}
    for part in parts:
        for key, m in part.items():
            if key in out:
                out[key].merge(m)
            else:
                out[key] = m
    return out


def _sample(r: Rounds, kind: int, file: int, group: int, share: float) -> BoolArray:
    """Rows drawn into the outcome-model (kind 0) or DM and DR (kind 1) sample (purpose 10)."""
    out: BoolArray = stream(DUOLINGO_STREAM, kind, file, group).random(len(r)) < share
    return out


def _slices(n: int) -> list[slice]:
    """Slices of a row group, so that memory does not grow with its size."""
    return [slice(start, min(n, start + SLICE)) for start in range(0, n, SLICE)]


def train_job(job: Job) -> dict[str, Any]:
    """Logging check, arm sums, the outcome-model sample and the DL2 rounds of one training
    row group."""
    file, path, group, languages = job
    whole = read_rounds(path, group, languages)
    sample = _sample(whole, 0, file, group, TRAIN_SHARE)
    out: dict[str, Any] = {
        "logging": {},
        "template": np.zeros((8, K)),
        "ui": np.zeros((8, len(languages) * K)),
        "rounds": len(whole),
        "rewards": float(whole.reward.sum()),
    }
    xs, ys, completed = [], [], []
    for part in _slices(len(whole)):
        r, keep = whole.take(part), sample[part]
        cols = np.broadcast_to(np.arange(K), r.eligible.shape)
        out["logging"] = _merge_moments([out["logging"], logging_moments(r)])
        out["template"] += arm_sums(r, cols, K)
        out["ui"] += arm_sums(r, r.language[:, None] * K + cols, len(languages) * K)
        xs.append(features(r.take(keep), r.selected[keep]))
        ys.append(r.reward[keep])
        completed.append(Completed.of(r))
    return out | {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "completed": Completed.concat(completed),
    }


def _block(m: dict[str, DayMoments], key: str, r: Rounds, z: FloatArray) -> None:
    m.setdefault(key, DayMoments(z.shape[1])).add(r.day, z, r.j)


def policy_moments(
    r: Rounds, scores: Scores, model: lgb.Booster, sample: BoolArray
) -> tuple[dict[str, DayMoments], dict[str, DayMoments]]:
    """Moments of (r, w r, w) for every policy, of the RD pairs and of the grid against
    (0.017, 15) on every round; of (r, w r, w, DM, DR) and the RD pairs' DR on ``sample``.
    w = pi(a_t) / b_t with b_t = 1 / |A_t|."""
    pis = policies(r, scores)
    w = {name: logged(r, pi) * r.size for name, pi in pis.items()}
    y = r.reward
    m: dict[str, DayMoments] = {}
    for name in w:
        _block(m, name, r, np.column_stack([y, w[name] * y, w[name]]))
    for a, b in PAIRS:
        _block(m, f"{a} - {b}", r, np.column_stack([w[a] * y, w[a], w[b] * y, w[b]]))
    grid = {k: v * r.size for k, v in grid_choices(r, scores).items()}
    for (gamma, h), wg in grid.items():
        z = np.column_stack([wg * y, grid[(GAMMA, HALF_LIFE)] * y])
        _block(m, f"grid {gamma:g} {h:g}", r, z)

    s = r.take(sample)
    q = outcome_matrix(s, model)
    q_logged = logged(s, q)
    found: dict[str, DayMoments] = {}
    dr = {}
    for name, pi in pis.items():
        ws = w[name][sample]
        dm = (pi[sample] * q).sum(axis=1)
        dr[name] = dm + ws * (s.reward - q_logged)
        _block(found, name, s, np.column_stack([s.reward, ws * s.reward, ws, dm, dr[name]]))
    for a, b in PAIRS:
        _block(found, f"{a} - {b}", s, np.column_stack([dr[a], dr[b]]))
    return m, found


def test_job(job: tuple[Job, Scores, str]) -> dict[str, Any]:
    """Logging check, policy moments and the DL2 rounds of one test row group."""
    (file, path, group, languages), scores, model_text = job
    whole = read_rounds(path, group, languages)
    sample = _sample(whole, 1, file, group, TEST_SHARE)
    model = lgb.Booster(model_str=model_text)
    logging: dict[str, DayMoments] = {}
    moments: dict[str, DayMoments] = {}
    found: dict[str, DayMoments] = {}
    completed = []
    for part in _slices(len(whole)):
        r = whole.take(part)
        m, f = policy_moments(r, scores, model, sample[part])
        logging = _merge_moments([logging, logging_moments(r)])
        moments = _merge_moments([moments, m])
        found = _merge_moments([found, f])
        completed.append(Completed.of(r))
    return {
        "logging": logging,
        "moments": moments,
        "sample": found,
        "completed": Completed.concat(completed),
    }


# ---------------------------------------------------------------------------------------------
# Results


def _values(m: DayMoments, sample: DayMoments) -> dict[str, Any]:
    """IPS and SNIPS on every test round; IPS, DM and DR on the sample; the inconclusive flag
    of 14.1 over IPS, SNIPS, DM and DR (conservative intervals)."""
    out: dict[str, Any] = {
        "IPS": m.estimate(lambda mu: float(mu[1])),
        "SNIPS": m.estimate(lambda mu: float(mu[1] / mu[2])),
        "IPS_sample": sample.estimate(lambda mu: float(mu[1])),
        "DM": sample.estimate(lambda mu: float(mu[3])),
        "DR": sample.estimate(lambda mu: float(mu[4])),
    }
    bounds = [out[k]["conservative"] for k in ("IPS", "SNIPS", "DM", "DR")]
    out["inconclusive"] = bool(max(b["ci_low"] for b in bounds) > min(b["ci_high"] for b in bounds))
    return out


def dl1(moments: dict[str, DayMoments], sample: dict[str, DayMoments]) -> dict[str, Any]:
    """Lifts over the mean test reward against the paper's (METHODS 15, DL1)."""
    base = moments["Baseline"].mean()
    mean_reward = float(base[0])
    if base[1] != base[0]:
        raise AssertionError("IPS of the uniform policy differs from the mean test reward")
    differs_from_paper = abs(mean_reward - PAPER_BASELINE) > 0.00005
    rows: dict[str, Any] = {}
    for name in DL1:
        lift = moments[name].estimate(lambda mu: float(mu[1] / mu[0] - 1))
        c = lift["conservative"]
        replicated = c["ci_low"] <= PAPER_LIFT[name] <= c["ci_high"]
        rows[name] = {
            "lift": lift,
            "paper_lift": PAPER_LIFT[name],
            "label": "replicated" if replicated else "differs",
            "difference": lift["value"] - PAPER_LIFT[name],
            "levels": _values(moments[name], sample[name]),
        }
        if differs_from_paper:
            rows[name]["note"] = "the test data differ from the paper's"
    differ = [name for name, row in rows.items() if row["label"] == "differs"]
    return {
        "mean_test_reward": mean_reward,
        "baseline": _values(moments["Baseline"], sample["Baseline"]),
        "rows": rows,
        "answer": "replicated" if not differ else "differs",
        "rows_that_differ": differ,
    }


def _test(diff: dict[str, Any]) -> dict[str, Any]:
    """Two-sided normal test at level 0.025 on the conservative standard error."""
    se, value = diff["conservative"]["se"], diff["value"]
    p = float(2 * norm.sf(abs(value) / se)) if se > 0 else float(value == 0)
    if p <= 0.025:
        label = "RD better" if diff["value"] > 0 else "RD worse"
    else:
        label = "no detectable difference"
    return diff | {"p_value": p, "label": label}


def dl3(moments: dict[str, DayMoments], sample: dict[str, DayMoments], m: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prior_strength": m,
        "levels": {name: _values(moments[name], sample[name]) for name in DL3},
    }
    for key, (a, b) in zip(("DL3a", "DL3b"), PAIRS, strict=True):
        pair = f"{a} - {b}"
        ips = moments[pair].estimate(lambda mu: float(mu[0] - mu[2]))
        out[key] = {
            "comparison": pair,
            "IPS": _test(ips),
            "SNIPS": moments[pair].estimate(lambda mu: float(mu[0] / mu[1] - mu[2] / mu[3])),
            "DR": sample[pair].estimate(lambda mu: float(mu[0] - mu[1])),
        }
    return out


def grid(moments: dict[str, DayMoments]) -> dict[str, Any]:
    """DL2 secondary: test-round IPS of each grid point as a paired difference against
    (0.017, 15); descriptive."""
    out = {}
    for gamma in GRID_GAMMA:
        for h in GRID_HALF_LIFE:
            m = moments[f"grid {gamma:g} {h:g}"]
            out[f"{gamma:g} {h:g}"] = {
                "IPS": m.estimate(lambda mu: float(mu[0])),
                "difference": m.estimate(lambda mu: float(mu[0] - mu[1])),
            }
    return out


def run(processed: Path, processes: int) -> dict[str, Any]:
    """Both passes and the results; stops at the line of D074, counting the conversion's
    seconds first. Day numbers group the rows of both splits and the outcome model is not
    cross-fitted, so every training time must precede every test time (D075)."""
    start = time.perf_counter()
    report = json.loads((processed / duolingo.REPORT).read_text("utf-8"))
    train, test = report["train"], report["test"]
    if train["datetime_max"] > test["datetime_min"]:
        raise ValueError(
            f"training times {train['datetime_min']} to {train['datetime_max']} do not precede "
            f"test times {test['datetime_min']} to {test['datetime_max']} (D075)"
        )
    deadline = Deadline(
        duolingo.STOP_LINE_S, sum(report[split]["seconds"] for split in ("train", "test"))
    )
    deadline.check()
    languages = sorted(set(report["train"]["languages"]) | set(report["test"]["languages"]))

    def jobs(split: str) -> list[Job]:
        return [
            (i, path, g, languages)
            for i, path in enumerate(duolingo.files(processed, split))
            for g in range(pq.ParquetFile(path).num_row_groups)
        ]

    timings: dict[str, float] = {}
    # Workers inherit the thread limits set before they start (main)
    with mp.get_context("spawn").Pool(processes) as pool:
        trained = list(deadline.results(pool.imap(train_job, jobs("train"))))
        timings["train_pass_s"] = round(time.perf_counter() - start, 1)
        x = np.concatenate([t["x"] for t in trained])
        y = np.concatenate([t["y"] for t in trained])
        model = fit_outcome_model(x, y, processes)
        template = sum(t["template"] for t in trained)
        ui = sum(t["ui"] for t in trained)
        rounds = sum(t["rounds"] for t in trained)
        rewards = sum(t["rewards"] for t in trained)
        rd, m = shrunk_scores(template, rounds, rewards)
        scores = Scores(raw_scores(template), raw_scores(ui), rd)
        timings["fit_s"] = round(time.perf_counter() - start, 1)
        deadline.check()
        text = model.model_to_string()
        tested = list(
            deadline.results(pool.imap(test_job, [(job, scores, text) for job in jobs("test")]))
        )
        timings["test_pass_s"] = round(time.perf_counter() - start, 1)
    moments = _merge_moments([t["moments"] for t in tested])
    sample = _merge_moments([t["sample"] for t in tested])
    completed = Completed.concat([t.pop("completed") for t in trained + tested])
    logging = _merge_moments([t["logging"] for t in trained + tested])
    check = logging_check(logging)
    every_row = logging["A|all"]
    out: dict[str, Any] = {
        "tier": "A",
        "note": "Value of one notification under a template policy given the logged "
        "histories; volume fatigue, long-term and deployed values are outside these logs.",
        "rows": {split: report[split]["rows"] for split in ("train", "test")},
        "conversion": {
            split: {
                k: report[split][k]
                for k in (
                    "commit",
                    "seconds",
                    "datetime_min",
                    "datetime_max",
                    "j_mean",
                    "j_positive_share",
                    "one_eligible_share",
                    "n_days_min",
                    "n_days_max",
                    "n_days_outside_0_30",
                )
            }
            for split in ("train", "test")
        },
        "logging_check": check,
        # Mean j_t per day over all rows: the within-day factor of every interval is
        # 1 + 2 times the mean over the rows it uses
        "j_mean_by_day": {str(d): e[3] / e[0] for d, e in sorted(every_row.days.items())},
        "interpreted": check["uniform"],
        "outcome_model_rows": len(y),
        "scores": {
            "template": dict(zip(duolingo.TEMPLATES, scores.template.tolist(), strict=True)),
            "rd": dict(zip(duolingo.TEMPLATES, scores.rd.tolist(), strict=True)),
        },
        "DL1": dl1(moments, sample),
        "DL2": dl2(completed, deadline) | {"grid": grid(moments)},
        "DL3": dl3(moments, sample, m),
    }
    timings["total_s"] = round(time.perf_counter() - start, 1)
    return out | {"timings_s": timings}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse the Duolingo notification logs.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument(
        "--smoke", action="store_true", help="synthetic archives; writes under results/runs/smoke"
    )
    args = parser.parse_args(argv)
    one_thread_per_process()
    commit = code_commit(args.root)
    if args.smoke:
        with tempfile.TemporaryDirectory() as folder:
            raw, processed = Path(folder) / "raw", Path(folder) / "processed"
            rows = fixtures.duolingo_raw(raw, SMOKE_ROWS, row_group=10_000)
            duolingo.convert(
                raw, processed, rows, {s: (f"{s}-part-1",) for s in rows}, commit=commit
            )
            result = {"commit": commit} | run(processed, args.processes)
        target = args.root / paths.RESULTS / SMOKE / RESULTS_FILE
    else:
        result = {"commit": commit} | run(
            paths.processed_dir(args.root, "duolingo"), args.processes
        )
        target = args.root / paths.RESULTS / RESULTS_FILE
    write_json(target, result)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
