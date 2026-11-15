"""Long-horizon off-policy estimators and logging designs in the simulator (METHODS 14.7).

``python -m push_lab.ope.horizon`` logs three designs on the evaluation users at eight settings,
estimates the difference in active-day share between each of N, T_half and T2 and one push a
day (T) with per-day DR, PDIS, SMIS, FQE, arm means and a surrogate index, runs the targets
online for the truth, and writes ``results/ope_horizon.json``; each log's estimates and each
setting's truth runs go to ``results/runs/ope_horizon/`` (ignored by git). Everything is Tier C.
"""

import argparse
import multiprocessing as mp
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.stats import t as student_t

from push_lab.data import paths
from push_lab.ope import estimators as est
from push_lab.ope import simulator as sim
from push_lab.runtime import Deadline, code_commit, one_thread_per_process, write_json
from push_lab.sim import env, world
from push_lab.sim.env import Action, Observation
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import (
    ARM_STREAM,
    DAYS,
    FOLD_STREAM,
    RESAMPLE_STREAM,
    SLOTS,
    Params,
    slot_time,
    stream,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Targets, in the order that indexes the arms of E2 and E3
TARGETS = ("N", "T_half", "T", "T2")
BASE = "T"
DIFFS = ("N", "T_half", "T2")
PRIMARY_DIFFS = ("N", "T2")
WEEK = 7
WEEKS = DAYS // WEEK
SETTINGS: dict[str, Params] = {
    "control": sim.CONTROL,
    "reference": Params(),
    **{f"annoyance_{a:g}": Params(annoyance=a) for a in (0.0, 0.025, 0.075, 0.1)},
    **{f"fatigue_half_life_{h:g}": Params(fatigue_half_life=h) for h in (3.0, 14.0)},
}
DESIGNS = ("E1", "E2", "E3")
LOG_SEEDS = {
    "E1": tuple(range(5_001, 5_051)),
    "E2": tuple(range(6_001, 6_051)),
    "E3": tuple(range(7_001, 7_051)),
}
OTHER_LOGS = 10
SURROGATE_SEED = 7_000
ESTIMATORS = {
    "E1": ("DR_day", "PDIS", "SMIS", "FQE"),
    "E2": ("PDIS_week", "SMIS_week", "FQE_week"),
    "E3": ("arm_means", "surrogate"),
}
STATEFUL = ("SMIS", "FQE", "SMIS_week", "FQE_week")
# Primary estimators per design (METHODS 14.7, decision rules)
PRIMARY = {
    "E1": ("SMIS", "FQE"),
    "E2": ("PDIS_week", "SMIS_week", "FQE_week"),
    "E3": ("arm_means",),
}
STATES = ("S0", "S_oracle", "S_half", "S_double", "S_none")
CONTROL_STATES = STATES[1:]
# Settings where the state controls run, on the first OTHER_LOGS log seeds
STATE_SETTINGS = ("reference", "fatigue_half_life_3", "fatigue_half_life_14")
# Settings of the implementation check
CHECK_SETTINGS = ("control", "reference")
CHECK_ESTIMATORS = (("E2", "PDIS_week"), ("E3", "arm_means"))
TRUTH_RUNS, TRUTH_BASE = 200, 300_000
SAMPLES = 1_000
REFITS = 50
ROUNDS = 100
REGRESSION: dict[str, Any] = sim.PARAMS | {"objective": "regression"}
Z = sim.Z
# Twelve intervals of the implementation check, each at level 1 - 0.05 / 12, with Student t
# quantiles so that the level holds with 10 logs (D069)
CHECK_LEVEL = 1 - 0.05 / 12
# Validation reads coverage only with this many logs (METHODS 14.7, D069)
COVERAGE_LOGS = 50
SEPARATION = 0.8
ALL_USERS = 27_285
RESULTS_FILE = "ope_horizon.json"
RUNS = Path("runs") / "ope_horizon"
# Stop line: 1.5 times the 5,000 s projected in D071 (D066, D074)
STOP_LINE_S = 1.5 * 5_000
# Two log seeds per design, outputs apart from the registered runs (D050)
SMOKE = Path("runs") / "smoke" / "ope_horizon"
SMOKE_LOGS = 2

# SMIS bins: counts by lower edges, decayed volume as a share of its reference, return stock
DELIVERY_EDGES = (1, 4, 7, 10)
OPEN_EDGES = (1, 2)
ACTIVE_EDGES = (3, 5, 7)
VOLUME_EDGES = (0.25, 0.75, 1.25, 1.75)
RETURN_EDGES = (0.25, 1.0)


# ---------------------------------------------------------------------------------------------
# Estimators on rounds [round, user]


def pdis(w: FloatArray, r: FloatArray) -> tuple[FloatArray, dict[str, float]]:
    """Per-user sums of rho_t r_t, rho_t the product of the weights up to round t, the Kish
    effective size of the last round's rho and the largest rho of any round. The estimate is
    their sum over n H."""
    rho = np.cumprod(w, axis=0)
    return (rho * r).sum(axis=0), {
        "effective_size": est.effective_size(rho[-1]),
        "max_rho": float(rho.max()),
    }


def smis(
    state: IntArray,
    states: int,
    w: FloatArray,
    r: FloatArray,
    counts: FloatArray,
    trace: list[FloatArray] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """State-marginalized importance sampling (Xie et al. 2019, Eq. 3.1 and 3.2, divided by H)
    under user multiplicities ``counts`` [resample, user]: c_t = omega_t(s_t) w_t with
    omega_0 = 1, omega_{t+1}(s) the mean of c_t over the users in state s at round t + 1, and
    the estimate the sum of c_t r_t over n H. Returns the estimates and the masses
    (1/n) sum c_{H-1}, one per resample; ``trace`` collects omega_{t+1} [resample, state] of
    the states present."""
    rounds, users = w.shape
    n = counts.sum(axis=1)
    omega = np.ones(counts.shape)
    total = np.zeros(len(counts))
    for t in range(rounds):
        c = omega * w[t]
        total += (counts * c * r[t]).sum(axis=1)
        if t + 1 == rounds:
            return total / (n * rounds), (counts * c).sum(axis=1) / n
        member = sparse.csr_matrix(
            (np.ones(users), (np.arange(users), state[t + 1])), shape=(users, states)
        )
        mass = np.asarray(member.T @ (counts * c).T).T
        present = np.asarray(member.T @ counts.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(present > 0, mass / present, 0.0)
        if trace is not None:
            trace.append(np.where(present > 0, ratio, np.nan))
        omega = ratio[:, state[t + 1]]
    raise ValueError("no rounds")


# A target's action mixture: (probability, scalar or [round, user]; action features [user, f])
Mixture = Sequence[tuple[float | FloatArray, FloatArray]]


def fqe(
    x: FloatArray,
    a: FloatArray,
    r: FloatArray,
    target: Mixture,
    users: IntArray | None = None,
) -> FloatArray:
    """Fitted Q evaluation (Le et al. 2019, Algorithm 3) over a finite horizon: the value to go
    from round 0 of each of ``users`` (default all). ``x`` [round, user, feature] holds the state
    features, ``a`` [round, user, feature] the logged action's features, and ``target`` the
    actions the target mixes with their probabilities."""
    rounds, n = r.shape
    idx = np.arange(n) if users is None else users
    probs = [np.broadcast_to(np.asarray(p, dtype=np.float64), (rounds, n)) for p, _ in target]
    value = np.zeros(len(idx))
    for t in reversed(range(rounds)):
        y = r[t, idx] + value
        data = lgb.Dataset(np.column_stack([x[t, idx], a[t, idx]]), y)
        model = lgb.train(REGRESSION, data, num_boost_round=ROUNDS)
        value = np.zeros(len(idx))
        for p, (_, action) in zip(probs, target, strict=True):
            value += p[t, idx] * model.predict(np.column_stack([x[t, idx], action[idx]]))
    return value


def resample_counts(rng: np.random.Generator, users: int, samples: int = SAMPLES) -> FloatArray:
    """How often each user is drawn in each of ``samples`` resamples with replacement."""
    idx = rng.integers(0, users, (samples, users))
    out: FloatArray = np.stack([np.bincount(i, minlength=users) for i in idx]).astype(np.float64)
    return out


def group_means(y: FloatArray, group: IntArray, groups: int, counts: FloatArray) -> FloatArray:
    """Mean of ``y`` over the users of each group under ``counts`` [resample, user]: [resample,
    group]."""
    member = sparse.csr_matrix(
        (np.ones(len(group)), (np.arange(len(group)), group)), shape=(len(group), groups)
    )
    total = np.asarray(member.T @ (counts * y).T).T
    size = np.asarray(member.T @ counts.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out: FloatArray = total / size
    return out


# ---------------------------------------------------------------------------------------------
# Logging designs


@dataclass
class ArmRule:
    """Each user follows the rule of its arm, ``arms`` [week, user] indexing ``TARGETS``. The rules
    share one run seed, so their templates agree, and a user whose arms are all X receives the
    sends of X run from day 0 on that seed."""

    arms: IntArray
    rules: tuple[env.Policy, ...]

    def act(self, obs: Observation) -> Action:
        actions = [rule.act(obs) for rule in self.rules]
        arm = self.arms[obs.day // WEEK]
        return Action(
            np.choose(arm, [a.send for a in actions]),
            np.choose(arm, [a.template for a in actions]),
            np.choose(arm, [a.propensity for a in actions]),
        )


def draw_arms(design: str, seed: int, users: int) -> IntArray:
    """Arms [week, user]: drawn each week under E2, once per user under E3 (purpose 7)."""
    k = len(TARGETS)
    if design == "E2":
        return np.stack(
            [stream(ARM_STREAM, seed, w).integers(0, k, users) for w in range(WEEKS)]
        ).astype(np.int64)
    if design == "E3":
        once = stream(ARM_STREAM, seed).integers(0, k, users).astype(np.int64)
        return np.repeat(once[None, :], WEEKS, axis=0)
    raise KeyError(design)


@dataclass(frozen=True)
class Log:
    design: str
    seed: int
    out: env.Outcome
    explorer: EpsilonGreedy | None
    arms: IntArray | None


def simulate(ctx: sim.Context, params: Params, design: str, seed: int) -> Log:
    """One log of ``design`` on run seed ``seed``."""
    run = env.draws(ctx.users, seed)
    if design == "E1":
        explorer = EpsilonGreedy(ctx.best, seed=seed)
        out = env.simulate(ctx.pop, params, explorer, run, sim.WORLD_SEED)
        return Log(design, seed, out, explorer, None)
    arms = draw_arms(design, seed, ctx.users)
    rules = tuple(sim.target_policy(t, ctx, seed) for t in TARGETS)
    out = env.simulate(ctx.pop, params, ArmRule(arms, rules), run, sim.WORLD_SEED)
    return Log(design, seed, out, None, arms)


# ---------------------------------------------------------------------------------------------
# States at the start of each round


def round_starts(design: str) -> list[int]:
    """First day of each round: every day on E1, every week on E2."""
    return list(range(DAYS)) if design == "E1" else list(range(0, DAYS, WEEK))


def window(daily: IntArray, starts: Sequence[int]) -> IntArray:
    """Sums of ``daily`` [day, user] over the 7 days before each start (fewer at the start)."""
    cum = np.vstack([np.zeros((1, daily.shape[1]), dtype=np.int64), np.cumsum(daily, axis=0)])
    return np.stack([cum[s] - cum[max(0, s - WEEK)] for s in starts])


def decayed_volume(delivered: BoolArray, half_life: float, starts: Sequence[int]) -> FloatArray:
    """Delivered volume decayed with ``half_life`` at the start of each start day, accumulated
    step by step as ``env.simulate`` accumulates its fatigue volume."""
    volume = np.zeros(delivered.shape[2])
    now = 0.0
    at_start = []
    for d in range(DAYS):
        volume *= 0.5 ** ((float(d) - now) / half_life)
        now = float(d)
        at_start.append(volume.copy())
        for s in range(SLOTS):
            t = slot_time(d, s)
            volume *= 0.5 ** ((t - now) / half_life)
            now = t
            volume[delivered[d, s]] += 1.0
    return np.stack([at_start[d] for d in starts])


def return_stock(opened: BoolArray, half_life: float, starts: Sequence[int]) -> FloatArray:
    """Return stock at the start of each start day, updated at the end of each day as in
    ``env.simulate``."""
    stock = np.zeros(opened.shape[2])
    at_start = []
    for d in range(DAYS):
        at_start.append(stock.copy())
        stock = stock * 0.5 ** (1.0 / half_life) + opened[d].sum(axis=0)
    return np.stack([at_start[d] for d in starts])


@dataclass(frozen=True)
class State:
    """A state read at the start of each round: SMIS bin [round, user] out of ``bins``, and the
    features that replace the 7-day counts for FQE [round, user, feature]."""

    index: IntArray
    bins: int
    features: FloatArray


def _bin(x: npt.NDArray[Any], edges: Sequence[float]) -> IntArray:
    """Bin numbers; each bin includes its lower edge."""
    out: IntArray = np.digitize(x, edges).astype(np.int64)
    return out


def state(name: str, out: env.Outcome, params: Params, starts: Sequence[int]) -> State:
    """S0 (7-day deliveries, opens and active days) or one of its controls (METHODS 14.7)."""
    delivered = out.delivered.sum(axis=1).astype(np.int64)
    opened = out.opened.sum(axis=1).astype(np.int64)
    active = window(out.visited.astype(np.int64), starts)
    a_bin = _bin(active, ACTIVE_EDGES)
    a_bins = len(ACTIVE_EDGES) + 1
    first: npt.NDArray[Any]
    second: npt.NDArray[Any]
    if name == "S_none":
        return State(a_bin, a_bins, active[..., None].astype(np.float64))
    if name == "S0":
        first = window(delivered, starts)
        first_bin, first_bins = _bin(first, DELIVERY_EDGES), len(DELIVERY_EDGES) + 1
    else:
        h = params.fatigue_half_life * {"S_oracle": 1.0, "S_half": 0.5, "S_double": 2.0}[name]
        first = decayed_volume(out.delivered, h, starts)
        first_bin = _bin(first / world.reference_volume(h), VOLUME_EDGES)
        first_bins = len(VOLUME_EDGES) + 1
    if name == "S_oracle":
        second = return_stock(out.opened, params.return_half_life, starts)
        second_bin, second_bins = _bin(second, RETURN_EDGES), len(RETURN_EDGES) + 1
    else:
        second = window(opened, starts)
        second_bin, second_bins = _bin(second, OPEN_EDGES), len(OPEN_EDGES) + 1
    index = (first_bin * second_bins + second_bin) * a_bins + a_bin
    features = np.stack([first, second, active], axis=-1).astype(np.float64)
    return State(index, first_bins * second_bins * a_bins, features)


# ---------------------------------------------------------------------------------------------
# One log


@dataclass(frozen=True)
class Rounds:
    """Rounds of one log: reward [round, user], the weight of each target [round, user], the
    FQE features that do not depend on the state, the logged action's features and each
    target's action mixture."""

    reward: FloatArray
    weight: dict[str, FloatArray]
    base: FloatArray
    action: FloatArray
    mixture: dict[str, Mixture]


def covariates(ctx: sim.Context) -> FloatArray:
    """Policy visit rate, score of the first list item and largest policy profile share."""
    s = ctx.static
    return np.column_stack([s.visit_rate, s.score[:, 0], s.profile.max(axis=1)])


def _one_hot(index: IntArray, size: int) -> FloatArray:
    out: FloatArray = (index[..., None] == np.arange(size)).astype(np.float64)
    return out


def e1_rounds(ctx: sim.Context, log: Log, rows: sim.UserDays) -> Rounds:
    assert log.explorer is not None
    n = ctx.users
    b = sim.logging_probability(log.explorer, rows.bits, rows.user)
    weight = {
        t: (sim.target_probability(t, ctx, rows.bits, rows.user) / b).reshape(DAYS, n)
        for t in TARGETS
    }
    keep = [i for i in range(rows.x.shape[1]) if i != sim.CONTEXT.index("volume")]
    one = _one_hot(ctx.best, SLOTS)
    nothing = np.zeros((n, SLOTS))
    mixture: dict[str, Mixture] = {
        "N": [(1.0, nothing)],
        "T_half": [(0.5, one), (0.5, nothing)],
        "T": [(1.0, one)],
        "T2": [(1.0, one + _one_hot(ctx.second, SLOTS))],
    }
    return Rounds(
        log.out.visited.astype(np.float64),
        weight,
        rows.x[:, keep].reshape(DAYS, n, len(keep)),
        rows.bits.reshape(DAYS, n, SLOTS).astype(np.float64),
        mixture,
    )


def e2_rounds(ctx: sim.Context, log: Log) -> Rounds:
    assert log.arms is not None
    n, k = ctx.users, len(TARGETS)
    reward = log.out.visited.reshape(WEEKS, WEEK, n).mean(axis=1)
    weight = {t: k * (log.arms == i).astype(np.float64) for i, t in enumerate(TARGETS)}
    base = np.broadcast_to(covariates(ctx), (WEEKS, n, 3))
    mixture: dict[str, Mixture] = {
        t: [(1.0, _one_hot(np.full(n, i, dtype=np.int64), k))] for i, t in enumerate(TARGETS)
    }
    return Rounds(reward, weight, base, _one_hot(log.arms, k), mixture)


def _interval(value: float, boot: FloatArray) -> dict[str, Any]:
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"value": value, "ci_low": float(low), "ci_high": float(high)}


def _differences(values: dict[str, float], boot: dict[str, FloatArray]) -> dict[str, Any]:
    """D_X = V(X) - V(T) with percentile intervals from paired resamples."""
    return {
        x: _interval(values[x] - values[BASE], boot[x] - boot[BASE]) | {"interval": "bootstrap"}
        for x in DIFFS
    }


def _entry(values: dict[str, float], boot: dict[str, FloatArray], **extra: Any) -> dict[str, Any]:
    return {"values": values, "D": _differences(values, boot)} | extra


def _refit_entry(values: dict[str, float], refits: list[dict[str, float]]) -> dict[str, Any]:
    """D_X with +/- 1.96 standard deviations of the refitted D_X (FQE-week at the reference)."""
    d: dict[str, Any] = {}
    for x in DIFFS:
        value = values[x] - values[BASE]
        sd = float(np.std([r[x] - r[BASE] for r in refits], ddof=1))
        d[x] = {"value": value, "ci_low": value - Z * sd, "ci_high": value + Z * sd}
        d[x] |= {"interval": f"{len(refits)} refits"}
    return {"values": values, "D": d}


def surrogate_features(ctx: sim.Context, out: env.Outcome) -> FloatArray:
    """Week-one active days and opens with the three user covariates."""
    active = out.visited[:WEEK].sum(axis=0)
    opens = out.opened[:WEEK].sum(axis=(0, 1))
    return np.column_stack([active, opens, covariates(ctx)]).astype(np.float64)


def fit_surrogate(ctx: sim.Context, params: Params) -> lgb.Booster:
    """The surrogate index h, fitted on the E3 log kept apart (seed 7,000)."""
    log = simulate(ctx, params, "E3", SURROGATE_SEED)
    y = log.out.visited.mean(axis=0)
    data = lgb.Dataset(surrogate_features(ctx, log.out), y)
    return lgb.train(REGRESSION, data, num_boost_round=ROUNDS)


def estimate_log(
    ctx: sim.Context,
    params: Params,
    design: str,
    seed: int,
    states: Sequence[str] = ("S0",),
    refits: int = 0,
    surrogate: lgb.Booster | None = None,
) -> dict[str, Any]:
    """Every estimator of ``design`` with its D_X intervals from one log. ``states`` are those
    of SMIS and FQE; ``refits`` > 0 gives FQE-week refit intervals; E3 needs ``surrogate``. One
    set of user resamples (purpose 8) serves every estimator, state and target, so the two
    targets of each D_X are resampled together."""
    start = time.perf_counter()
    log = simulate(ctx, params, design, seed)
    counts = resample_counts(stream(RESAMPLE_STREAM, seed, 0), ctx.users)
    result: dict[str, Any] = {"design": design, "seed": seed}
    if design == "E3":
        assert surrogate is not None
        result |= _arm_estimates(ctx, log, counts, surrogate)
    else:
        result["estimates"] = _round_estimates(ctx, params, log, counts, states, refits)
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def _arm_estimates(
    ctx: sim.Context, log: Log, counts: FloatArray, surrogate: lgb.Booster
) -> dict[str, Any]:
    """Arm means and the surrogate index on an E3 log."""
    assert log.arms is not None
    arm, k = log.arms[0], len(TARGETS)
    share = log.out.visited.mean(axis=0)
    predicted = np.asarray(surrogate.predict(surrogate_features(ctx, log.out)), dtype=np.float64)
    found = {}
    for name, y in (("arm_means", share), ("surrogate", predicted)):
        point = group_means(y, arm, k, np.ones((1, ctx.users)))[0]
        boot = group_means(y, arm, k, counts)
        found[name] = _entry(
            {t: float(point[i]) for i, t in enumerate(TARGETS)},
            {t: boot[:, i] for i, t in enumerate(TARGETS)},
        )
    return {
        "estimates": found,
        "arm_users": {t: int((arm == i).sum()) for i, t in enumerate(TARGETS)},
    }


def _round_estimates(
    ctx: sim.Context,
    params: Params,
    log: Log,
    counts: FloatArray,
    states: Sequence[str],
    refits: int,
) -> dict[str, Any]:
    """PDIS, SMIS and FQE (and per-day DR on E1) on an E1 or E2 log."""
    n = ctx.users
    once = np.ones((1, n))
    starts = round_starts(log.design)
    found: dict[str, Any] = {}
    if log.design == "E1":
        rows = sim.user_days(log.out, ctx.static)
        rounds = e1_rounds(ctx, log, rows)
        found["DR_day"] = per_day_dr(ctx, log, rows, counts)
        names = ("PDIS", "SMIS", "FQE")
    else:
        rounds = e2_rounds(ctx, log)
        names = ("PDIS_week", "SMIS_week", "FQE_week")
    scale = n * len(starts)
    r = rounds.reward

    sums: dict[str, FloatArray] = {}
    diagnostics: dict[str, Any] = {}
    for t in TARGETS:
        sums[t], diagnostics[t] = pdis(rounds.weight[t], r)
    found[names[0]] = _entry(
        {t: float(sums[t].sum() / scale) for t in TARGETS},
        {t: counts @ sums[t] / scale for t in TARGETS},
        diagnostics=diagnostics,
    )

    for name in states:
        s = state(name, log.out, params, starts)
        values: dict[str, float] = {}
        boot: dict[str, FloatArray] = {}
        mass: dict[str, Any] = {}
        for t in TARGETS:
            point, point_mass = smis(s.index, s.bins, rounds.weight[t], r, once)
            values[t], mass[t] = float(point[0]), {"mass": float(point_mass[0])}
            boot[t] = smis(s.index, s.bins, rounds.weight[t], r, counts)[0]
        found[f"{names[1]}|{name}"] = _entry(values, boot, diagnostics=mass)

        x = np.concatenate([rounds.base, s.features], axis=2)
        v0 = {t: fqe(x, rounds.action, r, rounds.mixture[t]) for t in TARGETS}
        values = {t: float(v0[t].sum() / scale) for t in TARGETS}
        if refits:
            redone = []
            for i in range(refits):
                users = stream(RESAMPLE_STREAM, log.seed, 1, i).integers(0, n, n)
                redone.append(
                    {
                        t: float(fqe(x, rounds.action, r, rounds.mixture[t], users).sum() / scale)
                        for t in TARGETS
                    }
                )
            found[f"{names[2]}|{name}"] = _refit_entry(values, redone)
        else:
            boot = {t: counts @ v0[t] / scale for t in TARGETS}
            found[f"{names[2]}|{name}"] = _entry(values, boot)
    return found


def per_day_dr(
    ctx: sim.Context, log: Log, rows: sim.UserDays, counts: FloatArray
) -> dict[str, Any]:
    """Per-day DR of METHODS 14.3 with the visit reward: cross-fitted outcome model with folds
    from purpose 9, held fixed over the user resamples."""
    assert log.explorer is not None
    b = sim.logging_probability(log.explorer, rows.bits, rows.user)
    q_logged, q_every = sim.outcome_models(rows, "visit", stream(FOLD_STREAM, log.seed))
    values, boot, diagnostics = {}, {}, {}
    dr = est.ESTIMATORS.index("DR")
    for t in TARGETS:
        pi_every = np.stack(
            [
                sim.target_probability(t, ctx, np.broadcast_to(v, rows.bits.shape), rows.user)
                for v in sim.VECTORS
            ],
            axis=1,
        )
        weight = sim.target_probability(t, ctx, rows.bits, rows.user) / b
        logged = est.Logged(
            rows.reward["visit"], weight, q_logged, (pi_every * q_every).sum(axis=1), rows.user
        )
        sums = logged.sums()
        values[t] = float(est.from_sums(sums.sum(axis=0))[dr])
        boot[t] = est.from_sums(counts @ sums)[:, dr]
        diagnostics[t] = {
            "effective_size": est.effective_size(weight),
            "max_weight": float(weight.max()),
        }
    return _entry(values, boot, diagnostics=diagnostics)


# ---------------------------------------------------------------------------------------------
# Truth


def truth_run(ctx: sim.Context, params: Params, k: int) -> dict[str, float]:
    """Active-day share of each target online on run seed 300,000 + k, the same for all."""
    return {t: sim.truth(ctx, params, t, TRUTH_BASE + k)["visit"] for t in TARGETS}


def truth_summary(
    runs: list[dict[str, float]] | None, exact: dict[str, float] | None
) -> dict[str, Any]:
    """D_X with its standard error: the mean of paired differences over the runs, or exact."""
    out: dict[str, Any] = {"levels": {}, "D": {}}
    if exact is not None:
        out["method"] = "exact"
        out["levels"] = {t: {"value": exact[t], "se": 0.0} for t in TARGETS}
        diffs = {x: (exact[x] - exact[BASE], 0.0) for x in DIFFS}
    else:
        assert runs
        out["method"] = f"{len(runs)} paired online runs"
        root = np.sqrt(len(runs))
        for t in TARGETS:
            v = np.array([run[t] for run in runs])
            out["levels"][t] = {"value": float(v.mean()), "se": float(v.std(ddof=1) / root)}
        diffs = {}
        for x in DIFFS:
            d = np.array([run[x] - run[BASE] for run in runs])
            diffs[x] = (float(d.mean()), float(d.std(ddof=1) / root))
    for x, (value, se) in diffs.items():
        low, high = value - Z * se, value + Z * se
        sign = 1 if low > 0 else -1 if high < 0 else 0
        out["D"][x] = {"value": value, "se": se, "ci_low": low, "ci_high": high, "sign": sign}
    return out


def setting_truth(
    ctx: sim.Context, setting: str, runs: list[dict[str, float]] | None
) -> dict[str, Any]:
    """The truth of a setting: in the control the exact values of METHODS 14.3 (D044), where days
    are independent; elsewhere the paired online runs."""
    if setting == "control":
        exact = {t: sim.exact_truth(ctx, SETTINGS[setting], t)["visit"] for t in TARGETS}
        return truth_summary(None, exact)
    return truth_summary(runs, None)


# ---------------------------------------------------------------------------------------------
# Measures, labels and decision rules


def _side(low: float, high: float) -> int:
    return 1 if low > 0 else -1 if high < 0 else 0


def measures(estimates: list[dict[str, float]], truth: dict[str, Any]) -> dict[str, Any]:
    """Bias, RMSE, width, coverage, verdict, direction, separation and label of one estimator
    for one D_X over its logs (METHODS 14.7)."""
    values = np.array([e["value"] for e in estimates])
    m = len(values)
    v, v_se = truth["value"], truth["se"]
    bias = float(values.mean() - v)
    se = float(np.sqrt(values.var(ddof=1) / m + v_se**2))
    hits = sum(e["ci_low"] <= v <= e["ci_high"] for e in estimates)
    low, high = sim.wilson(hits, m)
    reasons = []
    if abs(bias) > Z * se:
        reasons.append("bias")
    if m >= COVERAGE_LOGS:
        if high < 0.95:
            reasons.append("under-coverage")
        if low > 0.95:
            reasons.append("over-coverage")
    validated = not reasons
    q = float(student_t.ppf(0.975, m - 1))
    half = q * float(values.std(ddof=1)) / np.sqrt(m)
    mean = float(values.mean())
    estimate_side = _side(mean - half, mean + half)
    truth_side = int(truth["sign"])
    out: dict[str, Any] = {
        "logs": m,
        "bias": {"value": bias, "ci_low": bias - Z * se, "ci_high": bias + Z * se},
        "rmse": float(np.sqrt(np.mean((values - v) ** 2))),
        "mean_interval_width": float(np.mean([e["ci_high"] - e["ci_low"] for e in estimates])),
        "coverage": {"value": hits / m, "ci_low": low, "ci_high": high},
        "coverage_read": m >= COVERAGE_LOGS,
        "verdict": "validated" if validated else "not validated",
        "reasons": reasons,
        "mean": {"value": mean, "ci_low": mean - half, "ci_high": mean + half},
    }
    if truth_side == 0:
        excluding = sum(e["ci_low"] > 0 or e["ci_high"] < 0 for e in estimates) / m
        out |= {"direction": None, "separation": None, "share_excluding_zero": excluding}
        out["label"] = out["verdict"]
        return out
    if estimate_side == 0:
        direction = "not resolved"
    else:
        direction = "correct" if estimate_side == truth_side else "reversed"
    spread = float(values.std(ddof=1))
    separation = float(np.mean(truth_side * values > Z * spread))
    if direction == "reversed":
        label = "reverses"
    elif validated and direction == "correct" and separation >= SEPARATION:
        label = "recovers the ordering"
    elif validated:
        label = "validated, does not separate"
    else:
        label = "not validated"
    return out | {"direction": direction, "separation": separation, "label": label}


def implementation_check(settings: dict[str, Any]) -> dict[str, Any]:
    """PDIS-week and arm means are unbiased by construction: each of their 12 bias intervals, at
    level 1 - 0.05 / 12, must contain 0, or nothing in METHODS 14.7 is interpreted. The
    quantile is Student t with the Welch degrees of freedom of the bias (D069)."""
    rows = []
    for setting in CHECK_SETTINGS:
        for design, name in CHECK_ESTIMATORS:
            for x in DIFFS:
                entry = settings[setting]["designs"][design][name][x]
                bias, se = entry["bias"]["value"], entry["bias_se"]
                q = float(student_t.ppf(1 - (1 - CHECK_LEVEL) / 2, entry["bias_df"]))
                low, high = bias - q * se, bias + q * se
                rows.append(
                    {
                        "setting": setting,
                        "estimator": name,
                        "D": x,
                        "bias": bias,
                        "ci_low": low,
                        "ci_high": high,
                        "contains_zero": bool(low <= 0 <= high),
                    }
                )
    return {
        "level": CHECK_LEVEL,
        "intervals": rows,
        "passed": all(r["contains_zero"] for r in rows),
    }


def _key(name: str, state: str) -> str:
    return f"{name}|{state}" if name in STATEFUL else name


def summarize_estimator(
    logs: list[dict[str, Any]], key: str, truth: dict[str, Any]
) -> dict[str, Any]:
    """Measures of one estimator (and state) for every D_X over ``logs``, with the standard
    error of the bias, its Welch degrees of freedom, and the diagnostics of each target as
    their mean, smallest and largest value over the logs."""
    out: dict[str, Any] = {}
    for x in DIFFS:
        estimates = [log["estimates"][key]["D"][x] for log in logs]
        entry = measures(estimates, truth["D"][x])
        values = np.array([e["value"] for e in estimates])
        spread = values.var(ddof=1) / len(values)
        truth_var = truth["D"][x]["se"] ** 2
        entry["bias_se"] = float(np.sqrt(spread + truth_var))
        entry["bias_df"] = float(
            (spread + truth_var) ** 2
            / (spread**2 / (len(values) - 1) + truth_var**2 / (TRUTH_RUNS - 1))
        )
        out[x] = entry
    found = [log["estimates"][key].get("diagnostics") for log in logs]
    if found[0]:
        out["diagnostics"] = {
            t: {
                name: {
                    "mean": float(np.mean([d[t][name] for d in found])),
                    "min": float(np.min([d[t][name] for d in found])),
                    "max": float(np.max([d[t][name] for d in found])),
                }
                for name in found[0][t]
            }
            for t in TARGETS
        }
    return out


def summarize(
    logs: dict[tuple[str, str, int], dict[str, Any]],
    truths: dict[str, dict[str, Any]],
    log_seeds: dict[str, dict[str, tuple[int, ...]]],
) -> dict[str, Any]:
    """Per setting, design and estimator: the measures of each D_X; then the implementation
    check, the primary labels, the state reading and the projected arm-mean width."""
    settings: dict[str, Any] = {}
    for setting in SETTINGS:
        truth = truths[setting]
        entry: dict[str, Any] = {"params": asdict(SETTINGS[setting]), "truth": truth}
        entry["designs"] = {}
        for design in DESIGNS:
            seeds = log_seeds[setting][design]
            chosen = [logs[(setting, design, s)] for s in seeds]
            first = chosen[: min(OTHER_LOGS, len(chosen))]
            found: dict[str, Any] = {}
            for name in ESTIMATORS[design]:
                if name not in STATEFUL:
                    found[name] = summarize_estimator(chosen, name, truth)
                    continue
                found[_key(name, "S0")] = summarize_estimator(chosen, _key(name, "S0"), truth)
                if setting in STATE_SETTINGS:
                    for s in CONTROL_STATES:
                        found[_key(name, s)] = summarize_estimator(first, _key(name, s), truth)
                    if len(chosen) > len(first):
                        # The state reading compares S0 on the logs the controls ran on (D069)
                        key = f"{_key(name, 'S0')}|first_{len(first)}"
                        found[key] = summarize_estimator(first, _key(name, "S0"), truth)
            entry["designs"][design] = found
        settings[setting] = entry
    return {
        "settings": settings,
        "implementation_check": implementation_check(settings),
        "primary": primary(settings),
        "state_reading": state_reading(settings),
        "arm_mean_projection": arm_mean_projection(logs, log_seeds),
    }


def primary(settings: dict[str, Any]) -> dict[str, Any]:
    """The 12 primary labels at the reference with S0, and whether each design recovers the
    ordering: one of its primary estimators recovers both D_N and D_T2."""
    out: dict[str, Any] = {}
    for design, names in PRIMARY.items():
        found = settings["reference"]["designs"][design]
        labels = {
            name: {x: found[_key(name, "S0")][x]["label"] for x in PRIMARY_DIFFS} for name in names
        }
        recovers = any(
            all(label == "recovers the ordering" for label in by_x.values())
            for by_x in labels.values()
        )
        out[design] = {"labels": labels, "recovers the ordering": recovers}
    return out


def _unbiased_and_correct(m: dict[str, Any]) -> bool:
    b = m["bias"]
    return bool(b["ci_low"] <= 0 <= b["ci_high"]) and m["direction"] == "correct"


def state_reading(settings: dict[str, Any]) -> dict[str, Any]:
    """Whether SMIS and FQE depend on knowing the simulator's memory: with S-oracle the bias
    interval contains 0 and the direction is correct, with none of S0, S-half and S-double. At
    the reference S0 is read on the logs the controls ran on (D069)."""
    out: dict[str, Any] = {}
    for setting in STATE_SETTINGS:
        out[setting] = {}
        for design in ("E1", "E2"):
            found = settings[setting]["designs"][design]
            for name in ESTIMATORS[design]:
                if name not in STATEFUL:
                    continue
                s0 = _key(name, "S0")
                if f"{s0}|first_{OTHER_LOGS}" in found:
                    s0 = f"{s0}|first_{OTHER_LOGS}"
                keys = {"S0": s0} | {s: _key(name, s) for s in CONTROL_STATES}
                reading = {}
                for x in DIFFS:
                    holds = {s: _unbiased_and_correct(found[k][x]) for s, k in keys.items()}
                    others = [holds[s] for s in ("S0", "S_half", "S_double")]
                    reading[x] = holds | {
                        "depends on knowing the memory": holds["S_oracle"] and not any(others)
                    }
                out[setting][name] = reading
    return out


def arm_mean_projection(
    logs: dict[tuple[str, str, int], dict[str, Any]],
    log_seeds: dict[str, dict[str, tuple[int, ...]]],
) -> dict[str, Any]:
    """Mean half-width of the arm-mean intervals at the reference, projected to all users by
    the square root of the ratio of users."""
    out: dict[str, Any] = {}
    chosen = [logs[("reference", "E3", s)] for s in log_seeds["reference"]["E3"]]
    users = sum(chosen[0]["arm_users"].values())
    for x in DIFFS:
        intervals = [log["estimates"]["arm_means"]["D"][x] for log in chosen]
        half = float(np.mean([(i["ci_high"] - i["ci_low"]) / 2 for i in intervals]))
        out[x] = {"half_width": half, "projected": half * float(np.sqrt(users / ALL_USERS))}
    return out | {"users": users, "all_users": ALL_USERS}


# ---------------------------------------------------------------------------------------------
# Running


def design_seeds(logs: int | None = None) -> dict[str, dict[str, tuple[int, ...]]]:
    """Log seeds per setting and design: 50 at the reference and 10 elsewhere, or the first
    ``logs`` everywhere."""
    out: dict[str, dict[str, tuple[int, ...]]] = {}
    for setting in SETTINGS:
        size = len(LOG_SEEDS["E1"]) if setting == "reference" else OTHER_LOGS
        if logs is not None:
            size = logs
        out[setting] = {d: LOG_SEEDS[d][:size] for d in DESIGNS}
    return out


def jobs(seeds: dict[str, dict[str, tuple[int, ...]]]) -> list[tuple[Any, ...]]:
    """Every log job (setting, design, seed, states, refits) and truth job (setting, k),
    heaviest first."""
    out: list[tuple[Any, ...]] = []
    for setting, by_design in seeds.items():
        for design, design_seeds_ in by_design.items():
            for i, seed in enumerate(design_seeds_):
                states: tuple[str, ...] = ("S0",)
                if design != "E3" and setting in STATE_SETTINGS and i < OTHER_LOGS:
                    states += CONTROL_STATES
                refits = REFITS if design == "E2" and setting == "reference" else 0
                out.append(("log", setting, design, seed, states, refits))
    out.sort(key=lambda j: (-j[5] * len(j[4]), -len(j[4]), j[2]))
    for setting in SETTINGS:
        if setting != "control":
            out += [("truth", setting, k) for k in range(TRUTH_RUNS)]
    return out


_CONTEXTS: dict[str, sim.Context] = {}
_SURROGATES: dict[str, lgb.Booster] = {}


def context(setting: str) -> sim.Context:
    return _CONTEXTS["control" if setting == "control" else "reference"]


def _init(root: Path) -> None:
    _CONTEXTS.update(sim.load_contexts(root))


def _work(job: tuple[Any, ...]) -> tuple[tuple[Any, ...], Any]:
    if job[0] == "truth":
        _, setting, k = job
        return job, truth_run(context(setting), SETTINGS[setting], k)
    _, setting, design, seed, states, refits = job
    ctx, params = context(setting), SETTINGS[setting]
    surrogate = None
    if design == "E3":
        if setting not in _SURROGATES:
            _SURROGATES[setting] = fit_surrogate(ctx, params)
        surrogate = _SURROGATES[setting]
    result = estimate_log(ctx, params, design, seed, states, refits, surrogate)
    return job, result | {"setting": setting}


def run(root: Path, processes: int, logs: int | None = None, runs: Path = RUNS) -> dict[str, Any]:
    """Every log and truth run, then the summary; stops at the line of D074 while it waits for
    a worker."""
    start = time.perf_counter()
    deadline = Deadline(STOP_LINE_S)
    seeds = design_seeds(logs)
    folder = root / paths.RESULTS / runs
    work = jobs(seeds)
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    truth_runs: dict[str, dict[int, dict[str, float]]] = {s: {} for s in SETTINGS}
    with mp.get_context("spawn").Pool(processes, _init, (root,)) as pool:
        for count, (job, result) in enumerate(
            deadline.results(pool.imap_unordered(_work, work)), 1
        ):
            if job[0] == "truth":
                truth_runs[job[1]][job[2]] = result
            else:
                _, setting, design, seed, *_ = job
                results[(setting, design, seed)] = result
                write_json(folder / f"{setting}_{design}_{seed}.json", result, indent=None)
            print(f"{count}/{len(work)} {' '.join(map(str, job[:4]))}", flush=True)
    contexts = sim.load_contexts(root)
    truths: dict[str, Any] = {}
    for setting in SETTINGS:
        ordered = None
        if setting != "control":
            ordered = [truth_runs[setting][k] for k in range(TRUTH_RUNS)]
            write_json(folder / f"truth_{setting}.json", ordered, indent=None)
        ctx = contexts["control" if setting == "control" else "reference"]
        truths[setting] = setting_truth(ctx, setting, ordered)
    return {
        "tier": "C",
        "note": "Long-horizon estimates of differences in active-day share from simulator logs, "
        "against simulator truth; nothing here is Tier B.",
        "users": contexts["reference"].users,
        "world_seed": sim.WORLD_SEED,
        "log_seeds": {s: {d: list(v) for d, v in by.items()} for s, by in seeds.items()},
        "truth_runs": TRUTH_RUNS,
        **summarize(results, truths, seeds),
        "timings_s": round(time.perf_counter() - start, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Long-horizon OPE in the simulator.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two log seeds per design; writes under results/runs/smoke",
    )
    args = parser.parse_args(argv)
    one_thread_per_process()
    commit = code_commit(args.root)
    if args.smoke:
        result = {"commit": commit} | run(args.root, args.processes, SMOKE_LOGS, SMOKE)
        target = args.root / paths.RESULTS / SMOKE / RESULTS_FILE
    else:
        result = {"commit": commit} | run(args.root, args.processes)
        target = args.root / paths.RESULTS / RESULTS_FILE
    write_json(target, result)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
