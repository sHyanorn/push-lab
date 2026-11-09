"""Off-policy estimates on simulator logs checked against simulator truth (METHODS 14.3).

``python -m push_lab.ope.simulator`` logs the exploration policy on the evaluation users for
every log seed of each setting, estimates every target's value per user-day, runs every
target online for the truth, and writes ``results/ope_simulator_amended.json``; each log's
estimates go to ``results/runs/ope/`` (ignored by git). Everything is Tier C.

The first run (``results/ope_simulator.json``, D043) took every target's truth from the same 20
run seeds; this run uses the exact value where the setting allows it and 200 independent run
seeds per target otherwise (D044).
"""

import argparse
import multiprocessing as mp
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from push_lab.data import paths
from push_lab.ope import estimators as est
from push_lab.policy import rules
from push_lab.policy.features import NAMES, History, Static
from push_lab.runtime import code_commit, one_thread_per_process, write_json
from push_lab.sim import env, inputs, world
from push_lab.sim.env import Action, Observation
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import DAYS, SLOTS, Params

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

TARGETS = ("T", "T_half", "R", "N", "F", "T2")
REWARDS = ("visit", "opens")
LOG_SEEDS = tuple(range(1_001, 1_051))
# Independent truth seeds per target (D044): TRUTH_BASE + 1,000 * target index + k
TRUTH_RUNS, TRUTH_BASE = 200, 100_000
RESULTS_FILE = "ope_simulator_amended.json"
RUNS = Path("runs") / "ope"
# Two log seeds, outputs apart from the registered runs (D050)
SMOKE = Path("runs") / "smoke" / "ope"
SMOKE_HELP = "two log seeds; writes under results/runs/smoke for pipeline checks only"
# 97.5% quantile of the standard normal, for two-sided 95% intervals
Z = 1.959964
WORLD_SEED = 10_100
FOLDS = 2
PER_SLOT = 1 / SLOTS
CONTROL = Params(fatigue_elasticity=0.0, novelty=0.0, opt_out=0.0, return_boost=0.0, annoyance=0.0)
SETTINGS = {"control": CONTROL, "reference": Params()}
# Base model of METHODS Section 8, one thread per process
PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "num_threads": 1,
    "verbose": -1,
}
ROUNDS = 200
OBJECTIVE = {"visit": "binary", "opens": "regression"}
CONTEXT = (
    "visit_rate",
    "next_score",
    "day",
    "visit_share",
    "visited_yesterday",
    "opens",
    "days_since_open",
    "days_since_delivery",
    "volume",
)
# Every send vector over the four slots, as rows of bits
VECTORS: BoolArray = ((np.arange(2**SLOTS)[:, None] >> np.arange(SLOTS)) & 1).astype(bool)


@dataclass(frozen=True)
class Context:
    """Evaluation users of one setting with what the policies may read."""

    pop: env.Population
    static: Static
    best: IntArray
    second: IntArray
    fixed: int

    @property
    def users(self) -> int:
        return self.pop.users


def make_context(pop: env.Population, static: Static, control: bool) -> Context:
    """In the control, each user's p0 is held at the mean of the first ``DAYS`` list items, so
    no carryover through the list position remains."""
    if control:
        flat = np.repeat(pop.p0[:, :DAYS].mean(axis=1, keepdims=True), pop.p0.shape[1], axis=1)
        pop = replace(pop, p0=flat)
    order = np.argsort(-static.profile, axis=1, kind="stable")
    fixed = int(static.profile.mean(axis=0).argmax())
    return Context(pop, static, order[:, 0], order[:, 1], fixed)


@dataclass
class TwoSlots:
    """One push at each of the two best slots every day."""

    best: IntArray
    second: IntArray
    run_seed: int

    def act(self, obs: Observation) -> Action:
        send = (self.best == obs.slot) | (self.second == obs.slot)
        n = len(send)
        return Action(send, rules.templates(self.run_seed, obs.day, obs.slot, n), np.ones(n))


def target_policy(target: str, ctx: Context, seed: int) -> env.Policy:
    n = ctx.users
    if target == "T":
        return rules.Daily(ctx.best, seed)
    if target == "T_half":
        return rules.Daily(ctx.best, seed, half=True)
    if target == "R":
        return rules.RandomSlots(n, seed)
    if target == "N":
        return rules.Never(n, seed)
    if target == "F":
        return rules.Daily(np.full(n, ctx.fixed, dtype=np.int64), seed)
    if target == "T2":
        return TwoSlots(ctx.best, ctx.second, seed)
    raise KeyError(target)


def target_probability(target: str, ctx: Context, bits: BoolArray, user: IntArray) -> FloatArray:
    """pi(a | x) of send vectors ``bits`` [row, slot] for the rows' users."""
    slots = np.arange(SLOTS)
    one = slots[None, :] == ctx.best[user][:, None]
    nothing = (~bits).all(axis=1)
    out: FloatArray
    if target == "T":
        out = (bits == one).all(axis=1).astype(np.float64)
    elif target == "T_half":
        out = 0.5 * (bits == one).all(axis=1) + 0.5 * nothing
    elif target == "R":
        out = np.prod(np.where(bits, PER_SLOT, 1 - PER_SLOT), axis=1)
    elif target == "N":
        out = nothing.astype(np.float64)
    elif target == "F":
        out = (bits == (slots == ctx.fixed)[None, :]).all(axis=1).astype(np.float64)
    elif target == "T2":
        two = one | (slots[None, :] == ctx.second[user][:, None])
        out = (bits == two).all(axis=1).astype(np.float64)
    else:
        raise KeyError(target)
    return out


def logging_probability(explorer: EpsilonGreedy, bits: BoolArray, user: IntArray) -> FloatArray:
    """b(a | x) = prod over slots of p_s^a (1 - p_s)^(1 - a), p_s from the logging policy."""
    p = np.stack([explorer.send_probability(s)[user] for s in range(SLOTS)], axis=1)
    out: FloatArray = np.prod(np.where(bits, p, 1 - p), axis=1)
    return out


@dataclass(frozen=True)
class UserDays:
    """One row per (day, user) of a logged run."""

    x: FloatArray
    bits: BoolArray
    reward: dict[str, FloatArray]
    user: IntArray


def user_days(out: env.Outcome, static: Static) -> UserDays:
    """Context at the start of each day from the logged history, the day's send vector and
    the day's rewards."""
    n = len(out.opted_out_at)
    history = History(n)
    xs, bits, visits, opens, users = [], [], [], [], []
    columns = [NAMES.index(c) for c in CONTEXT]
    best = static.profile.argmax(axis=1)
    for d in range(DAYS):
        f = history.features(static)[:, 0, columns]
        opted = (out.opted_out_at >= 0) & (out.opted_out_at < d * SLOTS)
        xs.append(np.column_stack([f, static.profile, best, opted]))
        bits.append(out.send[d].T)
        visits.append(out.visited[d].astype(np.float64))
        opens.append(out.opened[d].sum(axis=0).astype(np.float64))
        users.append(np.arange(n))
        history.end_day(out.delivered[d], out.opened[d], out.visited[d])
    return UserDays(
        np.concatenate(xs),
        np.concatenate(bits),
        {"visit": np.concatenate(visits), "opens": np.concatenate(opens)},
        np.concatenate(users).astype(np.int64),
    )


def outcome_models(rows: UserDays, reward: str, seed: int) -> tuple[FloatArray, FloatArray]:
    """Cross-fitted q(x, a) at the logged vector [row] and at every vector [row, vector]."""
    fold = np.random.default_rng(seed).permutation(int(rows.user.max()) + 1)[rows.user] % FOLDS
    y = rows.reward[reward]
    logged = np.empty(len(y))
    every = np.empty((len(y), len(VECTORS)))
    params = PARAMS | {"objective": OBJECTIVE[reward]}
    for f in range(FOLDS):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        x = np.column_stack([rows.x[train], rows.bits[train]])
        model = lgb.train(params, lgb.Dataset(x, y[train]), num_boost_round=ROUNDS)
        logged[test] = model.predict(np.column_stack([rows.x[test], rows.bits[test]]))
        for v, vector in enumerate(VECTORS):
            bits = np.broadcast_to(vector, (len(test), SLOTS))
            every[test, v] = model.predict(np.column_stack([rows.x[test], bits]))
    return logged, every


def estimate_log(ctx: Context, params: Params, seed: int) -> dict[str, dict[str, Any]]:
    """Every estimator's value and interval for every target and reward from one log."""
    explorer = EpsilonGreedy(ctx.best, seed=seed)
    out = env.simulate(ctx.pop, params, explorer, env.draws(ctx.users, seed), WORLD_SEED)
    rows = user_days(out, ctx.static)
    b = logging_probability(explorer, rows.bits, rows.user)
    result: dict[str, dict[str, Any]] = {}
    for reward in REWARDS:
        q_logged, q_every = outcome_models(rows, reward, seed)
        for target in TARGETS:
            pi_logged = target_probability(target, ctx, rows.bits, rows.user)
            pi_every = np.stack(
                [
                    target_probability(target, ctx, np.broadcast_to(v, rows.bits.shape), rows.user)
                    for v in VECTORS
                ],
                axis=1,
            )
            logged = est.Logged(
                rows.reward[reward],
                pi_logged / b,
                q_logged,
                (pi_every * q_every).sum(axis=1),
                rows.user,
            )
            result.setdefault(target, {})[reward] = est.evaluate(logged, seed=seed)
    return result


def truth(ctx: Context, params: Params, target: str, seed: int) -> dict[str, float]:
    """Mean reward per user-day of one online run of the target."""
    policy = target_policy(target, ctx, seed)
    out = env.simulate(ctx.pop, params, policy, env.draws(ctx.users, seed), WORLD_SEED)
    return {"visit": float(out.visited.mean()), "opens": float(out.opened.sum(axis=1).mean())}


def truth_seeds(target: str) -> range:
    first = TRUTH_BASE + 1_000 * TARGETS.index(target)
    return range(first, first + TRUTH_RUNS)


def exact_truth(ctx: Context, params: Params, target: str) -> dict[str, float]:
    """Expected reward per user-day where no day depends on another (the control): opens at
    each slot are independent given a uniform template, so the visit probability of a day with
    send vector a is 1 - (1 - v) * prod over sent slots of (1 - P(open))."""
    if params.fatigue_elasticity or params.annoyance or params.opt_out or params.return_boost:
        raise ValueError("the exact truth needs a setting without carryover")
    if params.novelty or np.ptp(ctx.pop.p0, axis=1).any():
        raise ValueError("the exact truth needs no novelty and a flat p0 per user")
    timing = world.timing_multipliers(
        ctx.pop.profile, world.solve_eta(ctx.pop.profile, params.timing_lift)
    )
    copy = world.copy_effects(params.copy_spread, WORLD_SEED, world.TEMPLATES)
    p0 = ctx.pop.p0[:, 0]
    rate = params.base_rate * p0[:, None, None] * timing[:, :, None] * copy[None, None, :]
    open_p = np.minimum(1.0, rate).mean(axis=2)
    v = np.minimum(1.0, ctx.pop.visit_rate)
    user = np.arange(ctx.users)
    visit, opens = np.zeros(ctx.users), np.zeros(ctx.users)
    for vector in VECTORS:
        bits = np.broadcast_to(vector, (ctx.users, SLOTS))
        pi = target_probability(target, ctx, bits, user)
        no_open = np.prod(np.where(bits, 1 - open_p, 1.0), axis=1)
        visit += pi * (1 - (1 - v) * no_open)
        opens += pi * np.where(bits, open_p, 0.0).sum(axis=1)
    return {"visit": float(visit.mean()), "opens": float(opens.mean())}


def wilson(hits: int, n: int, z: float = Z) -> tuple[float, float]:
    p = hits / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def validate(estimates: list[dict[str, Any]], v: float, v_se: float) -> dict[str, Any]:
    """Bias, RMSE, coverage and the verdict of METHODS 14.4 for one estimator, against a truth
    ``v`` with standard error ``v_se`` (0 when exact)."""
    values = np.array([e["value"] for e in estimates])
    bias = values.mean() - v
    se = np.sqrt(values.var(ddof=1) / len(values) + v_se**2)
    hits = sum(e["ci_low"] <= v <= e["ci_high"] for e in estimates)
    low, high = wilson(hits, len(estimates))
    reasons = []
    if high < 0.95:
        reasons.append("under-coverage")
    if low > 0.95:
        reasons.append("over-coverage")
    if abs(bias) > Z * se:
        reasons.append("bias")
    return {
        "bias": {
            "value": float(bias),
            "ci_low": float(bias - Z * se),
            "ci_high": float(bias + Z * se),
        },
        "rmse": float(np.sqrt(np.mean((values - v) ** 2))),
        "coverage": {"value": hits / len(estimates), "ci_low": low, "ci_high": high},
        "mean_interval_width": float(np.mean([e["ci_high"] - e["ci_low"] for e in estimates])),
        "verdict": "validated" if not reasons else "not validated",
        "reasons": reasons,
    }


def truth_summary(ctx: Context, params: Params, target: str, exact: bool) -> dict[str, Any]:
    """Per reward: the truth, its standard error and how it was obtained."""
    if exact:
        exact_values = exact_truth(ctx, params, target)
        return {r: {"value": exact_values[r], "se": 0.0, "method": "exact"} for r in REWARDS}
    runs = [truth(ctx, params, target, seed) for seed in truth_seeds(target)]
    out: dict[str, Any] = {}
    for r in REWARDS:
        values = np.array([x[r] for x in runs])
        out[r] = {
            "value": float(values.mean()),
            "se": float(values.std(ddof=1) / np.sqrt(len(values))),
            "method": f"{len(values)} independent online runs",
        }
    return out


def summarize(
    logs: dict[int, dict[str, dict[str, Any]]], truths: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in TARGETS:
        out[target] = {}
        for reward in REWARDS:
            t = truths[target][reward]
            per_log = [logs[s][target][reward] for s in sorted(logs)]
            entry: dict[str, Any] = {
                "truth": t,
                "effective_size_median": float(np.median([p["effective_size"] for p in per_log])),
                "max_weight": float(max(p["max_weight"] for p in per_log)),
            }
            for name in est.ESTIMATORS:
                estimates = [p["estimates"][name] for p in per_log]
                entry[name] = validate(estimates, t["value"], t["se"])
            out[target][reward] = entry
    return out


_CONTEXTS: dict[str, Context] = {}


def load_contexts(root: Path) -> dict[str, Context]:
    inp = inputs.Inputs.load(paths.processed_dir(root, "sim") / inputs.CACHE)
    users = world.quick_sample(len(inp.user_ids))
    pop = env.Population(
        inp.visit_rate[users], inp.profile_sim[users], inp.lists["cascade"].p0[users]
    )
    static = Static(
        inp.profile_policy[users], inp.visit_rate_policy[users], inp.lists["cascade"].score[users]
    )
    return {name: make_context(pop, static, name == "control") for name in SETTINGS}


def _init(root: Path) -> None:
    _CONTEXTS.update(load_contexts(root))


def _log_work(job: tuple[str, int]) -> tuple[str, int, dict[str, dict[str, Any]]]:
    setting, seed = job
    return setting, seed, estimate_log(_CONTEXTS[setting], SETTINGS[setting], seed)


def run(
    root: Path, processes: int, log_seeds: Sequence[int] = LOG_SEEDS, runs: Path = RUNS
) -> dict[str, Any]:
    start = time.perf_counter()
    jobs = [(s, seed) for s in SETTINGS for seed in log_seeds]
    logs: dict[str, dict[int, dict[str, dict[str, Any]]]] = {s: {} for s in SETTINGS}
    with mp.get_context("spawn").Pool(processes, _init, (root,)) as pool:
        for count, (setting, seed, result) in enumerate(pool.imap_unordered(_log_work, jobs), 1):
            logs[setting][seed] = result
            saved = root / paths.RESULTS / runs / f"{setting}_{seed}.json"
            write_json(saved, result, indent=None)
            print(f"{count}/{len(jobs)} {setting} log {seed}", flush=True)
    contexts = load_contexts(root)
    out: dict[str, Any] = {
        "tier": "C",
        "note": "Estimates on simulator logs checked against simulator truth; nothing here is "
        "Tier B.",
        "users": contexts["reference"].users,
        "log_seeds": len(log_seeds),
        "truth": "exact in the control; 200 independent online runs per target otherwise",
        "world_seed": WORLD_SEED,
    }
    for setting, params in SETTINGS.items():
        ctx = contexts[setting]
        exact = setting == "control"
        truths = {t: truth_summary(ctx, params, t, exact) for t in TARGETS}
        out[setting] = summarize(logs[setting], truths)
    out["timings_s"] = round(time.perf_counter() - start, 1)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check OPE on simulator logs against truth.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help=SMOKE_HELP)
    args = parser.parse_args(argv)
    one_thread_per_process()
    commit = code_commit(args.root)
    if args.smoke:
        result = {"commit": commit} | run(args.root, args.processes, LOG_SEEDS[:2], SMOKE)
        target = args.root / paths.RESULTS / SMOKE / RESULTS_FILE
    else:
        result = {"commit": commit} | run(args.root, args.processes)
        target = args.root / paths.RESULTS / RESULTS_FILE
    write_json(target, result)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
