"""Sensitivity sweep of the simulator OPE check over the carryover parameters (METHODS 14.6).

``python -m push_lab.ope.carryover`` runs the check of METHODS 14.3 at every point of a
one-at-a-time grid over annoyance, fatigue, opt-out, return and novelty, with fresh log seeds,
and writes ``results/ope_carryover.json``; each log's estimates go to
``results/runs/ope_carryover/`` (ignored by git). Everything is Tier C.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from push_lab.data import paths
from push_lab.ope import simulator as sim
from push_lab.policy import compare, sweep
from push_lab.sim.world import RANGES, Params

FloatArray = npt.NDArray[np.float64]

NAMES = ("annoyance", "fatigue_elasticity", "opt_out", "return_boost", "novelty")
POINTS = 5
LOG_SEEDS = tuple(range(3_001, 3_011))
TRUTH_RUNS = 100
FOCUS = ("T_half", "N", "T2")
RESAMPLES, RESAMPLE_SEED = 10_000, 0
RESULTS_FILE = "ope_carryover.json"
RUNS = Path("runs") / "ope_carryover"


def settings() -> list[dict[str, Any]]:
    """Distinct grid points, each with the (name, index) pairs it stands for."""
    reference = asdict(Params())
    found: dict[tuple[float, ...], dict[str, Any]] = {}
    for name in NAMES:
        for i, value in enumerate(np.linspace(*RANGES[name], POINTS)):
            values = reference | {name: float(value)}
            entry = found.setdefault(tuple(values.values()), {"values": values, "grid": []})
            entry["grid"].append([name, i])
    return [entry | {"id": i} for i, entry in enumerate(found.values())]


def truth_runs(ctx: sim.Context, params: Params, target: str) -> dict[str, list[float]]:
    runs = [sim.truth(ctx, params, target, seed) for seed in sim.truth_seeds(target)[:TRUTH_RUNS]]
    return {r: [x[r] for x in runs] for r in sim.REWARDS}


_CONTEXT: list[sim.Context] = []


def _init(root: Path) -> None:
    _CONTEXT.append(sim.load_contexts(root)["reference"])


def _work(job: tuple[str, dict[str, Any], int | str]) -> tuple[str, int, int | str, Any]:
    kind, setting, key = job
    params = Params(**setting["values"])
    ctx = _CONTEXT[0]
    if kind == "log":
        assert isinstance(key, int)
        return kind, setting["id"], key, sim.estimate_log(ctx, params, key)
    assert isinstance(key, str)
    return kind, setting["id"], key, truth_runs(ctx, params, key)


def bias_resamples(estimates: FloatArray, truths: FloatArray, seed: int) -> FloatArray:
    """Bias over resamples of the log seeds and of the truth runs, independently. Every setting
    shares the log and truth seeds, so one ``seed`` for all settings pairs their resamples, as
    the rule of METHODS 12.7 requires."""
    rng = np.random.default_rng(seed)
    e = estimates[rng.integers(0, len(estimates), (RESAMPLES, len(estimates)))].mean(axis=1)
    t = truths[rng.integers(0, len(truths), (RESAMPLES, len(truths)))].mean(axis=1)
    out: FloatArray = e - t
    return out


def summarize(
    grid: list[dict[str, Any]],
    logs: dict[int, dict[int, Any]],
    truths: dict[int, dict[str, dict[str, list[float]]]],
) -> dict[str, Any]:
    per_setting: dict[int, dict[str, Any]] = {}
    boots: dict[tuple[int, str, str], FloatArray] = {}
    for s in grid:
        sid = s["id"]
        per_setting[sid] = {"id": sid, "values": s["values"], "grid": s["grid"], "targets": {}}
        for target in sim.TARGETS:
            values = np.array(truths[sid][target]["visit"])
            t_value, t_se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
            entry: dict[str, Any] = {"truth": {"value": t_value, "se": t_se}}
            for name in ("IPS", "DR"):
                ests = [logs[sid][seed][target]["visit"]["estimates"][name] for seed in LOG_SEEDS]
                entry[name] = sim.validate(ests, t_value, t_se)
                points = np.array([e["value"] for e in ests])
                boots[(sid, target, name)] = bias_resamples(points, values, RESAMPLE_SEED)
            per_setting[sid]["targets"][target] = entry
    flips: dict[str, Any] = {}
    for name in NAMES:
        members = sorted((g[1], s["id"]) for s in grid for g in s["grid"] if g[0] == name)
        ids = [sid for _, sid in members]
        x = np.array([per_setting[sid]["values"][name] for sid in ids])
        flips[name] = {"values": x.tolist()}
        for target in FOCUS:
            for estimator in ("DR", "IPS"):
                pts = np.array(
                    [per_setting[sid]["targets"][target][estimator]["bias"]["value"] for sid in ids]
                )
                b = np.stack([boots[(sid, target, estimator)] for sid in ids], axis=1)
                crossings = compare.flips(x, pts, b)
                flips[name][f"{target}.{estimator}"] = {
                    "bias": [compare.interval(p, b[:, j]) for j, p in enumerate(pts)],
                    "crossings": crossings,
                    # Paired change of the bias from the first to the last grid point
                    "change": compare.interval(pts[-1] - pts[0], b[:, -1] - b[:, 0]),
                    "result": "no flip in range"
                    if not crossings
                    else ("flip" if any(c["resolved"] for c in crossings) else "not resolved"),
                }
    return {"settings": [per_setting[s["id"]] for s in grid], "flips": flips}


PAIRS = (("N", "T"), ("T_half", "T"), ("T2", "T"))


def _interval(value: float, se: float, q: float) -> dict[str, float]:
    return {"value": value, "ci_low": value - q * se, "ci_high": value + q * se}


def orderings(result: dict[str, Any], folder: Path) -> dict[str, Any]:
    """Post hoc: along each grid, the true difference between two targets against its per-day
    estimate (mean over log seeds); reversed where both intervals exclude 0 with opposite signs."""
    settings = result["settings"]
    out: dict[str, Any] = {}
    for name in NAMES:
        members = sorted((g[1], s["id"]) for s in settings for g in s["grid"] if g[0] == name)
        by_id = {s["id"]: s for s in settings}
        entry: dict[str, Any] = {"values": [by_id[i]["values"][name] for _, i in members]}
        for a, b in PAIRS:
            pair: dict[str, Any] = {"truth": []} | {e: [] for e in ("DR", "IPS")}
            pair["reversed"] = {e: [] for e in ("DR", "IPS")}
            for _, i in members:
                ta, tb = by_id[i]["targets"][a]["truth"], by_id[i]["targets"][b]["truth"]
                se = float(np.hypot(ta["se"], tb["se"]))
                truth = _interval(ta["value"] - tb["value"], se, 1.959964)
                pair["truth"].append(truth)
                logs = [
                    json.loads((folder / f"{i}_{seed}.json").read_text("utf-8"))
                    for seed in result["log_seeds"]
                ]
                for e in ("DR", "IPS"):
                    d = np.array(
                        [
                            x[a]["visit"]["estimates"][e]["value"]
                            - x[b]["visit"]["estimates"][e]["value"]
                            for x in logs
                        ]
                    )
                    q = float(stats.t.ppf(0.975, len(d) - 1))
                    est = _interval(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), q)
                    pair[e].append(est)
                    pair["reversed"][e].append(
                        bool(
                            (est["ci_low"] > 0 and truth["ci_high"] < 0)
                            or (est["ci_high"] < 0 and truth["ci_low"] > 0)
                        )
                    )
            entry[f"{a}-{b}"] = pair
        out[name] = entry
    return {
        "note": "Post hoc, after the results of D048. Per-day estimate of a difference between "
        "targets against its truth, active-day share; t intervals over the log seeds, normal "
        "intervals over the truth runs.",
        "orderings": out,
    }


def run(root: Path, processes: int, reuse_logs: bool = False) -> dict[str, Any]:
    """Every log and truth job; with ``reuse_logs``, the per-log estimates saved by an earlier
    run are read back instead of recomputed (they are deterministic)."""
    start = time.perf_counter()
    grid = settings()
    folder = root / paths.RESULTS / RUNS
    folder.mkdir(parents=True, exist_ok=True)
    logs: dict[int, dict[int, Any]] = {s["id"]: {} for s in grid}
    truths: dict[int, dict[str, Any]] = {s["id"]: {} for s in grid}
    jobs: list[tuple[str, dict[str, Any], int | str]] = [
        ("truth", s, t) for s in grid for t in sim.TARGETS
    ]
    if reuse_logs:
        for s in grid:
            for seed in LOG_SEEDS:
                path = folder / f"{s['id']}_{seed}.json"
                logs[s["id"]][seed] = json.loads(path.read_text("utf-8"))
    else:
        jobs = [("log", s, seed) for s in grid for seed in LOG_SEEDS] + jobs
    with mp.get_context("spawn").Pool(processes, _init, (root,)) as pool:
        for count, (kind, sid, key, result) in enumerate(pool.imap_unordered(_work, jobs), 1):
            if kind == "log":
                logs[sid][int(key)] = result
                (folder / f"{sid}_{key}.json").write_text(json.dumps(result) + "\n", "utf-8")
            else:
                truths[sid][str(key)] = result
                path = folder / f"truth_{sid}_{key}.json"
                path.write_text(json.dumps(result) + "\n", "utf-8")
            print(f"{count}/{len(jobs)} {kind} setting {sid} {key}", flush=True)
    return {
        "tier": "C",
        "note": "Bias of per-day OPE against simulator truth along one-at-a-time grids of the "
        "carryover parameters; other parameters at the reference.",
        "log_seeds": list(LOG_SEEDS),
        "truth_runs": TRUTH_RUNS,
        "world_seed": sim.WORLD_SEED,
        **summarize(grid, logs, truths),
        "timings_s": round(time.perf_counter() - start, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep the OPE check over carryover.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument(
        "--reuse-logs", action="store_true", help="read the saved per-log estimates back"
    )
    parser.add_argument(
        "--post-hoc", action="store_true", help="add the post hoc orderings to existing results"
    )
    args = parser.parse_args(argv)
    target = args.root / paths.RESULTS / RESULTS_FILE
    commit = sweep.code_commit(args.root)
    if args.post_hoc:
        result = json.loads(target.read_text("utf-8"))
        result["post_hoc"] = {"commit": commit} | orderings(
            result, args.root / paths.RESULTS / RUNS
        )
    else:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        result = {"commit": commit}
        if args.reuse_logs:
            earlier = json.loads(target.read_text("utf-8"))
            result |= {"logs_commit": earlier.get("logs_commit", earlier["commit"])}
        result |= run(args.root, args.processes, args.reuse_logs)
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
