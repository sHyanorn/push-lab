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
    """Bias over resamples of the log seeds and of the truth runs, independently."""
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
        per_setting[sid] = {"values": s["values"], "grid": s["grid"], "targets": {}}
        for target in sim.TARGETS:
            values = np.array(truths[sid][target]["visit"])
            t_value, t_se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
            entry: dict[str, Any] = {"truth": {"value": t_value, "se": t_se}}
            for name in ("IPS", "DR"):
                ests = [logs[sid][seed][target]["visit"]["estimates"][name] for seed in LOG_SEEDS]
                entry[name] = sim.validate(ests, t_value, t_se)
                points = np.array([e["value"] for e in ests])
                boots[(sid, target, name)] = bias_resamples(points, values, RESAMPLE_SEED + sid)
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
                    "result": "no flip in range"
                    if not crossings
                    else ("flip" if any(c["resolved"] for c in crossings) else "not resolved"),
                }
    return {"settings": [per_setting[s["id"]] for s in grid], "flips": flips}


def run(root: Path, processes: int) -> dict[str, Any]:
    start = time.perf_counter()
    grid = settings()
    jobs: list[tuple[str, dict[str, Any], int | str]] = [
        ("log", s, seed) for s in grid for seed in LOG_SEEDS
    ] + [("truth", s, t) for s in grid for t in sim.TARGETS]
    logs: dict[int, dict[int, Any]] = {s["id"]: {} for s in grid}
    truths: dict[int, dict[str, Any]] = {s["id"]: {} for s in grid}
    folder = root / paths.RESULTS / RUNS
    folder.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(processes, _init, (root,)) as pool:
        for count, (kind, sid, key, result) in enumerate(pool.imap_unordered(_work, jobs), 1):
            if kind == "log":
                logs[sid][int(key)] = result
                (folder / f"{sid}_{key}.json").write_text(json.dumps(result) + "\n", "utf-8")
            else:
                truths[sid][str(key)] = result
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
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    commit = sweep.code_commit(args.root)
    result = {"commit": commit} | run(args.root, args.processes)
    target = args.root / paths.RESULTS / RESULTS_FILE
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
