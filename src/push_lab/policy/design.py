"""Sweep design: settings, users and seeds, frozen before any run (METHODS Section 12.6).

``python -m push_lab.policy.design`` writes ``results/policy_design.json``.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

from push_lab.data import paths
from push_lab.sim import world
from push_lab.sim.world import Params

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

SIM_NAMES = tuple(f.name for f in fields(Params))
POLICY_RANGES = {"profile_shift": (-3.0, 3.0), "profile_noise": (0.0, 1.0)}
NAMES = (*SIM_NAMES, *POLICY_RANGES)
RANGES = {**world.RANGES, **POLICY_RANGES}
REFERENCE = {**asdict(Params()), "profile_shift": 0.0, "profile_noise": 0.0}

POINTS = 64
LHS_SEED = 0
GRID_POINTS = 5
REFITS = 10
EVAL_SEEDS = (101, 102, 103)
FIT_USERS, FIT_SEED = 10_000, 1
DIAGNOSTIC_USERS, DIAGNOSTIC_SEED = 2_000, 2
BUDGET = 0.5
SECONDARY_BUDGET = 1.0
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, SETTINGS_SEED = 10_000, 0, 1
# Integer seeds of one setting must differ from each other (D032)
WORLD, EXPLORE, PERTURB, DIAGNOSTIC = 10_000, 20_000, 40_000, 50_000
GRID_OFFSET = 100
DESIGN_FILE = "policy_design.json"


def hypercube(points: int = POINTS, seed: int = LHS_SEED) -> FloatArray:
    """Scrambled Latin hypercube in the unit cube, one column per name in ``NAMES``."""
    sampler = qmc.LatinHypercube(d=len(NAMES), rng=np.random.default_rng(seed))
    out: FloatArray = sampler.random(points)
    return out


def scale(unit: FloatArray) -> list[dict[str, float]]:
    low = np.array([RANGES[n][0] for n in NAMES])
    high = np.array([RANGES[n][1] for n in NAMES])
    values = low + unit * (high - low)
    return [dict(zip(NAMES, map(float, row), strict=True)) for row in values]


def _seeds(offset: int, explore_offset: int | None = None) -> dict[str, Any]:
    explore = offset if explore_offset is None else explore_offset
    return {
        "world_seed": WORLD + offset,
        "explore_seed": EXPLORE + explore,
        "perturb_seed": PERTURB + offset,
        "diagnostic_seed": DIAGNOSTIC + offset,
        "eval_seeds": list(EVAL_SEEDS),
    }


def settings() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    points = scale(hypercube())
    for sweep, override in (
        ("main", {}),
        ("annoyance_free", {"annoyance": 0.0}),
        ("stress", {"opt_out": world.OPT_OUT_STRESS}),
    ):
        for j, values in enumerate(points):
            out.append(
                {"sweep": sweep, "point": j, "values": values | override, "grid": []} | _seeds(j)
            )
    # One-at-a-time grid around the reference, seeds fixed along every grid
    grid: dict[tuple[float, ...], dict[str, Any]] = {}
    reference = {n: float(REFERENCE[n]) for n in NAMES}
    grid[tuple(reference.values())] = {"values": reference, "grid": []}
    for name in NAMES:
        for i, v in enumerate(np.linspace(*RANGES[name], GRID_POINTS)):
            values = reference | {name: float(v)}
            entry = grid.setdefault(tuple(values.values()), {"values": values, "grid": []})
            entry["grid"].append([name, i])
    for entry in grid.values():
        out.append({"sweep": "grid", "point": None} | entry | _seeds(GRID_OFFSET))
    for r in range(1, REFITS + 1):
        out.append(
            {"sweep": "refit", "point": r, "values": reference, "grid": []}
            | _seeds(GRID_OFFSET, explore_offset=GRID_OFFSET + r)
        )
    for i, s in enumerate(out):
        s["id"] = i
    return out


def users(n_users: int) -> dict[str, list[int]]:
    """Disjoint evaluation, fit and diagnostic users, as sorted positions."""
    evaluation = world.quick_sample(n_users)
    rest = np.setdiff1d(np.arange(n_users), evaluation)
    fit = np.sort(np.random.default_rng(FIT_SEED).choice(rest, FIT_USERS, replace=False))
    rest = np.setdiff1d(rest, fit)
    diagnostic = np.sort(
        np.random.default_rng(DIAGNOSTIC_SEED).choice(rest, DIAGNOSTIC_USERS, replace=False)
    )
    return {
        "evaluation": evaluation.tolist(),
        "fit": fit.tolist(),
        "diagnostic": diagnostic.tolist(),
    }


def build(n_users: int) -> dict[str, Any]:
    all_settings = settings()
    chosen = users(n_users)
    return {
        "names": list(NAMES),
        "ranges": {n: list(RANGES[n]) for n in NAMES},
        "reference": {n: float(REFERENCE[n]) for n in NAMES},
        "budget": BUDGET,
        "secondary_budget": SECONDARY_BUDGET,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "settings_seed": SETTINGS_SEED,
        },
        "users": {k: len(v) for k, v in chosen.items()}
        | {
            "sha256": hashlib.sha256(
                json.dumps(chosen, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "population": n_users,
        },
        "settings": all_settings,
    }


def load(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((root / paths.RESULTS / DESIGN_FILE).read_text("utf-8"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the frozen sweep design.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--users", type=int, default=27_285, help="simulated population")
    args = parser.parse_args(argv)
    out = args.root / paths.RESULTS / DESIGN_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(args.users), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
