"""Template sweep design, frozen before any run (METHODS 13.5).

``python -m push_lab.policy.template_design`` writes ``results/template_design.json`` from the
frozen policy design: its 64 main points, its one-at-a-time points for four parameters and the
reference repeated with further run seeds.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from push_lab.creative.families import FAMILIES, TEMPLATES
from push_lab.data import paths
from push_lab.policy import design
from push_lab.policy import templates as tp

GRID_NAMES = ("base_rate", "copy_spread", "novelty", "novelty_half_life")
REPLICATES = 10
REPLICATE_SEED = 201
DESIGN_FILE = "template_design.json"
PRIMARY = {"B1": ("RD", "U"), "B2": ("RD", "D0")}


def _copy(s: dict[str, Any], sweep: str, **extra: Any) -> dict[str, Any]:
    return {
        "sweep": sweep,
        "point": s["point"],
        "policy_setting": s["id"],
        "values": s["values"],
        "grid": s["grid"],
        "world_seed": s["world_seed"],
        "perturb_seed": s["perturb_seed"],
        "eval_seeds": s["eval_seeds"],
    } | extra


def settings(policy_design: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source = policy_design["settings"]
    reference = policy_design["reference"]
    out += [_copy(s, "main") for s in source if s["sweep"] == "main"]
    grid = [s for s in source if s["sweep"] == "grid"]
    ref = [s for s in grid if s["values"] == reference]
    if len(ref) != 1:
        raise ValueError("the policy design must hold the reference setting once")
    for s in grid:
        kept = [g for g in s["grid"] if g[0] in GRID_NAMES]
        if kept or s is ref[0]:
            out.append(_copy(s, "grid", grid=kept))
    for r in range(REPLICATES):
        first = REPLICATE_SEED + 3 * r
        out.append(
            _copy(
                ref[0], "replicate", point=r + 1, grid=[], eval_seeds=[first, first + 1, first + 2]
            )
        )
    for i, s in enumerate(out):
        s["id"] = i
    return out


def build(root: Path, n_users: int) -> dict[str, Any]:
    source = root / paths.RESULTS / design.DESIGN_FILE
    policy_design = json.loads(source.read_text("utf-8"))
    return {
        "source": {
            "file": design.DESIGN_FILE,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "names": policy_design["names"],
        "reference": policy_design["reference"],
        "users": n_users,
        "templates": TEMPLATES,
        "families": len(FAMILIES),
        "kinds": list(tp.KINDS),
        "primary": {k: list(v) for k, v in PRIMARY.items()},
        "setting_level": 0.05 / len(PRIMARY),
        "metric": "opens per user over the 28 days, averaged over the run seeds",
        "bandit": {
            "penalty": tp.PENALTY,
            "penalty_half_life": tp.PENALTY_HALF_LIFE,
            "temperature": tp.TEMPERATURE,
            "theta": tp.THETA,
        },
        "bootstrap": policy_design["bootstrap"],
        "settings": settings(policy_design),
    }


def load(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((root / paths.RESULTS / DESIGN_FILE).read_text("utf-8"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the frozen template sweep design.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--users", type=int, default=27_285, help="simulated population")
    args = parser.parse_args(argv)
    out = args.root / paths.RESULTS / DESIGN_FILE
    data = build(args.root, args.users)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(data['settings'])} settings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
