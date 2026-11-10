"""Run every setting of the frozen template design (METHODS 13.4 and 13.5).

``python -m push_lab.policy.template_sweep`` writes one file pair per setting to
``results/runs/templates/`` (ignored by git), skips settings finished at the same commit and
refuses to mix commits, then writes ``results/templates.json`` with ``template_compare``.
Run it alone: it uses one thread per process.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from push_lab.creative import generate
from push_lab.creative.families import FAMILIES
from push_lab.data import paths
from push_lab.policy import design, profile, runs, template_compare, template_design
from push_lab.policy import templates as tp
from push_lab.runtime import code_commit, one_thread_per_process
from push_lab.sim import env, inputs, world
from push_lab.sim.world import DAYS, SLOTS, Params, slot_time

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

REPEAT_DAYS = 15.0
TOP = 10


@dataclass(frozen=True)
class Context:
    inputs: inputs.Inputs
    family: IntArray

    def population(self) -> env.Population:
        i = self.inputs
        return env.Population(i.visit_rate, i.profile_sim, i.lists["cascade"].p0)


def repeats(out: env.Outcome, templates: int, days: float = REPEAT_DAYS) -> FloatArray:
    """Per user, deliveries whose template had reached the user within ``days`` before."""
    n = out.delivered.shape[2]
    last = np.full((n, templates), -np.inf)
    count = np.zeros(n)
    for d in range(DAYS):
        for s in range(SLOTS):
            users = np.flatnonzero(out.delivered[d, s])
            k = out.template[d, s, users].astype(np.int64)
            t = slot_time(d, s)
            count[users] += t - last[users, k] <= days
            last[users, k] = t
    return count


def user_metrics(out: env.Outcome, templates: int) -> FloatArray:
    """Per user, in the order of ``runs.TEMPLATE_METRICS``."""
    return np.stack(
        [
            out.opened.sum(axis=(0, 1)),
            out.visited.sum(axis=0),
            out.opened[:7].sum(axis=(0, 1)),
            out.delivered.sum(axis=(0, 1)),
            repeats(out, templates),
        ]
    ).astype(np.float64)


def score_diagnostics(scores: FloatArray, copy: FloatArray) -> dict[str, float | None]:
    """Agreement of the learned scores with the drawn copy effects (none without a spread)."""
    # Without a drawn spread, or under full shrinkage, there is no ranking to compare
    if np.ptp(copy) == 0 or np.ptp(scores) == 0:
        return {"spearman": None, "top_overlap": None}
    top_true = set(np.argsort(-copy, kind="stable")[:TOP].tolist())
    top_learned = set(np.argsort(-scores, kind="stable")[:TOP].tolist())
    return {
        "spearman": float(spearmanr(scores, np.log(copy)).statistic),
        "top_overlap": len(top_true & top_learned) / TOP,
    }


def run_setting(ctx: Context, s: dict[str, Any]) -> tuple[FloatArray, dict[str, Any]]:
    """Per-user metrics [kind, metric, user] averaged over the run seeds, and info."""
    start = time.perf_counter()
    v = s["values"]
    params = Params(**{k: v[k] for k in design.SIM_NAMES})
    inp = ctx.inputs
    prof = profile.perturbed(
        inp.hourly_policy, v["profile_shift"], v["profile_noise"], s["perturb_seed"]
    )
    slot = prof.argmax(axis=1)
    k = len(ctx.family)
    copy = world.copy_effects(params.copy_spread, s["world_seed"], k, len(FAMILIES))
    pop = ctx.population()
    n = pop.users
    total = np.zeros((len(tp.KINDS), len(runs.TEMPLATE_METRICS), n))
    diagnostics: dict[str, list[dict[str, float | None]]] = {kind: [] for kind in tp.LEARNED}
    family_share = np.zeros(len(FAMILIES))
    for r in s["eval_seeds"]:
        run = env.draws(n, r)
        for i, kind in enumerate(tp.KINDS):
            policy = tp.Templated(
                kind,
                slot,
                ctx.family,
                inp.creator_clicks,
                inp.trend_rank,
                r,
                copy=copy,
                novelty=params.novelty,
                novelty_half_life=params.novelty_half_life,
            )
            out = env.simulate(pop, params, policy, run, s["world_seed"], k)
            total[i] += user_metrics(out, k)
            if kind in tp.LEARNED:
                diagnostics[kind].append(score_diagnostics(policy.scores, copy))
            if kind == "U":
                chosen = ctx.family[out.template[out.delivered]]
                family_share += np.bincount(chosen, minlength=len(FAMILIES)) / len(chosen)
    info: dict[str, Any] = {
        "id": s["id"],
        "kinds": list(tp.KINDS),
        "score_diagnostics": diagnostics,
        "uniform_family_share": (family_share / len(s["eval_seeds"])).round(6).tolist(),
        "timings_s": {"setting": round(time.perf_counter() - start, 1)},
    }
    return total / len(s["eval_seeds"]), info


def load_context(root: Path) -> Context:
    inp = inputs.Inputs.load(paths.processed_dir(root, "sim") / inputs.CACHE)
    catalog = generate.load_catalog(root)
    family = np.array([t["family"] for t in catalog["templates"]], dtype=np.int64)
    ids = [t["id"] for t in catalog["templates"]]
    if ids != list(range(len(ids))) or not (family == np.arange(len(ids)) % len(FAMILIES)).all():
        raise SystemExit("the catalog must number templates as 6j + f (D036)")
    return Context(inp, family)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the template sweep and compare.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--ids", type=int, nargs="*", help="only these setting ids")
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args(argv)
    one_thread_per_process()
    frozen = template_design.load(args.root)
    commit = code_commit(args.root)
    ctx = load_context(args.root)
    current = template_design.build(args.root, len(ctx.inputs.user_ids))
    if json.loads(json.dumps(current)) != frozen or len(ctx.family) != frozen["templates"]:
        raise SystemExit("the design in results/ differs from the code or the catalog")
    del ctx
    runs.run_pending(
        args.root,
        frozen,
        commit,
        args.ids,
        args.processes,
        runs.TEMPLATE_RUNS,
        load_context,
        run_setting,
    )
    if not args.no_compare and args.ids is None:
        template_compare.main(["--root", str(args.root)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
