"""Run every setting of the frozen design (METHODS Sections 12.3 to 12.6).

``python -m push_lab.policy.sweep`` writes one file pair per setting to
``results/runs/policy/`` (ignored by git), skips settings already written, then writes
``results/policy.json`` with ``compare``. Run it alone: it uses one thread per process.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from push_lab.data import paths
from push_lab.policy import design, profile, rules
from push_lab.policy.features import Rows, Static, logged_rows
from push_lab.policy.gate import KINDS, Models
from push_lab.sim import env, inputs
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import DAYS, SLOTS, Params

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

PRIMARY = ("N", "R", "F", "T", "T_half", "T_two_tower", "S", "G", "G2")
SECONDARY = tuple(f"{k}_lambda" for k in KINDS)
METRICS = ("active_days", "active_week4", "opens_week1", "opens", "delivered", "opted_out")
SECONDARY_SWEEPS = ("main",)
RUNS = Path("runs") / "policy"


@dataclass(frozen=True)
class Context:
    inputs: inputs.Inputs
    evaluation: IntArray
    fit: IntArray
    diagnostic: IntArray
    budget: float = design.BUDGET
    threads: int = 1

    def population(self, users: IntArray, name: str = "cascade") -> env.Population:
        i = self.inputs
        return env.Population(i.visit_rate[users], i.profile_sim[users], i.lists[name].p0[users])


def policies_for(sweep: str) -> tuple[str, ...]:
    return PRIMARY + SECONDARY if sweep in SECONDARY_SWEEPS else PRIMARY


def user_metrics(out: env.Outcome) -> FloatArray:
    """Per user, in the order of ``METRICS``."""
    return np.stack(
        [
            out.visited.sum(axis=0),
            out.visited[21:DAYS].sum(axis=0),
            out.opened[:7].sum(axis=(0, 1)),
            out.opened.sum(axis=(0, 1)),
            out.delivered.sum(axis=(0, 1)),
            out.opted_out_at >= 0,
        ]
    ).astype(np.float64)


def _user_day_best(models: Models, rows: Rows) -> IntArray:
    """Row index, per (user, day), of the slot with the highest predicted open probability."""
    p = models.predict_open(rows.x)
    key = rows.user * DAYS + rows.day
    order = np.lexsort((rows.slot, -p, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    out: IntArray = order[first].astype(np.int64)
    return out


def multipliers(models: Models, rows: Rows, budget: float) -> dict[str, float]:
    """Per kind, the (1 - budget) quantile of user-day scores on the fit logs, floored at 0."""
    best = _user_day_best(models, rows)
    x = rows.x[best]
    out: dict[str, float] = {}
    for kind in KINDS:
        score = models.row_scores(x, kind)
        out[kind] = max(0.0, float(np.quantile(score, 1 - budget)))
    return out


def diagnostics(models: Models, out: env.Outcome, rows: Rows) -> dict[str, float]:
    """Rank agreement of each score with the true same-day uplift (METHODS 12.3)."""
    assert out.open_prob is not None and out.organic is not None
    p_true = out.open_prob[rows.day, rows.slot, rows.user].astype(np.float64)
    true = p_true * (1 - out.organic[rows.day, rows.user])
    result: dict[str, float] = {}
    for kind in KINDS:
        result[f"spearman_{kind}"] = float(
            spearmanr(models.row_scores(rows.x, kind), true).statistic
        )
    p_hat = models.predict_open(rows.x)
    key = rows.user * DAYS + rows.day
    full = np.bincount(key, minlength=int(key.max()) + 1) == SLOTS
    keep = full[key]
    k, s = key[keep], rows.slot[keep]
    hat = np.full((int(key.max()) + 1, SLOTS), -np.inf)
    tru = np.full((int(key.max()) + 1, SLOTS), -np.inf)
    hat[k, s], tru[k, s] = p_hat[keep], true[keep]
    days = np.flatnonzero(full)
    result["best_slot_share"] = float(np.mean(hat[days].argmax(axis=1) == tru[days].argmax(axis=1)))
    result["user_days"] = float(len(days))
    return result


def run_setting(ctx: Context, s: dict[str, Any]) -> tuple[FloatArray, dict[str, Any]]:
    """Per-user metrics [policy, metric, user] averaged over the evaluation seeds, and info."""
    timings: dict[str, float] = {}
    start = time.perf_counter()
    v = s["values"]
    params = Params(**{k: v[k] for k in design.SIM_NAMES})
    world_seed = s["world_seed"]
    prof = profile.perturbed(
        ctx.inputs.hourly_policy, v["profile_shift"], v["profile_noise"], s["perturb_seed"]
    )
    static = Static(prof, ctx.inputs.visit_rate_policy, ctx.inputs.lists["cascade"].score)

    def logged(users: IntArray, seed: int, truth: bool) -> tuple[env.Outcome, Rows]:
        sub = static.subset(users)
        explorer = EpsilonGreedy(sub.profile.argmax(axis=1), seed=seed)
        draws = env.draws(len(users), seed)
        out = env.simulate(ctx.population(users), params, explorer, draws, world_seed, truth=truth)
        return out, logged_rows(out, sub)

    _, rows = logged(ctx.fit, s["explore_seed"], truth=False)
    models = Models.fit(rows, ctx.threads)
    timings["fit_s"] = time.perf_counter() - start
    info: dict[str, Any] = {
        "id": s["id"],
        "fit_rows": len(rows.treated),
        "fit_treated_share": float(rows.treated.mean()),
    }
    names = policies_for(s["sweep"])
    lam = multipliers(models, rows, ctx.budget) if any("lambda" in n for n in names) else {}
    info["multipliers"] = lam

    start = time.perf_counter()
    out, diag_rows = logged(ctx.diagnostic, s["diagnostic_seed"], truth=True)
    info["diagnostics"] = diagnostics(models, out, diag_rows)
    timings["diagnostics_s"] = time.perf_counter() - start

    start = time.perf_counter()
    ev = ctx.evaluation
    n = len(ev)
    ev_static = static.subset(ev)
    pop, pop_two_tower = ctx.population(ev), ctx.population(ev, "two_tower")
    fixed = int(prof.mean(axis=0).argmax())
    best = ev_static.profile.argmax(axis=1)
    info["fixed_slot"] = fixed
    total = np.zeros((len(names), len(METRICS), n))
    cap_days: dict[str, list[int]] = {}
    for r in s["eval_seeds"]:
        run = env.draws(n, r)
        for i, name in enumerate(names):
            policy: env.Policy
            if name == "N":
                policy = rules.Never(n, r)
            elif name == "R":
                policy = rules.RandomSlots(n, r)
            elif name == "F":
                policy = rules.Daily(np.full(n, fixed, dtype=np.int64), r)
            elif name in ("T", "T_two_tower"):
                policy = rules.Daily(best, r)
            elif name == "T_half":
                policy = rules.Daily(best, r, half=True)
            else:
                kind, _, variant = name.partition("_")
                policy = rules.Scored(
                    models,
                    ev_static,
                    kind,  # type: ignore[arg-type]
                    ctx.budget,
                    r,
                    multiplier=lam[kind] if variant == "lambda" else None,
                )
            population = pop_two_tower if name == "T_two_tower" else pop
            result = env.simulate(population, params, policy, run, world_seed)
            total[i] += user_metrics(result)
            if isinstance(policy, rules.Scored) and policy.multiplier is not None:
                cap_days.setdefault(name, []).append(policy.cap_binding_days)
    timings["evaluation_s"] = time.perf_counter() - start
    info["cap_binding_days"] = cap_days
    info["timings_s"] = {k: round(t, 1) for k, t in timings.items()}
    info["policies"] = list(names)
    return total / len(s["eval_seeds"]), info


def setting_paths(root: Path, setting_id: int) -> tuple[Path, Path]:
    base = root / paths.RESULTS / RUNS / f"setting_{setting_id:03d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


_CONTEXT: Context | None = None
_ROOT: Path | None = None
_COMMIT = ""


def _init(root: Path, commit: str) -> None:
    global _CONTEXT, _ROOT, _COMMIT
    _ROOT, _COMMIT = root, commit
    _CONTEXT = load_context(root)


def _work(s: dict[str, Any]) -> tuple[int, float]:
    assert _CONTEXT is not None and _ROOT is not None
    start = time.perf_counter()
    values, info = run_setting(_CONTEXT, s)
    npz, meta = setting_paths(_ROOT, s["id"])
    npz.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(info | {"commit": _COMMIT}, indent=1) + "\n", encoding="utf-8")
    # The archive marks a finished setting, so it is written last and moved into place whole
    partial = npz.with_name(npz.stem + ".partial.npz")
    np.savez_compressed(partial, metrics=values.astype(np.float32))
    os.replace(partial, npz)
    return s["id"], time.perf_counter() - start


def finished(root: Path, setting_id: int, commit: str) -> bool:
    """A setting is done when both files exist; done under another commit is an error."""
    npz, meta = setting_paths(root, setting_id)
    if not (npz.exists() and meta.exists()):
        return False
    ran = json.loads(meta.read_text("utf-8")).get("commit")
    if ran != commit:
        raise SystemExit(f"setting {setting_id} ran with commit {ran}, not {commit}")
    return True


def code_commit(root: Path) -> str:
    """HEAD, refusing a working tree whose code differs from it."""

    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    if git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"):
        raise SystemExit("commit the code before running the sweep")
    return git("rev-parse", "HEAD")


def load_context(root: Path) -> Context:
    inp = inputs.Inputs.load(paths.processed_dir(root, "sim") / inputs.CACHE)
    chosen = design.users(len(inp.user_ids))
    return Context(
        inp,
        np.asarray(chosen["evaluation"], dtype=np.int64),
        np.asarray(chosen["fit"], dtype=np.int64),
        np.asarray(chosen["diagnostic"], dtype=np.int64),
    )


def check_design(frozen: dict[str, Any], n_users: int) -> None:
    """Stop if the code no longer reproduces the frozen design."""
    current = design.build(n_users)
    if json.loads(json.dumps(current)) != frozen:
        raise SystemExit("the design in results/ differs from the code; see D031")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the policy sweep and compare.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--ids", type=int, nargs="*", help="only these setting ids")
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    frozen = design.load(args.root)
    commit = code_commit(args.root)
    # Load once here so a broken cache fails before any worker starts
    ctx = load_context(args.root)
    check_design(frozen, len(ctx.inputs.user_ids))
    del ctx
    todo = [
        s
        for s in frozen["settings"]
        if (args.ids is None or s["id"] in args.ids) and not finished(args.root, s["id"], commit)
    ]
    # Settings with secondary policies take longest; start them first
    todo.sort(key=lambda s: (s["sweep"] not in SECONDARY_SWEEPS, s["id"]))
    print(f"{len(todo)} settings to run at {commit}", flush=True)
    start = time.perf_counter()
    with mp.get_context("spawn").Pool(args.processes, _init, (args.root, commit)) as pool:
        for done, (sid, seconds) in enumerate(pool.imap_unordered(_work, todo), 1):
            elapsed = time.perf_counter() - start
            print(f"{done}/{len(todo)} setting {sid}: {seconds:.0f} s, {elapsed:.0f} s", flush=True)
    if not args.no_compare and args.ids is None:
        from push_lab.policy import compare

        compare.main(["--root", str(args.root)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
