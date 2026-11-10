"""Policy comparisons, decision labels, flip thresholds and the plausibility band
(METHODS Section 12.7). Every number here is Tier C.

``python -m push_lab.policy.compare`` reads ``results/runs/policy/`` and writes
``results/policy.json``.
"""

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from push_lab.data import paths
from push_lab.policy import design
from push_lab.policy.runs import POLICY_METRICS, POLICY_RUNS, setting_paths
from push_lab.runtime import rounded, write_json
from push_lab.stats import flip_result, flips, holm, interval, p_value

FloatArray = npt.NDArray[np.float64]

PRIMARY = {"H1": ("G", "S"), "H2": ("G2", "S"), "H3": ("G", "T"), "H4": ("T", "F")}
SECONDARY = {
    "G-N": ("G", "N"),
    "T-N": ("T", "N"),
    "F-N": ("F", "N"),
    "G-T_half": ("G", "T_half"),
    "T_half-T": ("T_half", "T"),
    "G2-T": ("G2", "T"),
    "L1": ("T", "T_two_tower"),
}
PER_PUSH = ("R", "F", "T", "S", "G", "G2")
SETTING_LEVEL = 0.05 / len(PRIMARY)
LABELED_SWEEPS = ("main", "annoyance_free")
MOST, SPLIT = 48, 4
BAND = (0.0007, 0.005)
FLIP_COMPARISONS = (*PRIMARY, "F-N")
H2_NOTE = (
    "G2 assumes an open and an organic visit are independent given the features; this "
    "holds in the simulator by construction, so part of any H2 win comes from the "
    "simulator's structure."
)
CHUNK = 1_000


class Bootstrap:
    """User resamples as count weights, shared by every setting (same users everywhere)."""

    def __init__(self, users: int, samples: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.weights = np.zeros((samples, users), dtype=np.float32)
        for start in range(0, samples, CHUNK):
            rows = min(CHUNK, samples - start)
            idx = rng.integers(0, users, size=(rows, users))
            flat = (np.arange(rows)[:, None] * users + idx).ravel()
            counts = np.bincount(flat, minlength=rows * users).reshape(rows, users)
            self.weights[start : start + rows] = counts
        self.users = users

    @property
    def samples(self) -> int:
        return int(self.weights.shape[0])

    def means(self, values: FloatArray) -> FloatArray:
        """Resampled means [sample, row] of ``values`` [row, user], in double precision."""
        out = np.empty((self.samples, len(values)))
        for start in range(0, self.samples, CHUNK):
            w = self.weights[start : start + CHUNK].astype(np.float64)
            out[start : start + CHUNK] = w @ values.T / self.users
        return out


def verdict(point: float, p: float, level: float = SETTING_LEVEL) -> str:
    if p > level:
        return "neither"
    return "win" if point > 0 else "loss"


def label(average: float, rejected: bool, wins: int, losses: int) -> str:
    """METHODS 12.7, step 3, in its order."""
    if rejected and average > 0 and wins >= MOST:
        return "beats in most sampled settings"
    if rejected and average < 0 and losses >= MOST:
        return "loses in most sampled settings"
    if wins >= SPLIT and losses >= SPLIT:
        return "depends on the setting"
    if rejected:
        return "better on average" if average > 0 else "worse on average"
    return "no detectable difference"


def band_class(low: float, high: float, band: tuple[float, float] = BAND) -> str:
    if high < band[0]:
        return "below"
    if low > band[1]:
        return "above"
    if low >= band[0] and high <= band[1]:
        return "inside"
    return "straddling"


@dataclass
class SettingResult:
    summary: dict[str, Any]
    boot: dict[str, FloatArray]
    points: dict[str, float]


def grid_flips(
    results: dict[int, SettingResult],
    settings: dict[int, dict[str, Any]],
    grid_ids: list[int],
    names: Sequence[str],
    keys: Sequence[str],
    estimate: Callable[[SettingResult, str], Any],
) -> dict[str, dict[str, Any]]:
    """Flips of each comparison along each one-at-a-time grid with at least two points."""
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        members = sorted((g[1], i) for i in grid_ids for g in settings[i]["grid"] if g[0] == name)
        if len(members) < 2:
            continue
        ids = [i for _, i in members]
        x = np.array([settings[i]["values"][name] for i in ids])
        out[name] = {}
        for key in keys:
            pts = np.array([results[i].points[key] for i in ids])
            crossings = flips(x, pts, np.stack([results[i].boot[key] for i in ids], axis=1))
            out[name][key] = {
                "values": [rounded(v) for v in x],
                "estimates": [estimate(results[i], key) for i in ids],
                "crossings": crossings,
                "result": flip_result(crossings),
            }
    return out


def labeled_entry(
    results: dict[int, SettingResult],
    settings: dict[int, dict[str, Any]],
    ids: list[int],
    key: str,
    average: dict[str, float],
    p: float,
) -> dict[str, Any]:
    """Sweep average and per-setting verdicts of a registered comparison (METHODS 12.7)."""
    entries = [results[i].summary["primary"][key] for i in ids]
    verdicts = [e["verdict"] for e in entries]
    return {
        "sweep_average": average | {"p_value": rounded(p)},
        "wins": verdicts.count("win"),
        "losses": verdicts.count("loss"),
        "neither": verdicts.count("neither"),
        "median_half_width": rounded(
            float(np.median([(e["ci_high"] - e["ci_low"]) / 2 for e in entries]))
        ),
        "losing_settings": [
            {"id": i, "values": settings[i]["values"]}
            for i, v in zip(ids, verdicts, strict=True)
            if v == "loss"
        ],
    }


def add_label(entry: dict[str, Any], rejected: bool | None) -> None:
    """Holm decision and label; ``None`` when the run is incomplete."""
    entry["holm_rejected"] = rejected
    entry["label"] = (
        label(entry["sweep_average"]["value"], rejected, entry["wins"], entry["losses"])
        if rejected is not None
        else "incomplete"
    )


def load_runs(
    root: Path, frozen: dict[str, Any], runs: Path
) -> Iterator[tuple[int, dict[str, Any], FloatArray]]:
    """Id, run info and metrics of every finished setting of a frozen design."""
    for s in frozen["settings"]:
        npz, meta = setting_paths(root, s["id"], runs)
        if not npz.exists():
            continue
        info = json.loads(meta.read_text("utf-8"))
        with np.load(npz) as z:
            metrics = z["metrics"].astype(np.float64)
        yield s["id"], info, metrics


def run_commit(infos: dict[int, dict[str, Any]]) -> str:
    """The one commit every setting ran at."""
    commits: set[str] = {info["commit"] for info in infos.values()}
    if len(commits) != 1:
        raise SystemExit(f"the settings come from {len(commits)} commits: {sorted(commits)}")
    return commits.pop()


def evaluate_setting(metrics: FloatArray, names: list[str], boot: Bootstrap) -> SettingResult:
    """Levels, differences and ratios of one setting with intervals; ``metrics`` is
    [policy, metric, user] in the order of ``POLICY_METRICS``."""
    pos = {n: i for i, n in enumerate(names)}
    active = POLICY_METRICS.index("active_days")
    rows: dict[str, FloatArray] = {}
    for p in names:
        for m, metric in enumerate(POLICY_METRICS):
            rows[f"{p}.{metric}"] = metrics[pos[p], m]
    for key, (first, second) in (PRIMARY | SECONDARY).items():
        rows[f"diff.{key}"] = metrics[pos[first], active] - metrics[pos[second], active]
    for p in PER_PUSH:
        rows[f"gain.{p}"] = metrics[pos[p], active] - metrics[pos["N"], active]
    keys = list(rows)
    stacked = np.stack([rows[k] for k in keys]).astype(np.float64)
    point = dict(zip(keys, stacked.mean(axis=1), strict=True))
    means = boot.means(stacked)
    b = {k: means[:, i] for i, k in enumerate(keys)}

    policies = {
        p: {
            metric: interval(point[f"{p}.{metric}"], b[f"{p}.{metric}"])
            for metric in POLICY_METRICS
        }
        for p in names
    }
    comparisons: dict[str, Any] = {}
    for key in PRIMARY:
        d = f"diff.{key}"
        pv = p_value(b[d])
        comparisons[key] = interval(point[d], b[d]) | {
            "p_value": rounded(pv),
            "verdict": verdict(point[d], pv),
        }
    comparisons["H2"]["note"] = H2_NOTE
    secondary = {key: interval(point[f"diff.{key}"], b[f"diff.{key}"]) for key in SECONDARY}
    per_push = {}
    for p in PER_PUSH:
        num, den = f"gain.{p}", f"{p}.delivered"
        if point[den] > 0:
            per_push[p] = interval(point[num] / point[den], b[num] / b[den])
    lift_boot = b["diff.F-N"] / b["N.active_days"]
    lift = interval(point["diff.F-N"] / point["N.active_days"], lift_boot)
    lift_low, lift_high = np.percentile(lift_boot, [2.5, 97.5])
    summary = {
        "policies": policies,
        "primary": comparisons,
        "secondary": secondary,
        "active_days_per_push_over_N": per_push,
        "daily_lift_F_over_N": lift | {"band": band_class(lift_low, lift_high)},
    }
    kept = {k: b[f"diff.{k}"] for k in (*PRIMARY, *SECONDARY)}
    points = {k: float(point[f"diff.{k}"]) for k in (*PRIMARY, *SECONDARY)}
    return SettingResult(summary, kept, points)


def sweep_average(
    boots: FloatArray, points: FloatArray, seed: int
) -> tuple[dict[str, float], float, FloatArray]:
    """Mean over settings with a two-level bootstrap: each user resample is paired with a
    resample of the settings. ``boots`` is [setting, sample]."""
    settings, samples = boots.shape
    pick = np.random.default_rng(seed).integers(0, settings, size=(samples, settings))
    averaged = boots[pick, np.arange(samples)[:, None]].mean(axis=1)
    point = float(points.mean())
    return interval(point, averaged), p_value(averaged), averaged


def summarize(
    results: dict[int, SettingResult], frozen: dict[str, Any], infos: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    settings = {s["id"]: s for s in frozen["settings"]}
    by_sweep: dict[str, list[int]] = {}
    for sid in sorted(results):
        by_sweep.setdefault(settings[sid]["sweep"], []).append(sid)
    seed = frozen["bootstrap"]["settings_seed"]

    designed = {
        sweep: sum(s["sweep"] == sweep for s in frozen["settings"])
        for sweep in sorted({s["sweep"] for s in frozen["settings"]})
    }
    complete = {sweep: len(by_sweep.get(sweep, [])) == n for sweep, n in designed.items()}
    labeled: dict[str, dict[str, Any]] = {}
    p_all: dict[str, float] = {}
    for sweep in LABELED_SWEEPS:
        ids = sorted(by_sweep.get(sweep, []), key=lambda i: settings[i]["point"])
        if not ids:
            continue
        labeled[sweep] = {}
        for key in PRIMARY:
            boots = np.stack([results[i].boot[key] for i in ids])
            pts = np.array([results[i].points[key] for i in ids])
            avg, p, _ = sweep_average(boots, pts, seed)
            labeled[sweep][key] = labeled_entry(results, settings, ids, key, avg, p)
            p_all[f"{sweep}.{key}"] = p
    # Labels and Holm need every registered setting; a partial run gets none
    ready = all(complete.get(sweep, True) for sweep in LABELED_SWEEPS)
    rejected = holm(p_all) if ready else {}
    for sweep, entries in labeled.items():
        for key, entry in entries.items():
            add_label(entry, rejected.get(f"{sweep}.{key}"))
            if key == "H2":
                entry["note"] = H2_NOTE

    stress: dict[str, dict[str, Any]] = {}
    if by_sweep.get("stress"):
        for key in PRIMARY:
            verdicts = [results[i].summary["primary"][key]["verdict"] for i in by_sweep["stress"]]
            stress[key] = {v: verdicts.count(v) for v in ("win", "loss", "neither")}

    band: dict[str, int] = {}
    for i in by_sweep.get("main", []):
        c = results[i].summary["daily_lift_F_over_N"]["band"]
        band[c] = band.get(c, 0) + 1

    flipped = grid_flips(
        results,
        settings,
        by_sweep.get("grid", []),
        frozen["names"],
        FLIP_COMPARISONS,
        lambda r, key: (r.summary["primary"] | r.summary["secondary"])[key],
    )

    fit_variability: dict[str, dict[str, Any]] = {}
    refits = by_sweep.get("refit", [])
    if len(refits) > 1:
        for key in PRIMARY:
            pts = np.array([results[i].points[key] for i in refits])
            se = np.mean([results[i].boot[key].std() for i in refits])
            fit_variability[key] = {
                "sd_over_refits": rounded(pts.std(ddof=1)),
                "mean_bootstrap_se": rounded(se),
                "refits": len(refits),
            }
        fit_variability["H2"]["note"] = H2_NOTE

    diagnostics: dict[str, dict[str, float]] = {}
    for sweep, ids in by_sweep.items():
        keys = list(infos[ids[0]]["diagnostics"])
        diagnostics[sweep] = {
            k: rounded(float(np.median([infos[i]["diagnostics"][k] for i in ids]))) for k in keys
        }

    samples = len(next(iter(results.values())).boot["H1"])
    return {
        "tier": "C",
        "note": "Simulator results with the sweep; settings are points drawn uniformly over "
        "the parameter box, not probabilities of the real world.",
        "p_value_resolution": rounded(2 / samples),
        "complete": complete,
        "primary": labeled,
        "primary_holm": {k: bool(v) for k, v in rejected.items()},
        "stress_verdicts": {"counts": stress, "H2_note": H2_NOTE},
        "band_F_over_N": {"bounds": list(BAND), "main_sweep_counts": band},
        "flips": flipped,
        "fit_variability": fit_variability,
        "score_diagnostics_median": diagnostics,
        "settings": [
            {
                "id": i,
                "sweep": settings[i]["sweep"],
                "values": {k: rounded(v) for k, v in settings[i]["values"].items()},
                "multipliers": infos[i].get("multipliers", {}),
                "cap_binding_days": infos[i].get("cap_binding_days", {}),
                "fixed_slot": infos[i]["fixed_slot"],
                "diagnostics": {k: rounded(v) for k, v in infos[i]["diagnostics"].items()},
                "timings_s": infos[i]["timings_s"],
            }
            | results[i].summary
            for i in sorted(results)
        ],
    }


def run(root: Path) -> dict[str, Any]:
    frozen = design.load(root)
    boot = Bootstrap(
        frozen["users"]["evaluation"],
        frozen["bootstrap"]["samples"],
        frozen["bootstrap"]["seed"],
    )
    results: dict[int, SettingResult] = {}
    infos: dict[int, dict[str, Any]] = {}
    for sid, info, metrics in load_runs(root, frozen, POLICY_RUNS):
        results[sid] = evaluate_setting(metrics, info["policies"], boot)
        infos[sid] = info
    out = {"commit": run_commit(infos)} | summarize(results, frozen, infos)
    out["settings_run"] = len(results)
    out["settings_designed"] = len(frozen["settings"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the policies of the sweep.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    result = run(args.root)
    out = args.root / paths.RESULTS / "policy.json"
    write_json(out, result)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
