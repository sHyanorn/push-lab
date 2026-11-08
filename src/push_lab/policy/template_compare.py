"""Template policy comparisons (METHODS 13.5). Every number here is Tier C.

``python -m push_lab.policy.template_compare`` reads ``results/runs/templates/`` and writes
``results/templates.json``.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from push_lab.data import paths
from push_lab.policy import compare, sweep, template_design
from push_lab.policy.compare import Bootstrap, _r, interval, p_value
from push_lab.policy.template_design import GRID_NAMES, PRIMARY
from push_lab.policy.template_sweep import METRICS, RUNS
from push_lab.policy.templates import KINDS

FloatArray = npt.NDArray[np.float64]

SECONDARY = {"D0-U": ("D0", "U"), "RAW-RD": ("RAW", "RD"), "O-U": ("O", "U")}
PAIRS = PRIMARY | SECONDARY
SETTING_LEVEL = 0.05 / len(PRIMARY)
LABELED_SWEEP = "main"
NOTE = (
    "The simulator never reads the text. A template's effect is a random draw per world, "
    "unrelated to its wording. These comparisons test whether the selection policy finds and "
    "rotates the templates with the largest drawn effects; they say nothing about whether the "
    "generated copy is good or which wording works better for real users."
)


@dataclass
class SettingResult:
    summary: dict[str, Any]
    boot: dict[str, FloatArray]
    points: dict[str, float]


def evaluate_setting(metrics: FloatArray, kinds: list[str], boot: Bootstrap) -> SettingResult:
    """``metrics`` is [kind, metric, user] in the order of ``METRICS``."""
    pos = {k: i for i, k in enumerate(kinds)}
    opens, active = METRICS.index("opens"), METRICS.index("active_days")
    rows: dict[str, FloatArray] = {}
    for kind in kinds:
        for m, metric in enumerate(METRICS):
            rows[f"{kind}.{metric}"] = metrics[pos[kind], m]
    for key, (a, b) in PAIRS.items():
        rows[f"opens.{key}"] = metrics[pos[a], opens] - metrics[pos[b], opens]
        rows[f"active.{key}"] = metrics[pos[a], active] - metrics[pos[b], active]
    keys = list(rows)
    stacked = np.stack([rows[k] for k in keys])
    point = dict(zip(keys, stacked.mean(axis=1), strict=True))
    means = boot.means(stacked)
    bt = {k: means[:, i] for i, k in enumerate(keys)}

    def iv(key: str) -> dict[str, float]:
        return interval(point[key], bt[key])

    def ratio(num: str, den: str) -> dict[str, float] | None:
        if point[den] == 0:
            return None
        return interval(point[num] / point[den], bt[num] / bt[den])

    primary = {}
    for key in PRIMARY:
        p = p_value(bt[f"opens.{key}"])
        primary[key] = iv(f"opens.{key}") | {
            "p_value": _r(p),
            "verdict": compare.verdict(point[f"opens.{key}"], p, SETTING_LEVEL),
        }
    oracle = iv("opens.O-U")
    share = (
        ratio("opens.B1", "opens.O-U") if oracle["ci_low"] > 0 or oracle["ci_high"] < 0 else None
    )
    summary = {
        "kinds": {k: {m: iv(f"{k}.{m}") for m in METRICS} for k in kinds},
        "primary": primary,
        "secondary_opens": {k: iv(f"opens.{k}") for k in SECONDARY},
        "active_days": {k: iv(f"active.{k}") for k in PAIRS},
        "opens_per_push": {k: ratio(f"{k}.opens", f"{k}.delivered") for k in kinds},
        "repeat_rate": {k: ratio(f"{k}.repeats", f"{k}.delivered") for k in kinds},
        "share_of_oracle_gain": share,
    }
    kept = {k: bt[f"opens.{k}"] for k in PAIRS}
    points = {k: float(point[f"opens.{k}"]) for k in PAIRS}
    return SettingResult(summary, kept, points)


def _median(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return _r(float(np.median(known))) if known else None


def summarize(
    results: dict[int, SettingResult], frozen: dict[str, Any], infos: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    settings = {s["id"]: s for s in frozen["settings"]}
    by_sweep: dict[str, list[int]] = {}
    for sid in sorted(results):
        by_sweep.setdefault(settings[sid]["sweep"], []).append(sid)
    designed = {
        sw: sum(s["sweep"] == sw for s in frozen["settings"])
        for sw in sorted({s["sweep"] for s in frozen["settings"]})
    }
    complete = {sw: len(by_sweep.get(sw, [])) == n for sw, n in designed.items()}
    seed = frozen["bootstrap"]["settings_seed"]

    main_ids = sorted(by_sweep.get(LABELED_SWEEP, []), key=lambda i: settings[i]["point"])
    labeled: dict[str, Any] = {}
    p_all: dict[str, float] = {}
    averages: dict[str, Any] = {}
    if main_ids:
        for key in PAIRS:
            boots = np.stack([results[i].boot[key] for i in main_ids])
            pts = np.array([results[i].points[key] for i in main_ids])
            avg, p, _ = compare.sweep_average(boots, pts, seed)
            if key not in PRIMARY:
                averages[key] = avg
                continue
            entries = [results[i].summary["primary"][key] for i in main_ids]
            verdicts = [e["verdict"] for e in entries]
            labeled[key] = {
                "sweep_average": avg | {"p_value": _r(p)},
                "wins": verdicts.count("win"),
                "losses": verdicts.count("loss"),
                "neither": verdicts.count("neither"),
                "median_half_width": _r(
                    float(np.median([(e["ci_high"] - e["ci_low"]) / 2 for e in entries]))
                ),
                "losing_settings": [
                    {"id": i, "values": settings[i]["values"]}
                    for i, v in zip(main_ids, verdicts, strict=True)
                    if v == "loss"
                ],
            }
            p_all[key] = p
    # Labels and Holm need every main setting; a partial run gets none
    rejected = compare.holm(p_all) if complete.get(LABELED_SWEEP) else {}
    for key, entry in labeled.items():
        ok = rejected.get(key)
        entry["holm_rejected"] = ok
        entry["label"] = (
            compare.label(entry["sweep_average"]["value"], ok, entry["wins"], entry["losses"])
            if ok is not None
            else "incomplete"
        )
    shares = [results[i].summary["share_of_oracle_gain"] for i in main_ids]
    oracle_share = {
        "median": _median([s["value"] if s else None for s in shares]),
        "settings_with_oracle_gain": sum(s is not None for s in shares),
    }

    grid_ids = by_sweep.get("grid", [])
    flips: dict[str, Any] = {}
    for name in GRID_NAMES:
        members = sorted((g[1], i) for i in grid_ids for g in settings[i]["grid"] if g[0] == name)
        if len(members) < 2:
            continue
        ids = [i for _, i in members]
        x = np.array([settings[i]["values"][name] for i in ids])
        flips[name] = {}
        for key in PRIMARY:
            pts = np.array([results[i].points[key] for i in ids])
            crossings = compare.flips(x, pts, np.stack([results[i].boot[key] for i in ids], axis=1))
            flips[name][key] = {
                "values": [_r(v) for v in x],
                "estimates": [results[i].summary["primary"][key] for i in ids],
                "crossings": crossings,
                "result": "no flip in range"
                if not crossings
                else ("flip" if any(c["resolved"] for c in crossings) else "not resolved"),
            }

    variability: dict[str, Any] = {}
    replicates = by_sweep.get("replicate", [])
    if len(replicates) > 1:
        for key in PRIMARY:
            pts = np.array([results[i].points[key] for i in replicates])
            se = np.mean([results[i].boot[key].std() for i in replicates])
            variability[key] = {
                "sd_over_seed_triples": _r(pts.std(ddof=1)),
                "mean_bootstrap_se": _r(se),
                "triples": len(replicates),
            }

    diagnostics: dict[str, Any] = {}
    for sw, ids in by_sweep.items():
        diagnostics[sw] = {
            kind: {
                stat: _median([d[stat] for i in ids for d in infos[i]["score_diagnostics"][kind]])
                for stat in ("spearman", "top_overlap")
            }
            for kind in infos[ids[0]]["score_diagnostics"]
        }

    samples = len(next(iter(results.values())).boot["B1"])
    return {
        "tier": "C",
        "note": NOTE,
        "settings_note": "Settings are points drawn uniformly over the parameter box, not "
        "probabilities of the real world.",
        "metric": frozen["metric"],
        "p_value_resolution": _r(2 / samples),
        "complete": complete,
        "primary": labeled,
        "primary_holm": {k: bool(v) for k, v in rejected.items()},
        "secondary_sweep_averages": averages,
        "share_of_oracle_gain": oracle_share,
        "flips": flips,
        "seed_variability": variability,
        "score_diagnostics_median": diagnostics,
        "settings": [
            {
                "id": i,
                "sweep": settings[i]["sweep"],
                "values": {k: _r(v) for k, v in settings[i]["values"].items()},
                "uniform_family_share": infos[i]["uniform_family_share"],
                "timings_s": infos[i]["timings_s"],
            }
            | results[i].summary
            for i in sorted(results)
        ],
    }


def run(root: Path) -> dict[str, Any]:
    frozen = template_design.load(root)
    b = frozen["bootstrap"]
    boot = Bootstrap(frozen["users"], b["samples"], b["seed"])
    results: dict[int, SettingResult] = {}
    infos: dict[int, dict[str, Any]] = {}
    commits: set[str] = set()
    for s in frozen["settings"]:
        npz, meta = sweep.setting_paths(root, s["id"], RUNS)
        if not npz.exists():
            continue
        info = json.loads(meta.read_text("utf-8"))
        commits.add(info["commit"])
        with np.load(npz) as z:
            metrics = z["metrics"].astype(np.float64)
        if info["kinds"] != list(KINDS):
            raise SystemExit(f"setting {s['id']} ran other policies: {info['kinds']}")
        results[s["id"]] = evaluate_setting(metrics, info["kinds"], boot)
        infos[s["id"]] = info
    if len(commits) != 1:
        raise SystemExit(f"the settings come from {len(commits)} commits: {sorted(commits)}")
    out = {"commit": commits.pop()} | summarize(results, frozen, infos)
    out["settings_run"] = len(results)
    out["settings_designed"] = len(frozen["settings"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the template policies.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    result = run(args.root)
    out = args.root / paths.RESULTS / "templates.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
