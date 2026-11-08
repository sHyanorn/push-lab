"""Uplift evaluation on Hillstrom and Criteo (METHODS Sections 7-10).

Writes ``results/uplift.json``. Every number is Tier A: measured on randomized trials.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from push_lab.data import paths, uplift_trials
from push_lab.rank import metrics
from push_lab.runtime import rounded, write_json
from push_lab.uplift import curves
from push_lab.uplift.forest import CRITEO_FOREST_ROWS, CausalForest
from push_lab.uplift.learners import SLearner, TLearner, XLearner

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

OUTCOMES = ("visit", "conversion")
PRIMARY_OUTCOME = "visit"
BUDGETS = np.array([0.1, 0.2, 0.3, 0.5])
LEARNERS = ("s_learner", "t_learner", "x_learner", "causal_forest")
METHODS = (*LEARNERS, "response", "random")
SEED = 0

HILLSTROM_NUMERIC = ("recency", "history", "mens", "womens", "newbie")
HILLSTROM_CATEGORIES = {
    "zip_code": ("Surburban", "Urban", "Rural"),
    "channel": ("Phone", "Web", "Multichannel"),
}
CRITEO_FEATURES = tuple(f"f{i}" for i in range(12))
CRITEO_COLUMNS = [*CRITEO_FEATURES, "treatment", *OUTCOMES]
QINI_GRID_POINTS = 101
DECILES = 10
CRITEO_BATCH_ROWS = 1 << 20

CURVES = {
    "qini": "joint Qini curve (Radcliffe 2007), coefficient normalized as scikit-uplift "
    "qini_auc_score(negative_effect=True)",
    "auuc": "joint uplift curve, normalized as scikit-uplift uplift_auc_score",
}


@dataclass
class Settings:
    bootstrap_samples: int = metrics.BOOTSTRAP_SAMPLES
    criteo_bins: int = curves.CRITEO_BINS
    forest_rows: int = CRITEO_FOREST_ROWS
    batch_rows: int = CRITEO_BATCH_ROWS


@dataclass(frozen=True)
class Trial:
    """Test rows of one trial and every method's score per outcome."""

    t: IntArray
    outcomes: dict[str, IntArray]
    scores: dict[str, dict[str, FloatArray]]


def _interval(point: float, boot: FloatArray) -> dict[str, Any]:
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return {"value": rounded(point), "ci_low": rounded(low), "ci_high": rounded(high), "tier": "A"}


def _p_value(boot: FloatArray) -> float:
    boot = boot[~np.isnan(boot)]
    return min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))


# Statistics of one method's counts: Qini, AUUC, then per budget the incremental responders per
# 1,000 rows and the share of the treat-everyone effect, then the treat-everyone effect per 1,000,
# then the Qini curve on the grid
def _statistics(c: curves.Counts) -> FloatArray:
    at = curves.uplift_at(c, np.r_[BUDGETS, 1.0])
    full = at[-1]
    share = at[:-1] / full if full != 0 else np.full(len(BUDGETS), np.nan)
    per_1000 = at / c.n * 1000
    _, grid = curves.qini_on_grid(c, QINI_GRID_POINTS)
    out: FloatArray = np.r_[
        curves.qini_coefficient(c), curves.auuc(c), per_1000[:-1], share, per_1000[-1], grid
    ]
    return out


def _band(point: FloatArray, boot: FloatArray) -> dict[str, list[float]]:
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {
        "value": [rounded(v) for v in point],
        "ci_low": [rounded(v) for v in low],
        "ci_high": [rounded(v) for v in high],
    }


def _bootstrap(
    cell: curves.Cells,
    stat: Callable[[list[curves.Counts]], FloatArray],
    samples: int,
) -> FloatArray:
    sizes = [int(g.max()) + 1 for g in cell.groups]
    rows = []
    for w in curves.resampled_weights(cell.count, samples, SEED):
        rows.append(
            stat(
                [
                    curves.counts(g, cell.t, cell.y, w, n)
                    for g, n in zip(cell.groups, sizes, strict=True)
                ]
            )
        )
    return np.array(rows)


def _groups(score: FloatArray, bins: int | None) -> IntArray:
    return curves.exact_groups(score) if bins is None else curves.quantile_groups(score, bins)


def _control_per_bin(groups: IntArray, t: IntArray) -> dict[str, float]:
    control = np.bincount(groups[t == 0], minlength=int(groups.max()) + 1)
    present = np.bincount(groups) > 0
    control = control[present]
    q = np.percentile(control, [5, 50, 95])
    return {
        "bins": int(present.sum()),
        "min": int(control.min()),
        "p5": rounded(q[0]),
        "p50": rounded(q[1]),
        "p95": rounded(q[2]),
        "bins_without_control": int((control == 0).sum()),
    }


def _assess(
    t: IntArray, y: IntArray, scores: dict[str, FloatArray], bins: int | None, samples: int
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Every method's metrics with intervals; also returns the bootstrap Qini coefficients."""
    grid = np.linspace(0.0, 1.0, QINI_GRID_POINTS)
    out: dict[str, Any] = {
        "methods": {},
        "budget": {},
        "qini_curve": {"tier": "A", "share": [rounded(v) for v in grid], "methods": {}},
    }
    boot_qini: dict[str, FloatArray] = {}
    k = len(BUDGETS)
    full = 2 + 2 * k
    for name, score in scores.items():
        groups = _groups(score, bins)
        point = _statistics(curves.counts(groups, t, y))
        boot = _bootstrap(curves.cells([groups], t, y), lambda c: _statistics(c[0]), samples)
        boot_qini[name] = boot[:, 0]
        out["methods"][name] = {
            "qini": _interval(point[0], boot[:, 0]),
            "auuc": _interval(point[1], boot[:, 1]),
        }
        everyone = _interval(point[full], boot[:, full])
        # A share of an effect that may be zero is not defined; it is left out then
        share_defined = everyone["ci_low"] > 0 or everyone["ci_high"] < 0
        out["budget"][name] = {
            str(b): {
                "per_1000": _interval(point[2 + i], boot[:, 2 + i]),
                "share_of_full_effect": _interval(point[2 + k + i], boot[:, 2 + k + i])
                if share_defined
                else None,
            }
            for i, b in enumerate(BUDGETS)
        }
        out["treat_everyone_per_1000"] = everyone
        out["qini_curve"]["methods"][name] = _band(point[full + 1 :], boot[:, full + 1 :])
        if bins is not None:
            out.setdefault("control_rows_per_bin", {})[name] = _control_per_bin(groups, t)
    return out, boot_qini


def _paired_qini(
    t: IntArray, y: IntArray, a: FloatArray, b: FloatArray, bins: int | None, samples: int
) -> dict[str, Any]:
    """Qini coefficient of a minus that of b, bootstrapped on the same resampled rows."""
    ga, gb = _groups(a, bins), _groups(b, bins)
    point = curves.qini_coefficient(curves.counts(ga, t, y)) - curves.qini_coefficient(
        curves.counts(gb, t, y)
    )
    boot = _bootstrap(
        curves.cells([ga, gb], t, y),
        lambda c: np.array([curves.qini_coefficient(c[0]) - curves.qini_coefficient(c[1])]),
        samples,
    )[:, 0]
    return _interval(point, boot) | {"p_value": rounded(_p_value(boot))}


def _by_decile(t: IntArray, y: IntArray, score: FloatArray, samples: int) -> dict[str, Any]:
    """Response rates and uplift per 1,000 rows by score decile, highest scores first."""
    groups = curves.quantile_groups(score, DECILES)

    def stat(c: list[curves.Counts]) -> FloatArray:
        n_t, n_c = c[0].n_t, c[0].n_c
        treated = np.divide(c[0].y_t, n_t, out=np.full_like(n_t, np.nan), where=n_t > 0)
        control = np.divide(c[0].y_c, n_c, out=np.full_like(n_c, np.nan), where=n_c > 0)
        out: FloatArray = np.r_[treated, control, treated - control] * 1000
        return out

    cell = curves.cells([groups], t, y)
    point = stat([curves.counts(groups, t, y)])
    boot = _bootstrap(cell, stat, samples)
    n = int(groups.max()) + 1
    parts = {"treated_per_1000": 0, "control_per_1000": n, "uplift_per_1000": 2 * n}
    return {"tier": "A"} | {
        name: _band(point[i : i + n], boot[:, i : i + n]) for name, i in parts.items()
    }


def _against_zero(est: dict[str, Any], boot: FloatArray) -> dict[str, Any]:
    return est | {"p_value": rounded(_p_value(boot))}


Effect = Callable[[FloatArray], FloatArray]


def _fit(
    x: FloatArray, t: IntArray, y: IntArray, forest_rows: IntArray, timings: dict[str, float]
) -> dict[str, Effect]:
    """Fit every learner on the training rows; returns each method's scoring function."""

    def timed(name: str, fit: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        model = fit()
        timings[name] = round(time.perf_counter() - start, 1)
        return model

    s = timed("s_learner", lambda: SLearner.fit(x, t, y))
    tl = timed("t_learner", lambda: TLearner.fit(x, t, y))
    xl = timed("x_learner_stage2", lambda: XLearner.fit(x, t, y, stage1=tl))
    fx, ft, fy = x[forest_rows], t[forest_rows], y[forest_rows]
    forest = timed("causal_forest", lambda: CausalForest.fit(fx, ft, fy))
    return {
        "s_learner": s.effect,
        "t_learner": tl.effect,
        "x_learner": xl.effect,
        "causal_forest": forest.effect,
        "response": tl.response,
    }


def hillstrom_features(df: pd.DataFrame) -> FloatArray:
    columns = [df[c].to_numpy(dtype=np.float64) for c in HILLSTROM_NUMERIC]
    for name, values in HILLSTROM_CATEGORIES.items():
        columns += [(df[name] == v).to_numpy(dtype=np.float64) for v in values]
    return np.column_stack(columns)


def run_hillstrom(root: Path, timings: dict[str, Any]) -> tuple[Trial, dict[str, Any]]:
    df = uplift_trials.load_hillstrom(paths.processed_dir(root, "hillstrom") / "hillstrom.parquet")
    t_all = df["treatment"].to_numpy(dtype=np.int64)
    strata = t_all * 2 + df[PRIMARY_OUTCOME].to_numpy(dtype=np.int64)
    train, test = train_test_split(
        np.arange(len(df)), test_size=0.5, stratify=strata, random_state=SEED
    )
    x = hillstrom_features(df)
    t, t_test = t_all[train], t_all[test]
    scores: dict[str, dict[str, FloatArray]] = {}
    outcomes: dict[str, IntArray] = {}
    for outcome in OUTCOMES:
        y_all = df[outcome].to_numpy(dtype=np.int64)
        outcomes[outcome] = y_all[test]
        timings[outcome] = {}
        models = _fit(x[train], t, y_all[train], np.arange(len(train)), timings[outcome])
        scores[outcome] = {name: effect(x[test]) for name, effect in models.items()}
        scores[outcome]["random"] = np.random.default_rng(SEED).random(len(test))
    setup = {
        "contrast": f"{uplift_trials.HILLSTROM_TREATED} against {uplift_trials.HILLSTROM_CONTROL}",
        "training_rows": len(train),
        "test_rows": len(test),
        "treated_share_training": rounded(t.mean()),
        "treated_share_test": rounded(t_test.mean()),
        "curves": "exact",
    }
    return Trial(t_test, outcomes, scores), setup


def score_criteo_test(
    path: Path, models: dict[str, dict[str, Effect]], batch_rows: int
) -> tuple[IntArray, dict[str, IntArray], dict[str, dict[str, FloatArray]]]:
    """Score the rows outside the training sample in one pass, reading labels from the same
    batches as the features so that scores and labels line up by construction."""
    t_parts: list[IntArray] = []
    y_parts: dict[str, list[IntArray]] = {o: [] for o in models}
    s_parts: dict[str, dict[str, list[FloatArray]]] = {
        o: {name: [] for name in m} for o, m in models.items()
    }
    batches = pq.ParquetFile(path).iter_batches(
        batch_size=batch_rows, columns=[*CRITEO_COLUMNS, "train"]
    )
    for batch in batches:
        frame = batch.to_pandas()
        frame = frame[~frame["train"]]
        x = frame[list(CRITEO_FEATURES)].to_numpy(dtype=np.float64)
        t_parts.append(frame["treatment"].to_numpy(dtype=np.int64))
        for outcome, methods in models.items():
            y_parts[outcome].append(frame[outcome].to_numpy(dtype=np.int64))
            for name, effect in methods.items():
                s_parts[outcome][name].append(effect(x))
    return (
        np.concatenate(t_parts),
        {o: np.concatenate(p) for o, p in y_parts.items()},
        {o: {n: np.concatenate(p) for n, p in m.items()} for o, m in s_parts.items()},
    )


def run_criteo(
    root: Path, settings: Settings, timings: dict[str, Any]
) -> tuple[Trial, dict[str, Any]]:
    path = paths.processed_dir(root, "criteo") / "criteo.parquet"
    train = pq.read_table(path, columns=CRITEO_COLUMNS, filters=[("train", "==", True)])
    x = np.column_stack([train[c].to_numpy().astype(np.float64) for c in CRITEO_FEATURES])
    t = train["treatment"].to_numpy().astype(np.int64)
    rng = np.random.default_rng(SEED)
    forest_rows = np.sort(rng.choice(len(t), size=min(settings.forest_rows, len(t)), replace=False))
    models: dict[str, dict[str, Effect]] = {}
    for outcome in OUTCOMES:
        y = train[outcome].to_numpy().astype(np.int64)
        timings[outcome] = {}
        models[outcome] = _fit(x, t, y, forest_rows, timings[outcome])
    del train, x
    start = time.perf_counter()
    t_test, outcomes, scores = score_criteo_test(path, models, settings.batch_rows)
    timings["scoring"] = round(time.perf_counter() - start, 1)
    for outcome in OUTCOMES:
        scores[outcome]["random"] = np.random.default_rng(SEED).random(len(t_test))
    setup = {
        "contrast": "treatment == 1 (assigned to ads) against treatment == 0, intention to treat",
        "training_rows": len(t),
        "forest_training_rows": len(forest_rows),
        "test_rows": len(t_test),
        "treated_share_training": rounded(t.mean()),
        "treated_share_test": rounded(t_test.mean()),
        "curves": f"binned at {settings.criteo_bins} score quantiles",
    }
    return Trial(t_test, outcomes, scores), setup


def _sklift_difference(trial: Trial) -> float:
    """Largest absolute difference from scikit-uplift over every Hillstrom method and outcome."""
    try:
        from sklift import metrics as sk
    except ImportError as err:
        raise RuntimeError(
            "the scikit-uplift cross-check needs the development dependencies (uv sync)"
        ) from err

    worst = 0.0
    for outcome, y in trial.outcomes.items():
        for score in trial.scores[outcome].values():
            c = curves.counts(curves.exact_groups(score), trial.t, y)
            worst = max(
                worst,
                abs(
                    curves.qini_coefficient(c)
                    - sk.qini_auc_score(y, score, trial.t, negative_effect=True)
                ),
                abs(curves.auuc(c) - sk.uplift_auc_score(y, score, trial.t)),
            )
    return worst


def _evaluate(trial: Trial, bins: int | None, samples: int) -> tuple[dict[str, Any], FloatArray]:
    """Metrics for every outcome; returns the primary outcome's bootstrap T-learner Qini."""
    out: dict[str, Any] = {}
    boot_t = np.empty(0)
    for outcome, y in trial.outcomes.items():
        out[outcome], boot = _assess(trial.t, y, trial.scores[outcome], bins, samples)
        if outcome == PRIMARY_OUTCOME:
            boot_t = boot["t_learner"]
    return out, boot_t


def run(root: Path, settings: Settings | None = None, criteo: bool = True) -> dict[str, Any]:
    """Both trials; with ``criteo=False`` Hillstrom only, without Holm decisions (D050)."""
    s = settings or Settings()
    timings: dict[str, Any] = {"hillstrom": {}, "criteo": {}} if criteo else {"hillstrom": {}}
    primary: dict[str, Any] = {}
    result: dict[str, Any] = {"curves": CURVES}

    hill, hill_setup = run_hillstrom(root, timings["hillstrom"])
    hill_metrics, hill_boot = _evaluate(hill, None, s.bootstrap_samples)
    y = hill.outcomes[PRIMARY_OUTCOME]
    scores = hill.scores[PRIMARY_OUTCOME]
    primary["U1_hillstrom_t_learner_qini"] = _against_zero(
        hill_metrics[PRIMARY_OUTCOME]["methods"]["t_learner"]["qini"], hill_boot
    )
    primary["U2_hillstrom_t_learner_vs_response_qini"] = _paired_qini(
        hill.t, y, scores["t_learner"], scores["response"], None, s.bootstrap_samples
    )
    hill_setup["scikit_uplift_max_abs_difference"] = float(f"{_sklift_difference(hill):.3g}")
    result["hillstrom"] = {"setup": hill_setup, **hill_metrics}
    if not criteo:
        result["primary"] = primary
        result["timings_s"] = timings
        return result

    crit, crit_setup = run_criteo(root, s, timings["criteo"])
    start = time.perf_counter()
    crit_metrics, crit_boot = _evaluate(crit, s.criteo_bins, s.bootstrap_samples)
    y = crit.outcomes[PRIMARY_OUTCOME]
    scores = crit.scores[PRIMARY_OUTCOME]
    primary["U3_criteo_t_learner_qini"] = _against_zero(
        crit_metrics[PRIMARY_OUTCOME]["methods"]["t_learner"]["qini"], crit_boot
    )
    primary["U4_criteo_t_learner_vs_response_qini"] = _paired_qini(
        crit.t, y, scores["t_learner"], scores["response"], s.criteo_bins, s.bootstrap_samples
    )
    crit_metrics[PRIMARY_OUTCOME]["deciles_post_hoc"] = {
        name: _by_decile(crit.t, y, scores[name], s.bootstrap_samples)
        for name in ("t_learner", "response")
    }
    timings["criteo"]["bootstrap"] = round(time.perf_counter() - start, 1)
    result["criteo"] = {"setup": crit_setup, **crit_metrics}

    rejected = metrics.holm({name: entry["p_value"] for name, entry in primary.items()})
    for name, entry in primary.items():
        entry["holm_rejected"] = rejected[name]
        entry["p_value_resolution"] = rounded(2 / s.bootstrap_samples)
    result["primary"] = primary
    result["timings_s"] = timings
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate uplift methods on Hillstrom and Criteo.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--only",
        choices=["hillstrom"],
        help="skip Criteo and Holm; writes results/runs/uplift_hillstrom.json",
    )
    args = parser.parse_args(argv)
    result = run(args.root, criteo=args.only is None)
    name = "uplift.json" if args.only is None else "runs/uplift_hillstrom.json"
    out = args.root / paths.RESULTS / name
    write_json(out, result, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
