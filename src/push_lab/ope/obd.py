"""Validation of the off-policy estimators on the Open Bandit Dataset (METHODS 14.2).

``python -m push_lab.ope.obd`` estimates the BTS policy's click rate from the uniform random
logs of the Men's campaign and checks it against the click rate of the BTS logs. It uses the
small sample, and the full campaign too when ``prepare --only obd_full`` has been run. Writes
``results/ope_obd.json``. Every number here is Tier B only for an estimator that passes. None
passed on the full campaign, so the project has no Tier B result (D046). ``obd_blocks``
recomputes the intervals post hoc (D063).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab.data import obd, paths
from push_lab.data.schemas import OBD_MEN_ACTIONS
from push_lab.ope import estimators as est
from push_lab.runtime import code_commit, write_json

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Production Beta prior of the Men's campaign, obp/policy/conf/prior_bts.yaml at the pinned
# commit (sources.OBD_COMMIT; file sha256 81a54fbd...)
ALPHA: tuple[int, ...] = (
    *(47, 8, 62, 142, 3, 6, 100, 48, 23, 71, 61, 13, 16, 518, 30, 7, 4, 23, 8, 10, 11, 11),
    *(18, 121, 11, 4, 32, 30, 9, 22, 6, 23, 5, 54),
)
BETA: tuple[int, ...] = (
    *(12198, 3566, 15993, 35522, 2367, 3100, 24665, 13210, 7061, 18061, 17449, 5644, 6787),
    *(111326, 8776, 3334, 2271, 7389, 2659, 3665, 4724, 3561, 5085, 27407, 4601, 2923),
    *(8897, 8654, 4000, 6580, 3174, 6766, 2602, 14506),
)
POSITIONS = 3
DRAWS, DRAW_SEED, DRAW_CHUNK = 1_000_000, 0, 50_000
FOLDS, FOLD_SEED = 2, 0
SAMPLES, BOOT_SEED = 1_000, 0
MAX_RELATIVE_ERROR = 0.5
SCORE_CHUNK = 20_000
# Rows shown together share user features and lie within a second of each other
IMPRESSION_GAP_US = 1_000_000
# Base model of METHODS Section 8
PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "num_threads": 4,
    "verbose": -1,
}
ROUNDS = 200
NOTE_SMALL = (
    "The small sample has 46 clicks in the random logs and 69 in the BTS logs, so this check "
    "has low power: it can show a gross failure, not a small bias."
)


def bts_distribution(
    alpha: Sequence[float] = ALPHA,
    beta: Sequence[float] = BETA,
    draws: int = DRAWS,
    seed: int = DRAW_SEED,
) -> FloatArray:
    """P(item at position) [item, position] of BTS with no updates: each draw ranks the items
    by a sample from their Beta prior and shows the top ``POSITIONS`` in order."""
    a, b = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(a), POSITIONS))
    for start in range(0, draws, DRAW_CHUNK):
        n = min(DRAW_CHUNK, draws - start)
        top = np.argsort(-rng.beta(a, b, size=(n, len(a))), axis=1)[:, :POSITIONS]
        for k in range(POSITIONS):
            counts[:, k] += np.bincount(top[:, k], minlength=len(a))
    out: FloatArray = counts / draws
    return out


def features(log: pd.DataFrame, items: pd.DataFrame, item: IntArray) -> pd.DataFrame:
    """Context of each row with ``item`` in place of the logged one: user features, user-item
    affinities, the affinity to ``item``, its item features and the position."""
    user = [c for c in log.columns if c.startswith("user_feature_")]
    affinity = [f"user_item_affinity_{k}" for k in range(OBD_MEN_ACTIONS)]
    x = log[user + affinity].reset_index(drop=True).copy()
    for c in user:
        x[c] = x[c].astype("category")
    x["own_affinity"] = log[affinity].to_numpy()[np.arange(len(log)), item]
    feats = items.set_index("item_id").loc[item].reset_index(drop=True)
    for c in feats.columns:
        numeric = pd.api.types.is_numeric_dtype(feats[c])
        x[c] = feats[c] if numeric else feats[c].astype("category")
    x["item"] = pd.Categorical(item, categories=range(OBD_MEN_ACTIONS))
    x["position"] = log["position"].to_numpy()
    return x


def _categories(train: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Use the training categories so codes mean the same thing in both frames."""
    out = x.copy()
    for c in train.columns:
        if isinstance(train[c].dtype, pd.CategoricalDtype):
            out[c] = pd.Categorical(out[c].astype(object), categories=train[c].cat.categories)
    return out


def impressions(log: pd.DataFrame) -> IntArray:
    """Impression of each row: OBD has no impression id, so rows in time order with the same
    user features less than ``IMPRESSION_GAP_US`` apart form one. Chains can merge consecutive
    impressions of a user, which only makes the resampling units coarser; a row of another
    user in between splits one, which happens about 110 times in the full random logs."""
    stamp = pd.to_datetime(log["timestamp"], format="ISO8601")
    t = stamp.dt.as_unit("us").astype("int64").to_numpy()
    order = np.argsort(t, kind="stable")
    user = [c for c in log.columns if c.startswith("user_feature_")]
    key = pd.util.hash_pandas_object(log[user].iloc[order], index=False).to_numpy()
    ts = t[order]
    same = np.r_[False, (np.diff(ts) < IMPRESSION_GAP_US) & (key[1:] == key[:-1])]
    out = np.empty(len(log), dtype=np.int64)
    out[order] = np.cumsum(~same) - 1
    return out


def outcome_model(
    log: pd.DataFrame, items: pd.DataFrame, dist: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Cross-fitted q(x_i, a_i) and q(x_i, pi): folds split impressions, and each row is scored
    by the model of the other fold."""
    n = len(log)
    logged_item = log["item_id"].to_numpy(dtype=np.int64)
    position = log["position"].to_numpy(dtype=np.int64) - 1
    group = impressions(log)
    fold = np.random.default_rng(FOLD_SEED).permutation(int(group.max()) + 1)[group] % FOLDS
    x_logged = features(log, items, logged_item)
    y = log["click"].to_numpy()
    q_logged, q_target = np.empty(n), np.empty(n)
    for f in range(FOLDS):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        xt = x_logged.iloc[train]
        model = lgb.train(PARAMS, lgb.Dataset(xt, y[train]), num_boost_round=ROUNDS)
        q_logged[test] = model.predict(_categories(xt, x_logged.iloc[test]))
        for start in range(0, len(test), SCORE_CHUNK):
            rows = test[start : start + SCORE_CHUNK]
            sub = log.iloc[rows]
            every = np.tile(np.arange(OBD_MEN_ACTIONS), len(rows))
            repeated = sub.loc[sub.index.repeat(OBD_MEN_ACTIONS)]
            scores = model.predict(_categories(xt, features(repeated, items, every)))
            scores = np.asarray(scores).reshape(len(rows), OBD_MEN_ACTIONS)
            q_target[rows] = (scores * dist[:, position[rows]].T).sum(axis=1)
    return q_logged, q_target


def _weights(log: pd.DataFrame, dist: FloatArray) -> FloatArray:
    """Target probability of each logged (item, position) over its logging propensity."""
    item = log["item_id"].to_numpy(dtype=np.int64)
    position = log["position"].to_numpy(dtype=np.int64) - 1
    weight: FloatArray = dist[item, position] / log["propensity_score"].to_numpy()
    return weight


def _ratio(
    value: float, truth: float, boot: FloatArray, truth_boot: FloatArray
) -> dict[str, float]:
    """An estimate over the on-policy value. Resample i of the estimate is divided by resample
    i of the on-policy value. In ``validate`` the two come from different logs and seeds, so
    they are independent and the index only matches them up (D060); in ``obd_blocks`` they
    come from the same blocks."""
    ratio = boot / truth_boot
    return {
        "value": value / truth,
        "ci_low": float(np.percentile(ratio, 2.5)),
        "ci_high": float(np.percentile(ratio, 97.5)),
    }


def logged(log: pd.DataFrame, items: pd.DataFrame, dist: FloatArray) -> est.Logged:
    weight = _weights(log, dist)
    q_logged, q_target = outcome_model(log, items, dist)
    return est.Logged(
        log["click"].to_numpy(dtype=np.float64), weight, q_logged, q_target, impressions(log)
    )


def on_policy(log: pd.DataFrame, samples: int, seed: int) -> tuple[float, FloatArray]:
    """Click rate and its resamples over impressions."""
    clicks = log["click"].to_numpy(dtype=np.float64)
    ones, zeros = np.ones(len(clicks)), np.zeros(len(clicks))
    data = est.Logged(clicks, ones, zeros, zeros, impressions(log))
    boot = est.bootstrap(data, samples, seed)[:, est.ESTIMATORS.index("IPS")]
    return float(clicks.mean()), boot


def verdicts(
    report: dict[str, Any], boot: FloatArray, truth: float, truth_boot: FloatArray
) -> dict[str, Any]:
    """The check of each estimator against the on-policy value ``truth``: ``report`` is its
    ``est.summarize`` and ``boot`` and ``truth_boot`` are resamples matched by index."""
    out: dict[str, Any] = {}
    for i, name in enumerate(est.ESTIMATORS):
        e = report["estimates"][name]
        relative = np.abs(boot[:, i] - truth_boot) / truth_boot
        contains = e["ci_low"] <= truth <= e["ci_high"]
        median_error = float(np.median(relative))
        out[name] = {
            "ratio_to_on_policy": _ratio(e["value"], truth, boot[:, i], truth_boot),
            "relative_error_median": median_error,
            "relative_error_interval": np.percentile(relative, [2.5, 97.5]).tolist(),
            "rmse": float(np.sqrt(np.mean((boot[:, i] - truth_boot) ** 2))),
            "interval_contains_on_policy": bool(contains),
            "validated": bool(contains and median_error < MAX_RELATIVE_ERROR),
        }
    return out


def validate(
    random_log: pd.DataFrame,
    bts_log: pd.DataFrame,
    items: pd.DataFrame,
    dist: FloatArray,
    samples: int = SAMPLES,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    data = logged(random_log, items, dist)
    boot = est.bootstrap(data, samples, seed)
    report: Any = est.summarize(data, boot)
    # The BTS logs are resampled on their own, with another seed and their own impressions
    truth, truth_boot = on_policy(bts_log, samples, seed + 1)
    low, high = np.percentile(truth_boot, [2.5, 97.5])
    return {
        "on_policy": {"value": truth, "ci_low": float(low), "ci_high": float(high)},
        "random_rows": len(random_log),
        "random_impressions": int(impressions(random_log).max()) + 1,
        "bts_impressions": int(impressions(bts_log).max()) + 1,
        "random_clicks": int(random_log["click"].sum()),
        "bts_rows": len(bts_log),
        "bts_clicks": int(bts_log["click"].sum()),
        **report,
        "validation": verdicts(report, boot, truth, truth_boot),
    }


def empirical_target(
    random_log: pd.DataFrame,
    bts_log: pd.DataFrame,
    prior: FloatArray,
    samples: int = SAMPLES,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Post hoc: IPS with the BTS logs' own (item, position) frequencies as the target, as a
    ratio to the on-policy value, and that ratio minus the one with the target ``prior``. Both
    are resampled by impression as in ``validate``, with the same resamples of the random logs
    for the two targets, so their difference is paired. The on-policy resamples are independent
    of them, and the empirical target is held fixed across resamples (D060)."""
    dist = np.zeros((len(ALPHA), POSITIONS))
    position = bts_log["position"].to_numpy(dtype=np.int64) - 1
    np.add.at(dist, (bts_log["item_id"].to_numpy(dtype=np.int64), position), 1)
    dist /= dist.sum(axis=0, keepdims=True)
    clicks = random_log["click"].to_numpy(dtype=np.float64)
    zeros, group = np.zeros(len(clicks)), impressions(random_log)
    ips = est.ESTIMATORS.index("IPS")
    value, boot = [], []
    for target in (dist, prior):
        data = est.Logged(clicks, _weights(random_log, target), zeros, zeros, group)
        value.append(est.point(data)["IPS"])
        boot.append(est.bootstrap(data, samples, seed)[:, ips])
    truth, truth_boot = on_policy(bts_log, samples, seed + 1)
    return {
        "note": "Post hoc: IPS against the BTS logs' empirical (item, position) frequencies",
        "ratio_to_on_policy": _ratio(value[0], truth, boot[0], truth_boot),
        "difference_from_fixed_prior": _ratio(
            value[0] - value[1], truth, boot[0] - boot[1], truth_boot
        ),
    }


def run(root: Path) -> dict[str, Any]:
    dist = bts_distribution()
    out: dict[str, Any] = {
        "tier": "B for an estimator that passes; the check itself is on real data",
        "target": "BTS with the production prior of the Men's campaign, no updates",
        "bts_distribution": {"draws": DRAWS, "seed": DRAW_SEED},
        "rule": f"interval contains the on-policy value and median relative error below "
        f"{MAX_RELATIVE_ERROR}",
    }
    for name, dataset in (("small", "obd"), ("full", "obd_full")):
        folder = paths.processed_dir(root, dataset)
        if not (folder / "random_men.parquet").exists():
            out[name] = "not run: data not prepared"
            continue
        random_log, bts_log = obd.load(folder, "random"), obd.load(folder, "bts")
        items = obd.load(folder, "random", items=True)
        out[name] = validate(random_log, bts_log, items, dist)
    out["small_note"] = NOTE_SMALL
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the estimators on OBD.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--post-hoc", action="store_true", help="add the empirical-target check to the results"
    )
    args = parser.parse_args(argv)
    target = args.root / paths.RESULTS / "ope_obd.json"
    if args.post_hoc:
        result = json.loads(target.read_text("utf-8"))
        folder = paths.processed_dir(args.root, "obd_full")
        logs = obd.load(folder, "random"), obd.load(folder, "bts")
        result["full"]["post_hoc"] = {"commit": code_commit(args.root)} | empirical_target(
            *logs, bts_distribution()
        )
    else:
        result = {"commit": code_commit(args.root)} | run(args.root)
    write_json(target, result)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
