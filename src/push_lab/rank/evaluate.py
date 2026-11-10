"""Recall and ranking evaluation on the random-exposure evaluation half (METHODS Sections 1-6).

Writes ``results/ranking.json``. Every metric is Tier A except the model-internal
truncation measures, which are labeled as such.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab import stats
from push_lab.data import kuairand, paths, splits
from push_lab.rank import cascade, fine, metrics, truncation
from push_lab.rank.features import FeatureBuilder, fit_cutoff
from push_lab.recall import popularity, two_tower
from push_lab.recall.index import ItemIndex, overlap
from push_lab.runtime import rounded, write_json

RECALL_K = (50, 100, 200, 500)
TRUNCATION_N = (50, 100, 200, 500)
FUSION_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DRIFT_FEATURES = (
    "user_per_day",
    "item_per_day",
    "user_click_rate",
    "item_click_rate",
    "tower_score",
)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SINGLE_FEATURES = (
    "item_per_day",
    "item_click_rate",
    "item_long_rate",
    "popularity_rank",
    "user_author_click_rate",
    "user_tag_click_rate",
    "days_since_upload",
    "video_duration",
    "tower_score",
)
SAMPLE_USERS = 1000
LATENCY_USERS = 200
SCORE_CHUNK_USERS = 50

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass
class Settings:
    tower: two_tower.Config = field(default_factory=two_tower.Config)
    fine_params: dict[str, Any] = field(default_factory=dict)
    sample_users: int = SAMPLE_USERS
    latency_users: int = LATENCY_USERS
    bootstrap_samples: int = metrics.BOOTSTRAP_SAMPLES


def _est(e: metrics.Estimate) -> dict[str, float]:
    return {k: rounded(v) if isinstance(v, float) else v for k, v in e.as_dict().items()}


def _scores_by_user(
    builder: FeatureBuilder,
    ranker: fine.FineRanker,
    user_ids: IntArray,
    item_ids: IntArray,
) -> FloatArray:
    """Fused fine scores of every item in ``item_ids`` for each user, one row per user."""
    out = np.empty((len(user_ids), len(item_ids)))
    for start in range(0, len(user_ids), SCORE_CHUNK_USERS):
        block = user_ids[start : start + SCORE_CHUNK_USERS]
        x = builder.build(np.repeat(block, len(item_ids)), np.tile(item_ids, len(block)))
        fused = fine.fuse(ranker.predict(x))
        out[start : start + len(block)] = fused.reshape(len(block), len(item_ids))
    return out


def _latency(
    builder: FeatureBuilder, ranker: fine.FineRanker, user_ids: IntArray, kept: IntArray
) -> float:
    """Milliseconds per user to build features for and score the kept candidates."""
    valid = kept != cascade.PAD
    users = np.repeat(user_ids, valid.sum(axis=1))
    start = time.perf_counter()
    ranker.predict(builder.build(users, kept[valid]))
    return 1000 * (time.perf_counter() - start) / len(user_ids)


def _signed(feature: str, values: pd.Series) -> FloatArray:
    """Higher is better for every feature except ranks."""
    sign = -1.0 if feature == "popularity_rank" else 1.0
    out: FloatArray = sign * values.to_numpy(dtype=np.float64)
    return out


def _quantiles(values: pd.Series) -> dict[str, float]:
    q = values.quantile(list(QUANTILES))
    return {f"p{round(100 * p)}": rounded(v) for p, v in zip(QUANTILES, q, strict=True)}


@dataclass(frozen=True)
class Inputs:
    """Processed tables, the evaluation half of the random log and its user resamples."""

    standard: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    catalog: IntArray
    eval_users: IntArray
    rows: pd.DataFrame
    positives: list[set[int]]
    idx: IntArray


@dataclass(frozen=True)
class Retrieval:
    """Cutoff c2: tower vectors, both searches, per-user recall and the candidate pool."""

    window: pd.DataFrame
    builder: FeatureBuilder
    user_vecs: npt.NDArray[np.float32]
    item_vecs: npt.NDArray[np.float32]
    hnsw: IntArray
    personal: IntArray
    recall_values: dict[str, dict[int, FloatArray]]
    pool: IntArray
    pool_recall: FloatArray


@dataclass(frozen=True)
class FineModel:
    """Cutoff c1: the fine ranker with its window, feature builder and label rows."""

    window: pd.DataFrame
    builder: FeatureBuilder
    train_rows: pd.DataFrame
    valid_rows: pd.DataFrame
    ranker: fine.FineRanker


@dataclass(frozen=True)
class RowScores:
    """Features, labels and per-user ranking metrics of the evaluation rows."""

    x_eval: pd.DataFrame
    u: IntArray
    click: FloatArray
    weights: FloatArray
    ranking: dict[str, dict[str, FloatArray]]
    fusion: dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True)
class TruncationStudy:
    """Positions of the sampled users, the users timed, the output per N and catalog latency."""

    sample: IntArray
    lat_users: IntArray
    by_size: list[dict[str, Any]]
    catalog_latency: float


@dataclass(frozen=True)
class Primary:
    """Comparisons C1 and C2 with their Holm decisions."""

    c1: metrics.Comparison
    c2: metrics.Comparison
    rejected: dict[str, bool]
    p_resolution: dict[str, float]


def load_inputs(root: Path, bootstrap_samples: int) -> Inputs:
    processed = paths.processed_dir(root, "kuairand")
    standard = kuairand.load(processed, "log_standard_early")
    random_log = kuairand.load(processed, "log_random")
    users = kuairand.load(processed, "users")
    items = two_tower.item_side_features(kuairand.load(processed, "videos_basic"))
    catalog = items["video_id"].to_numpy(dtype=np.int64)

    eval_users, _ = splits.user_halves(random_log)
    rows = random_log[random_log["user_id"].isin(eval_users)].reset_index(drop=True)
    clicked = rows[rows["is_click"] == 1].groupby("user_id")["video_id"].agg(set)
    positives: list[set[int]] = [set(clicked.get(u, set())) for u in eval_users]
    idx = metrics.resample_indices(len(eval_users), samples=bootstrap_samples)
    return Inputs(standard, users, items, catalog, eval_users, rows, positives, idx)


def retrieve(data: Inputs, config: two_tower.Config, timings: dict[str, float]) -> Retrieval:
    """Cutoff c2: retrieval, coarse and fine ranking at evaluation."""
    catalog, eval_users, positives = data.catalog, data.eval_users, data.positives
    window2, builder2 = fit_cutoff(
        data.standard, splits.CUTOFF_EVAL, data.users, data.items, config, timings
    )
    tower2 = builder2.tower
    user_vecs = tower2.user_vectors(eval_users)
    item_vecs = tower2.item_vectors(catalog)
    k_max = min(max(RECALL_K), len(catalog))
    start = time.perf_counter()
    hnsw = ItemIndex(catalog, item_vecs).search(user_vecs, k_max)
    timings["hnsw_search_s"] = rounded(time.perf_counter() - start)
    start = time.perf_counter()
    # Exact search feeds every reported number; HNSW is checked against it (D018)
    personal = ItemIndex(catalog, item_vecs, exact=True).search(user_vecs, k_max)
    timings["exact_search_s"] = rounded(time.perf_counter() - start)

    popular = popularity.popularity(window2)
    trending = popularity.trending(window2, splits.CUTOFF_EVAL)
    sources = {
        "two_tower": personal,
        "popularity": np.tile(popular[:k_max], (len(eval_users), 1)),
        "trending": np.tile(trending[:k_max], (len(eval_users), 1)),
    }
    recall_values = {
        name: {k: metrics.recall_at_k(r, positives, k) for k in RECALL_K if k <= r.shape[1]}
        for name, r in sources.items()
    }
    pool = cascade.build_pool(personal[:, : cascade.PERSONAL_K], trending[: cascade.TRENDING_K])
    pool_recall = metrics.recall_at_k(pool, positives, pool.shape[1])
    return Retrieval(
        window2, builder2, user_vecs, item_vecs, hnsw, personal, recall_values, pool, pool_recall
    )


def fit_fine_ranker(data: Inputs, s: Settings, timings: dict[str, float]) -> FineModel:
    """Cutoff c1: fine-ranker training rows."""
    standard = data.standard
    window1, builder1 = fit_cutoff(
        standard, splits.CUTOFF_TRAIN, data.users, data.items, s.tower, timings
    )
    train_rows = splits.between(standard, *splits.LABELS_TRAIN)
    valid_rows = splits.between(standard, *splits.LABELS_VALID)
    start = time.perf_counter()
    ranker = fine.FineRanker.fit(builder1, train_rows, valid_rows, s.fine_params)
    timings["fine_fit_s"] = rounded(time.perf_counter() - start)
    return FineModel(window1, builder1, train_rows, valid_rows, ranker)


def score_evaluation_rows(
    data: Inputs, builder2: FeatureBuilder, ranker: fine.FineRanker
) -> RowScores:
    """Per-user ranking metrics of every scorer and fusion weight on the evaluation rows."""
    rows, eval_users, idx = data.rows, data.eval_users, data.idx
    x_eval = builder2.build_rows(rows)
    pred = ranker.predict(x_eval)
    u = rows["user_id"].to_numpy(dtype=np.int64)
    click = rows["is_click"].to_numpy(float)
    long_view = rows["long_view"].to_numpy(float)
    scorers: dict[str, FloatArray] = {
        "fused": fine.fuse(pred),
        "click_head": pred["is_click"],
        "long_view_head": pred["long_view"],
        "two_tower": x_eval["tower_score"].to_numpy(dtype=np.float64),
        "popularity": -x_eval["popularity_rank"].to_numpy(dtype=np.float64),
    }
    weights = metrics.impressions(u, eval_users)

    def per_user(score: FloatArray) -> dict[str, FloatArray]:
        return {
            "gauc_click": metrics.user_auc(u, score, click, eval_users),
            "gauc_long_view": metrics.user_auc(u, score, long_view, eval_users),
            "ndcg5_click": metrics.user_ndcg(u, score, click, eval_users),
        }

    ranking = {name: per_user(score) for name, score in scorers.items()}
    fusion = {
        str(w): {k: _est(metrics.estimate(v, idx)) for k, v in per_user(fine.fuse(pred, w)).items()}
        for w in FUSION_GRID
    }
    return RowScores(x_eval, u, click, weights, ranking, fusion)


def truncation_study(
    data: Inputs,
    retrieval: Retrieval,
    ranker: fine.FineRanker,
    s: Settings,
    timings: dict[str, float],
) -> TruncationStudy:
    """Survival, agreement, regret and fine-ranking latency at each coarse top-N."""
    catalog, eval_users, positives, idx = data.catalog, data.eval_users, data.positives, data.idx
    builder2, pool = retrieval.builder, retrieval.pool
    user_vecs, item_vecs = retrieval.user_vecs, retrieval.item_vecs
    rng = np.random.default_rng(splits.SPLIT_SEED)
    sample = np.sort(
        rng.choice(len(eval_users), min(s.sample_users, len(eval_users)), replace=False)
    )
    start = time.perf_counter()
    catalog_scores = _scores_by_user(builder2, ranker, eval_users[sample], catalog)
    timings["catalog_scoring_s"] = rounded(time.perf_counter() - start)
    sizes = [n for n in TRUNCATION_N if n < pool.shape[1]] + [pool.shape[1]]
    study = truncation.study(
        pool, user_vecs, catalog, item_vecs, positives, sample, catalog_scores, sizes
    )
    sample_idx = metrics.resample_indices(len(sample), samples=s.bootstrap_samples)
    lat_users = eval_users[sample[: s.latency_users]]
    truncation_out = []
    for result in study:
        kept = cascade.coarse(
            pool[sample[: s.latency_users]],
            user_vecs[sample[: s.latency_users]],
            catalog,
            item_vecs,
            result.n,
        )
        truncation_out.append(
            {
                "n": result.n,
                "survival": _est(metrics.estimate(result.survival, idx)) | {"tier": "A"},
                "top1_agreement": _est(metrics.estimate(result.agreement, sample_idx))
                | {"tier": "model-internal"},
                "score_regret": _est(metrics.estimate(result.regret, sample_idx))
                | {"tier": "model-internal"},
                "fine_ms_per_user_batched": rounded(_latency(builder2, ranker, lat_users, kept)),
            }
        )
    catalog_latency = _latency(builder2, ranker, lat_users, np.tile(catalog, (len(lat_users), 1)))
    return TruncationStudy(sample, lat_users, truncation_out, catalog_latency)


def primary_comparisons(
    recall_values: dict[str, dict[int, FloatArray]],
    ranking: dict[str, dict[str, FloatArray]],
    idx: IntArray,
    bootstrap_samples: int,
) -> Primary:
    c1 = metrics.compare(recall_values["two_tower"][200], recall_values["popularity"][200], idx)
    c2 = metrics.compare(ranking["fused"]["gauc_click"], ranking["two_tower"]["gauc_click"], idx)
    rejected = stats.holm({"C1": c1.p_value, "C2": c2.p_value})
    # With B resamples the smallest nonzero p-value is 2/B; 0 means below it
    p_resolution = {"p_value_resolution": rounded(2 / bootstrap_samples)}
    return Primary(c1, c2, rejected, p_resolution)


def validation_diagnostics(
    model: FineModel, bootstrap_samples: int
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """The same scorers on the fine ranker's own validation rows (standard logs, not random).

    Returns the features of those rows, each scorer's GAUC and each booster's top-10 gains.
    """
    builder1, valid_rows, ranker = model.builder, model.valid_rows, model.ranker
    x_valid = builder1.build_rows(valid_rows)
    pred_valid = ranker.predict(x_valid)
    v_users = valid_rows["user_id"].to_numpy(dtype=np.int64)
    v_order = np.unique(v_users)
    v_idx = metrics.resample_indices(len(v_order), samples=bootstrap_samples)
    in_distribution = {
        name: _est(
            metrics.estimate(
                metrics.user_auc(v_users, score, valid_rows["is_click"].to_numpy(float), v_order),
                v_idx,
            )
        )
        for name, score in {
            "fused": fine.fuse(pred_valid),
            "click_head": pred_valid["is_click"],
            "two_tower": x_valid["tower_score"].to_numpy(dtype=np.float64),
            "popularity": -x_valid["popularity_rank"].to_numpy(dtype=np.float64),
        }.items()
    }
    importance = {
        target: dict(
            sorted(
                zip(
                    x_valid.columns,
                    (round(float(g), 1) for g in booster.feature_importance("gain")),
                    strict=True,
                ),
                key=lambda kv: -kv[1],
            )[:10]
        )
        for target, booster in ranker.boosters.items()
    }
    return x_valid, in_distribution, importance


def feature_drift(
    model: FineModel, builder2: FeatureBuilder, x_eval: pd.DataFrame, x_valid: pd.DataFrame
) -> dict[str, dict[str, dict[str, float]]]:
    """Feature drift between the two cutoffs."""
    x_train = model.builder.build_rows(model.train_rows)
    drift: dict[str, dict[str, dict[str, float]]] = {
        f: {"c1_training_rows": _quantiles(x_train[f]), "c2_evaluation_rows": _quantiles(x_eval[f])}
        for f in DRIFT_FEATURES
    }
    # Separates the refit of the tower from the change of rows: same standard rows, c2 tower
    x_valid_c2 = builder2.build_rows(model.valid_rows)
    drift["tower_score"]["c1_validation_rows"] = _quantiles(x_valid["tower_score"])
    drift["tower_score"]["c2_on_the_same_validation_rows"] = _quantiles(x_valid_c2["tower_score"])
    return drift


def single_feature_gauc(data: Inputs, scores: RowScores) -> dict[str, dict[str, float]]:
    """Each numeric feature alone as a score on the random exposures (Tier A, secondary)."""
    u, x_eval, click = scores.u, scores.x_eval, scores.click
    eval_users, idx = data.eval_users, data.idx
    return {
        f: _est(
            metrics.estimate(
                metrics.user_auc(u, _signed(f, x_eval[f]), click, eval_users),
                idx,
            )
        )
        for f in SINGLE_FEATURES
    }


def setup_counts(
    data: Inputs, retrieval: Retrieval, model: FineModel, sample: IntArray
) -> dict[str, Any]:
    return {
        "evaluation_users": len(data.eval_users),
        "evaluation_impressions": len(data.rows),
        "users_with_a_click": int(sum(1 for p in data.positives if p)),
        "catalog_items": len(data.catalog),
        "window_rows": {"c1": len(model.window), "c2": len(retrieval.window)},
        "fine_training_rows": len(model.train_rows),
        "fine_validation_rows": len(model.valid_rows),
        "fine_best_rounds": model.ranker.best_rounds,
        "truncation_sample_users": len(sample),
    }


def hnsw_overlap(hnsw: IntArray, personal: IntArray) -> dict[str, float]:
    return {
        "hnsw_overlap_at_200": rounded(overlap(hnsw[:, :200], personal[:, :200])),
        "hnsw_overlap_at_500": rounded(overlap(hnsw, personal)),
    }


def recall_estimates(retrieval: Retrieval, idx: IntArray) -> dict[str, Any]:
    pool, pool_recall = retrieval.pool, retrieval.pool_recall
    return {
        name: {str(k): _est(metrics.estimate(v, idx)) | {"tier": "A"} for k, v in by_k.items()}
        for name, by_k in retrieval.recall_values.items()
    } | {"pool": {str(pool.shape[1]): _est(metrics.estimate(pool_recall, idx)) | {"tier": "A"}}}


def ranking_estimates(scores: RowScores, idx: IntArray) -> dict[str, Any]:
    weights = scores.weights
    return {
        name: {
            "gauc_click": _est(metrics.estimate(m["gauc_click"], idx)),
            "gauc_click_impression_weighted": _est(metrics.estimate(m["gauc_click"], idx, weights)),
            "gauc_long_view": _est(metrics.estimate(m["gauc_long_view"], idx)),
            "ndcg5_click": _est(metrics.estimate(m["ndcg5_click"], idx)),
            "tier": "A",
        }
        for name, m in scores.ranking.items()
    }


def primary_entries(primary: Primary) -> dict[str, Any]:
    c1, c2, rejected, p_resolution = primary.c1, primary.c2, primary.rejected, primary.p_resolution
    return {
        "C1_two_tower_vs_popularity_recall200": _est(c1.difference)
        | {"p_value": rounded(c1.p_value), "holm_rejected": rejected["C1"]}
        | p_resolution,
        "C2_fused_vs_two_tower_gauc_click": _est(c2.difference)
        | {"p_value": rounded(c2.p_value), "holm_rejected": rejected["C2"]}
        | p_resolution,
    }


def run(root: Path, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or Settings()
    timings: dict[str, float] = {}
    data = load_inputs(root, s.bootstrap_samples)
    retrieval = retrieve(data, s.tower, timings)
    model = fit_fine_ranker(data, s, timings)
    scores = score_evaluation_rows(data, retrieval.builder, model.ranker)
    cut = truncation_study(data, retrieval, model.ranker, s, timings)
    primary = primary_comparisons(
        retrieval.recall_values, scores.ranking, data.idx, s.bootstrap_samples
    )
    x_valid, in_distribution, importance = validation_diagnostics(model, s.bootstrap_samples)
    drift = feature_drift(model, retrieval.builder, scores.x_eval, x_valid)
    single_feature = single_feature_gauc(data, scores)

    return {
        "setup": setup_counts(data, retrieval, model, cut.sample),
        "ann": hnsw_overlap(retrieval.hnsw, retrieval.personal),
        "recall": recall_estimates(retrieval, data.idx),
        "ranking": ranking_estimates(scores, data.idx),
        "fusion_weight": scores.fusion,
        "truncation": cut.by_size,
        "catalog_fine_ms_per_user_batched": rounded(cut.catalog_latency),
        "latency_note": (
            f"features and prediction for {len(cut.lat_users)} users in one call, divided by the "
            "number of users; not a per-request latency"
        ),
        "primary": primary_entries(primary),
        "diagnostics": {
            "gauc_click_on_fine_validation_rows": in_distribution
            | {
                "tier": "not unbiased: standard-log impressions",
                "note": (
                    "these rows also drive early stopping, so the fine ranker's value is optimistic"
                ),
            },
            "fine_feature_gain_top10": importance,
            "single_feature_gauc_click": single_feature | {"tier": "A"},
        },
        "feature_drift": drift,
        "timings_s": timings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate recall and ranking.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    result = run(args.root)
    out = args.root / paths.RESULTS / "ranking.json"
    write_json(out, result, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
