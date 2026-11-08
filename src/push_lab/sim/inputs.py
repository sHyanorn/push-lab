"""Simulator inputs fitted on KuaiRand (METHODS Section 11.1).

Per-user arrays are cached under ``data/processed/sim/`` and never committed.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab.data import kuairand, paths
from push_lab.rank import cascade, fine, splits
from push_lab.rank.evaluate import fit_cutoff
from push_lab.rank.features import CATEGORICAL, FeatureBuilder
from push_lab.recall import popularity, two_tower
from push_lab.recall.index import ItemIndex
from push_lab.sim.world import DAYS, SLOTS, WINDOWS

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# One candidate per send slot of the horizon, so a list never runs out
CANDIDATES = DAYS * SLOTS
COARSE_N = 200
VALID_SHARE = 0.2
SPLIT_SEED = 0
PROFILE_PRIOR = 10.0
VISIT_PRIOR_DAYS = 2.0
SIM_PERIOD = (20220422, 20220508)
POLICY_PERIOD = (splits.FIRST_DATE, 20220421)
UTC_OFFSET_HOURS = 8
CHUNK_USERS = 500
LISTS = ("cascade", "two_tower")
CACHE = "inputs.npz"


@dataclass
class Settings:
    tower: two_tower.Config = field(default_factory=two_tower.Config)
    fine_params: dict[str, Any] = field(default_factory=dict)
    candidates: int = CANDIDATES
    coarse_n: int = COARSE_N
    chunk_users: int = CHUNK_USERS


@dataclass(frozen=True)
class CandidateList:
    """Per user, items in the policy's ranking order, the policy-side score that produced
    the order, and the response model's click probability."""

    items: IntArray
    score: FloatArray
    p0: FloatArray


@dataclass(frozen=True)
class Inputs:
    """Simulator-side arrays (``visit_rate``, ``profile_sim``, ``p0``) and policy-side arrays
    from the policy period (``profile_policy``, ``hourly_policy``, ``visit_rate_policy``)."""

    user_ids: IntArray
    visit_rate: FloatArray
    profile_sim: FloatArray
    profile_policy: FloatArray
    hourly_policy: FloatArray
    visit_rate_policy: FloatArray
    lists: dict[str, CandidateList]

    ARRAYS = (
        "user_ids",
        "visit_rate",
        "profile_sim",
        "profile_policy",
        "hourly_policy",
        "visit_rate_policy",
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {name: getattr(self, name) for name in self.ARRAYS}
        for name, c in self.lists.items():
            arrays |= {f"{name}.items": c.items, f"{name}.score": c.score, f"{name}.p0": c.p0}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> Inputs:
        with np.load(path) as z:
            lists = {
                name: CandidateList(z[f"{name}.items"], z[f"{name}.score"], z[f"{name}.p0"])
                for name in LISTS
            }
            return cls(**{name: z[name] for name in cls.ARRAYS}, lists=lists)


@dataclass(frozen=True)
class Validation:
    """Held-out simulator-half rows scored by the response model."""

    users: IntArray
    labels: FloatArray
    predicted: FloatArray
    best_rounds: int
    rows_train: int
    gain_share: dict[str, float]


def local_hour(time_ms: npt.NDArray[np.integer]) -> IntArray:
    """Hour of day in UTC+8, the assumed time zone of KuaiRand (METHODS Section 11.1)."""
    hours: IntArray = ((time_ms.astype(np.int64) // 3_600_000) + UTC_OFFSET_HOURS) % 24
    return hours


def hourly_shares(log: pd.DataFrame) -> FloatArray:
    counts = np.bincount(local_hour(log["time_ms"].to_numpy()), minlength=24)
    out: FloatArray = counts / counts.sum()
    return out


def hourly_counts(log: pd.DataFrame, user_ids: IntArray) -> FloatArray:
    """Rows per user in each local hour; users outside ``user_ids`` drop out."""
    hour = local_hour(log["time_ms"].to_numpy())
    log_users = log["user_id"].to_numpy(dtype=np.int64)
    pos = np.searchsorted(user_ids, log_users)
    known = (pos < len(user_ids)) & (user_ids[np.minimum(pos, len(user_ids) - 1)] == log_users)
    counts = np.zeros((len(user_ids), 24))
    np.add.at(counts, (pos[known], hour[known]), 1.0)
    return counts


def windows_from_hourly(hourly: FloatArray) -> FloatArray:
    """Rows in each send slot's response window; hours outside the windows drop out."""
    out: FloatArray = np.stack([hourly[:, a:b].sum(axis=1) for a, b in WINDOWS], axis=1)
    return out


def window_counts(log: pd.DataFrame, user_ids: IntArray) -> FloatArray:
    """Rows per user in each send slot's response window; rows outside the windows drop out."""
    return windows_from_hourly(hourly_counts(log, user_ids))


def smoothed_profile(counts: FloatArray, prior: float = PROFILE_PRIOR) -> FloatArray:
    """a[u, s] = (n[u, s] + prior * pi[s]) / (n[u] + prior), pi the population shares."""
    pi = counts.sum(axis=0) / counts.sum()
    out: FloatArray = (counts + prior * pi) / (counts.sum(axis=1, keepdims=True) + prior)
    return out


def visit_rates(log: pd.DataFrame, user_ids: IntArray, days: int) -> FloatArray:
    """Smoothed share of days with any row: (n + 2 v_bar) / (days + 2)."""
    active = log.groupby("user_id")["date"].nunique()
    n = active.reindex(user_ids, fill_value=0).to_numpy(dtype=np.float64)
    v_bar = float(np.mean(n / days))
    out: FloatArray = np.asarray(
        (n + VISIT_PRIOR_DAYS * v_bar) / (days + VISIT_PRIOR_DAYS), dtype=np.float64
    )
    return out


def _days(first: int, last: int) -> int:
    return (splits.to_date(last) - splits.to_date(first)).days + 1


def fit_response(
    builder: FeatureBuilder, rows: pd.DataFrame, fine_params: dict[str, Any]
) -> tuple[lgb.Booster, Validation]:
    """Click model on simulator-half random exposures; 20% of users for early stopping."""
    users = np.sort(rows["user_id"].unique()).astype(np.int64)
    permuted = np.random.default_rng(SPLIT_SEED).permutation(users)
    valid_users = permuted[: round(len(users) * VALID_SHARE)]
    is_valid = rows["user_id"].isin(valid_users).to_numpy()
    u = rows["user_id"].to_numpy(dtype=np.int64)
    i = rows["video_id"].to_numpy(dtype=np.int64)
    y = rows["is_click"].to_numpy(dtype=np.float64)
    x = builder.build(u, i)
    d_train = lgb.Dataset(x[~is_valid], y[~is_valid], categorical_feature=CATEGORICAL)
    d_valid = lgb.Dataset(
        x[is_valid], y[is_valid], categorical_feature=CATEGORICAL, reference=d_train
    )
    booster = lgb.train(
        {**fine.PARAMS, **fine_params},
        d_train,
        num_boost_round=fine.MAX_ROUNDS,
        valid_sets=[d_valid],
        callbacks=[lgb.early_stopping(fine.EARLY_STOPPING, verbose=False)],
    )
    predicted = np.asarray(
        booster.predict(x[is_valid], num_iteration=booster.best_iteration), dtype=np.float64
    )
    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    shares = dict(zip(booster.feature_name(), gain / gain.sum(), strict=True))
    return booster, Validation(
        u[is_valid],
        y[is_valid],
        predicted,
        booster.best_iteration,
        int((~is_valid).sum()),
        {k: float(v) for k, v in sorted(shares.items(), key=lambda kv: -kv[1])},
    )


def _predict(booster: lgb.Booster, x: pd.DataFrame) -> FloatArray:
    out: FloatArray = np.asarray(
        booster.predict(x, num_iteration=booster.best_iteration), dtype=np.float64
    )
    return out


def candidate_lists(
    builder: FeatureBuilder,
    ranker: fine.FineRanker,
    response: lgb.Booster,
    trending: IntArray,
    catalog: IntArray,
    user_ids: IntArray,
    s: Settings,
) -> dict[str, CandidateList]:
    """The cascade's and the two-tower score's top items per user, with p0 for each."""
    if s.candidates > min(s.coarse_n, len(catalog)):
        raise ValueError("the lists need at least as many coarse survivors and items as candidates")
    tower = builder.tower
    item_vecs = tower.item_vectors(catalog)
    index = ItemIndex(catalog, item_vecs, exact=True)
    k_personal = min(cascade.PERSONAL_K, len(catalog))
    c = s.candidates
    out = {
        name: CandidateList(
            np.empty((len(user_ids), c), dtype=np.int64),
            np.empty((len(user_ids), c)),
            np.empty((len(user_ids), c)),
        )
        for name in LISTS
    }
    for start in range(0, len(user_ids), s.chunk_users):
        block = user_ids[start : start + s.chunk_users]
        rows = slice(start, start + len(block))
        user_vecs = tower.user_vectors(block)
        personal = index.search(user_vecs, k_personal)
        pool = cascade.build_pool(personal, trending[: cascade.TRENDING_K])
        kept = cascade.coarse(pool, user_vecs, catalog, item_vecs, s.coarse_n)
        valid = kept != cascade.PAD
        fused = np.full(kept.shape, -np.inf)
        fused[valid] = fine.fuse(
            ranker.predict(builder.build(np.repeat(block, valid.sum(axis=1)), kept[valid]))
        )
        order = np.argsort(-fused, axis=1, kind="stable")[:, :c]
        cascade_items = np.take_along_axis(kept, order, axis=1)
        tower_items = personal[:, :c]
        lookup = np.full(int(catalog.max()) + 1, -1, dtype=np.int64)
        lookup[catalog] = np.arange(len(catalog))
        tower_scores = np.einsum("ud,ucd->uc", user_vecs, item_vecs[lookup[tower_items]])
        for name, items, score in (
            ("cascade", cascade_items, np.take_along_axis(fused, order, axis=1)),
            ("two_tower", tower_items, tower_scores.astype(np.float64)),
        ):
            x = builder.build(np.repeat(block, c), items.ravel())
            out[name].items[rows] = items
            out[name].score[rows] = score
            out[name].p0[rows] = _predict(response, x).reshape(len(block), c)
    return out


def policy_period(standard: pd.DataFrame) -> pd.DataFrame:
    """Standard-log rows the send-time policy may learn from (D014): before 04-22."""
    return splits.between(standard, *POLICY_PERIOD)


def date_agreement(log: pd.DataFrame, offsets: tuple[int, ...] = (7, 8, 9)) -> dict[int, float]:
    """Share of rows whose ``date`` equals the calendar date of ``time_ms`` at each UTC offset."""
    ms = log["time_ms"].to_numpy(dtype=np.int64)
    out = {}
    for h in offsets:
        local = pd.to_datetime(ms + h * 3_600_000, unit="ms").strftime("%Y%m%d").astype(int)
        out[h] = float(np.mean(local.to_numpy() == log["date"].to_numpy()))
    return out


@dataclass(frozen=True)
class Fitted:
    inputs: Inputs
    validation: Validation
    response_users: IntArray
    hourly_sim: FloatArray
    date_agreement: dict[int, float]
    timings: dict[str, float]


def build(root: Path, settings: Settings | None = None) -> Fitted:
    s = settings or Settings()
    processed = paths.processed_dir(root, "kuairand")
    standard = kuairand.load(processed, "log_standard_early")
    late = kuairand.load(processed, "log_standard_late")
    random_log = kuairand.load(processed, "log_random")
    users = kuairand.load(processed, "users")
    items = two_tower.item_side_features(kuairand.load(processed, "videos_basic"))
    catalog = items["video_id"].to_numpy(dtype=np.int64)
    user_ids = np.sort(users["user_id"].to_numpy(dtype=np.int64))
    timings: dict[str, float] = {}

    window2, builder2 = fit_cutoff(standard, splits.CUTOFF_EVAL, users, items, s.tower, timings)
    _, builder1 = fit_cutoff(standard, splits.CUTOFF_TRAIN, users, items, s.tower, timings)
    start = time.perf_counter()
    ranker = fine.FineRanker.fit(
        builder1,
        splits.between(standard, *splits.LABELS_TRAIN),
        splits.between(standard, *splits.LABELS_VALID),
        s.fine_params,
    )
    timings["fine_fit_s"] = round(time.perf_counter() - start, 1)

    _, sim_half = splits.user_halves(random_log)
    sim_rows = random_log[random_log["user_id"].isin(sim_half)].reset_index(drop=True)
    start = time.perf_counter()
    response, validation = fit_response(builder2, sim_rows, s.fine_params)
    timings["response_fit_s"] = round(time.perf_counter() - start, 1)

    start = time.perf_counter()
    lists = candidate_lists(
        builder2,
        ranker,
        response,
        popularity.trending(window2, splits.CUTOFF_EVAL),
        catalog,
        user_ids,
        s,
    )
    timings["candidates_s"] = round(time.perf_counter() - start, 1)

    columns = ["user_id", "date", "time_ms"]
    period = pd.concat(
        [
            splits.between(late, *SIM_PERIOD)[columns],
            splits.between(random_log, *SIM_PERIOD)[columns],
        ]
    )
    early = policy_period(standard)[columns]
    hourly_policy = hourly_counts(early, user_ids)
    inputs = Inputs(
        user_ids=user_ids,
        visit_rate=visit_rates(period, user_ids, _days(*SIM_PERIOD)),
        profile_sim=smoothed_profile(window_counts(period, user_ids)),
        profile_policy=smoothed_profile(windows_from_hourly(hourly_policy)),
        hourly_policy=hourly_policy,
        visit_rate_policy=visit_rates(early, user_ids, _days(*POLICY_PERIOD)),
        lists=lists,
    )
    return Fitted(
        inputs,
        validation,
        np.sort(sim_rows["user_id"].unique()).astype(np.int64),
        hourly_shares(period),
        date_agreement(period),
        timings,
    )
