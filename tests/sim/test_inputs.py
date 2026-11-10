from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from push_lab.data import fixtures, kuairand, paths, splits
from push_lab.rank import fine
from push_lab.rank.features import fit_cutoff
from push_lab.recall import popularity, two_tower
from push_lab.sim import build, inputs
from push_lab.sim.world import SLOTS


def test_local_hour_is_utc_plus_8() -> None:
    # 2022-04-22 00:30 UTC is 08:30 in UTC+8
    ms = np.array([pd.Timestamp("2022-04-22 00:30", tz="UTC").value // 1_000_000])
    assert inputs.local_hour(ms)[0] == 8


def test_window_counts_and_smoothing() -> None:
    def ms(hour: int) -> int:
        return int(pd.Timestamp(f"2022-04-22 {hour:02d}:10", tz="Asia/Shanghai").value // 1e6)

    log = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 7],
            "time_ms": [ms(9), ms(13), ms(5), ms(22), ms(9)],
        }
    )
    counts = inputs.window_counts(log, np.array([1, 2, 3]))
    # 05:00 is outside every window; user 7 is not in the population
    np.testing.assert_array_equal(counts, [[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    profile = inputs.smoothed_profile(counts, prior=10.0)
    np.testing.assert_allclose(profile.sum(axis=1), 1.0)
    pi = counts.sum(axis=0) / counts.sum()
    np.testing.assert_allclose(profile[2], pi)


def test_hourly_counts_sum_to_the_windows() -> None:
    def ms(hour: int) -> int:
        return int(pd.Timestamp(f"2022-04-12 {hour:02d}:40", tz="Asia/Shanghai").value // 1e6)

    log = pd.DataFrame({"user_id": [1, 1, 2, 2, 9], "time_ms": [ms(9), ms(5), ms(23), ms(23), 0]})
    users = np.array([1, 2])
    hourly = inputs.hourly_counts(log, users)
    assert hourly.shape == (2, 24)
    assert hourly[0, 9] == 1 and hourly[0, 5] == 1 and hourly[1, 23] == 2
    np.testing.assert_array_equal(
        inputs.windows_from_hourly(hourly), inputs.window_counts(log, users)
    )


def test_visit_rates_are_smoothed_toward_the_mean() -> None:
    log = pd.DataFrame({"user_id": [1, 1, 1, 2], "date": [1, 2, 2, 1]})
    v = inputs.visit_rates(log, np.array([1, 2, 3]), days=4)
    mean = (2 / 4 + 1 / 4 + 0) / 3
    np.testing.assert_allclose(v, (np.array([2, 1, 0]) + 2 * mean) / 6)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_build_runs_on_synthetic_data(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=4_000)
    settings = build.Settings(
        inputs=inputs.Settings(
            tower=two_tower.Config(epochs=1, batch=256, threads=1),
            fine_params={"min_data_in_leaf": 5, "num_threads": 1},
            chunk_users=16,
        ),
        bootstrap_samples=50,
    )
    result = build.run(tmp_path, settings)
    assert result["inputs"]["users"] == fixtures.N_USERS
    assert len(result["inputs"]["slot_shares"]["simulator_profile"]) == SLOTS
    assert result["exploration"]["min_propensity"] == pytest.approx(0.075 / 60)
    cached = inputs.Inputs.load(paths.processed_dir(tmp_path, "sim") / inputs.CACHE)
    for c in cached.lists.values():
        assert c.items.shape == (fixtures.N_USERS, inputs.CANDIDATES)
        assert ((c.p0 > 0) & (c.p0 < 1)).all()
        # Items within a user's list are distinct
        assert all(len(set(row)) == len(row) for row in c.items)
    order = cached.lists["cascade"].score
    assert (np.diff(order, axis=1) <= 0).all()
    # The policy profile is the smoothed windowing of the policy-period hourly counts
    np.testing.assert_allclose(
        cached.profile_policy,
        inputs.smoothed_profile(inputs.windows_from_hourly(cached.hourly_policy)),
    )
    assert ((cached.visit_rate_policy > 0) & (cached.visit_rate_policy < 1)).all()
    shape = cached.lists["cascade"].items.shape
    assert cached.creator_clicks.shape == cached.trend_rank.shape == shape
    assert (cached.creator_clicks >= 0).all()
    assert set(np.unique(cached.trend_rank)) <= set(range(inputs.TREND_TOP + 1))


def test_copy_attributes() -> None:
    window = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "video_id": [10, 11, 12, 10, 12],
            "is_click": [1, 1, 0, 0, 1],
        }
    )
    # Videos 10 and 11 share author 7; 12 and 13 are by author 8
    videos = pd.DataFrame({"video_id": [10, 11, 12, 13], "author_id": [7, 7, 8, 8]})
    items = np.array([[11, 13, 12], [13, 10, 11]])
    trending = np.array([12, 10, *range(100, 300)])
    creator, rank = inputs.copy_attributes(window, videos, trending, np.array([1, 2]), items)
    np.testing.assert_array_equal(creator, [[2, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(rank, [[0, 0, 1], [0, 2, 0]])
    items_far = np.array([[trending[99], trending[100]]])
    _, rank_far = inputs.copy_attributes(window, videos, trending, np.array([1]), items_far)
    np.testing.assert_array_equal(rank_far, [[100, 0]])


TINY = inputs.Settings(
    tower=two_tower.Config(epochs=1, batch=256, threads=1),
    fine_params={"min_data_in_leaf": 5, "num_threads": 1},
    chunk_users=16,
)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_response_model_uses_only_the_simulator_half(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=4_000)
    fitted = inputs.build(tmp_path, TINY)
    random_log = kuairand.load(paths.processed_dir(tmp_path, "kuairand"), "log_random")
    evaluation, simulator = splits.user_halves(random_log)
    assert set(fitted.response_users) <= set(simulator)
    assert not set(fitted.response_users) & set(evaluation)
    assert set(fitted.validation.users) <= set(simulator)


def test_policy_period_ends_before_the_simulated_period() -> None:
    log = pd.DataFrame({"date": [20220409, 20220421, 20220422, 20220508]})
    assert inputs.policy_period(log)["date"].max() == 20220421


class TowerScore:
    """Stands in for the response model: returns the two-tower score feature."""

    best_iteration = 0

    def predict(self, x: pd.DataFrame, **_: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        return x["tower_score"].to_numpy(dtype=np.float64)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_p0_is_aligned_with_the_items(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=4_000)
    processed = paths.processed_dir(tmp_path, "kuairand")
    standard = kuairand.load(processed, "log_standard_early")
    users = kuairand.load(processed, "users")
    items = two_tower.item_side_features(kuairand.load(processed, "videos_basic"))
    timings: dict[str, float] = {}
    window, builder = fit_cutoff(standard, splits.CUTOFF_EVAL, users, items, TINY.tower, timings)
    _, builder1 = fit_cutoff(standard, splits.CUTOFF_TRAIN, users, items, TINY.tower, timings)
    ranker = fine.FineRanker.fit(
        builder1,
        splits.between(standard, *splits.LABELS_TRAIN),
        splits.between(standard, *splits.LABELS_VALID),
        TINY.fine_params,
    )
    user_ids = np.sort(users["user_id"].to_numpy(dtype=np.int64))
    lists = inputs.candidate_lists(
        builder,
        ranker,
        cast(lgb.Booster, TowerScore()),
        popularity.trending(window, splits.CUTOFF_EVAL),
        items["video_id"].to_numpy(dtype=np.int64),
        user_ids,
        TINY,
    )
    # With p0 equal to the tower score, each item's p0 is its own two-tower score
    tower = lists["two_tower"]
    np.testing.assert_allclose(tower.p0, tower.score, rtol=1e-5, atol=1e-6)
    assert (np.diff(tower.score, axis=1) <= 1e-6).all()
