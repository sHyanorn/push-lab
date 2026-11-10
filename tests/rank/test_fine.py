import numpy as np
import pandas as pd
import pytest

from push_lab.data import fixtures, splits
from push_lab.rank import fine
from push_lab.rank.features import FeatureBuilder, smoothed
from push_lab.recall import two_tower


def _parts(log: pd.DataFrame, cutoff: int) -> FeatureBuilder:
    tables = fixtures.processed_tables(n=4_000)
    items = two_tower.item_side_features(tables["kuairand", "videos_basic"])
    users = tables["kuairand", "users"]
    window = splits.window(log, cutoff)
    config = two_tower.Config(epochs=1, batch=128, threads=1)
    tower = two_tower.TwoTower(users, items, window, config)
    tower.fit()
    return FeatureBuilder(window, cutoff, users, items, tower)


def _log() -> pd.DataFrame:
    log = fixtures.processed_tables(n=4_000)["kuairand", "log_standard_early"]
    rng = np.random.default_rng(0)
    dates = rng.choice([20220410, 20220412, 20220414, 20220416, 20220419, 20220420], len(log))
    return log.assign(date=dates)


def test_smoothed_rate_is_prior_without_impressions() -> None:
    got = smoothed(pd.Series([0.0]), pd.Series([0.0]), 0.3)
    assert got.tolist() == [pytest.approx(0.3)]


def test_builder_rejects_rows_on_or_after_cutoff() -> None:
    log = _log()
    tables = fixtures.processed_tables(n=100)
    items = two_tower.item_side_features(tables["kuairand", "videos_basic"])
    tower = two_tower.TwoTower(tables["kuairand", "users"], items, log, two_tower.Config())
    with pytest.raises(ValueError, match="on or after the cutoff"):
        FeatureBuilder(log, splits.CUTOFF_TRAIN, tables["kuairand", "users"], items, tower)


def test_features_ignore_future_rows() -> None:
    log = _log()
    future = log.assign(date=20220421, is_click=1, long_view=1)
    a = _parts(log, splits.CUTOFF_TRAIN)
    b = _parts(pd.concat([log, future], ignore_index=True), splits.CUTOFF_TRAIN)
    users = np.arange(fixtures.N_USERS, dtype=np.int64)
    items = (users * 7) % fixtures.N_VIDEOS
    pd.testing.assert_frame_equal(a.build(users, items), b.build(users, items))


def test_fit_is_deterministic() -> None:
    log = _log()
    builder = _parts(log, splits.CUTOFF_TRAIN)
    labels = splits.between(log, 20220416, 20220419)
    valid = splits.between(log, 20220420, 20220420)
    params = {"min_data_in_leaf": 5, "num_threads": 1}
    a = fine.FineRanker.fit(builder, labels, valid, params)
    b = fine.FineRanker.fit(builder, labels, valid, params)
    x = builder.build(valid["user_id"].to_numpy(), valid["video_id"].to_numpy())
    pa, pb = a.predict(x), b.predict(x)
    for target in fine.TARGETS:
        np.testing.assert_array_equal(pa[target], pb[target])
    fused = fine.fuse(pa)
    assert ((fused >= 0) & (fused <= 1)).all()


def test_early_stopping_reads_the_validation_rows() -> None:
    # Fixture clicks are noise, so validation loss is lowest after a few rounds
    log = _log()
    builder = _parts(log, splits.CUTOFF_TRAIN)
    labels = splits.between(log, 20220416, 20220419)
    valid = splits.between(log, 20220420, 20220420)
    ranker = fine.FineRanker.fit(builder, labels, valid, {"min_data_in_leaf": 5, "num_threads": 1})
    # LightGBM disables early stopping when the validation set is the training set (best 0)
    assert all(0 < r < fine.MAX_ROUNDS // 4 for r in ranker.best_rounds.values())
