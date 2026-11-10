import numpy as np
import pandas as pd

from push_lab.data import fixtures, splits


def test_window_excludes_cutoff_and_later() -> None:
    log = pd.DataFrame({"date": [20220414, 20220415, 20220421, 20220422], "x": [1, 2, 3, 4]})
    assert splits.window(log, splits.CUTOFF_TRAIN)["date"].max() < splits.CUTOFF_TRAIN
    assert splits.window(log, splits.CUTOFF_EVAL)["x"].tolist() == [1, 2, 3]


def test_window_days() -> None:
    assert splits.window_days(splits.CUTOFF_TRAIN) == 6
    assert splits.window_days(splits.CUTOFF_EVAL) == 13
    assert splits.shift(20220430, 2) == 20220502


def test_user_halves_disjoint_complete_and_stable() -> None:
    log = fixtures.processed_tables(n=2_000)["kuairand", "log_random"]
    evaluation, simulator = splits.user_halves(log)
    assert not set(evaluation) & set(simulator)
    assert set(evaluation) | set(simulator) == set(log["user_id"])
    again, _ = splits.user_halves(log.sample(frac=1.0, random_state=1))
    np.testing.assert_array_equal(evaluation, again)
