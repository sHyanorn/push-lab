import numpy as np
import pytest

from push_lab import stats


def test_p_value_counts_both_tails_and_leaves_nan_out() -> None:
    assert stats.p_value(np.array([1.0, 2.0, 3.0, -1.0])) == pytest.approx(0.5)
    assert stats.p_value(np.ones(10)) == 0.0
    assert stats.p_value(np.array([1.0, 2.0, 3.0, -1.0, np.nan, np.nan])) == pytest.approx(0.5)


def test_holm() -> None:
    assert stats.holm({"C1": 0.01, "C2": 0.04}) == {"C1": True, "C2": True}
    assert stats.holm({"C1": 0.03, "C2": 0.04}) == {"C1": False, "C2": False}
    assert stats.holm({"C1": 0.001, "C2": 0.03}) == {"C1": True, "C2": True}
    assert stats.holm({"C1": 0.001, "C2": 0.06}) == {"C1": True, "C2": False}
    assert stats.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert stats.holm({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


def test_interval_is_the_rounded_percentile_interval() -> None:
    out = stats.interval(0.12345678, np.arange(1_001) / 1_000)
    assert out == {"value": 0.123457, "ci_low": 0.025, "ci_high": 0.975}


def test_flips_are_resolved_only_when_both_intervals_exclude_zero() -> None:
    rng = np.random.default_rng(1)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    points = np.array([1.0, 0.5, -0.5, -0.02])
    boot = points + rng.normal(scale=0.1, size=(4_000, 4))
    crossings = stats.flips(x, points, boot)
    assert len(crossings) == 1
    c = crossings[0]
    assert c["resolved"] and c["at"] == pytest.approx(1.5)
    assert c["ci_low"] < 1.5 < c["ci_high"]
    weak = stats.flips(
        x, np.array([0.05, -0.05, -0.05, -0.05]), boot - points + [0.05, -0.05, -0.05, -0.05]
    )
    assert len(weak) == 1 and not weak[0]["resolved"]
    assert stats.flips(x, np.ones(4), boot - points + 1) == []
