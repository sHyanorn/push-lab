import numpy as np
import pandas as pd
import pytest

from push_lab.data import fixtures
from push_lab.data.schemas import OBD_MEN_ACTIONS
from push_lab.ope import estimators as est
from push_lab.ope import obd, obd_blocks

UNIFORM = np.full((OBD_MEN_ACTIONS, 3), 1 / OBD_MEN_ACTIONS)


def _log(seed: int, n: int = 6000, rate: float = 0.2) -> pd.DataFrame:
    """A uniform random log spread over the week, with clicks at ``rate``."""
    rng = np.random.default_rng(seed)
    log = fixtures.obd_log(rng, n, policy="random")
    start = pd.Timestamp("2019-11-24", tz="UTC")
    log["timestamp"] = (start + pd.to_timedelta(rng.uniform(0, 7 * 86400, n), unit="s")).astype(str)
    log["click"] = (rng.random(n) < rate).astype(int)
    return log


def _logged(log: pd.DataFrame) -> est.Logged:
    clicks = log["click"].to_numpy(dtype=np.float64)
    ones, zeros = np.ones(len(clicks)), np.zeros(len(clicks))
    return est.Logged(clicks, ones, zeros, zeros, obd_blocks.blocks(log))


def test_blocks_count_ten_minutes_from_the_start_of_the_week() -> None:
    stamps = ["2019-11-24 00:00:00+00:00", "2019-11-24 00:09:59.999999+00:00"]
    stamps += ["2019-11-24 00:10:00+00:00", "2019-11-30 23:59:59+00:00"]
    np.testing.assert_array_equal(
        obd_blocks.blocks(pd.DataFrame({"timestamp": stamps})), [0, 0, 1, 1007]
    )
    with pytest.raises(ValueError, match="outside"):
        obd_blocks.blocks(pd.DataFrame({"timestamp": ["2019-12-01 00:00:00+00:00"]}))


def test_multiplicities_follow_the_registered_stream() -> None:
    draws = obd_blocks.multiplicities(3)
    rng = np.random.default_rng([20261110, 0, 2])
    np.testing.assert_array_equal(draws[2], rng.multinomial(1008, np.full(1008, 1 / 1008)))
    assert draws.shape == (3, 1008) and (draws.sum(axis=1) == 1008).all()


def test_both_logs_are_resampled_with_the_same_blocks() -> None:
    # The same rows as both logs: with joint blocks the estimate equals the on-policy value in
    # every resample, which resampling the two logs separately would not give
    logged = _logged(_log(0))
    out = obd_blocks.check(logged, logged, obd_blocks.multiplicities(50))
    ips = out["validation"]["IPS"]
    assert ips["ratio_to_on_policy"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert ips["relative_error_median"] == 0 and ips["validated"]
    assert out["on_policy"]["ci_low"] < out["on_policy"]["value"] < out["on_policy"]["ci_high"]


def test_check_applies_the_registered_rule_to_block_intervals() -> None:
    low, high = _logged(_log(1, rate=0.1)), _logged(_log(2, rate=0.2))
    out = obd_blocks.check(low, high, obd_blocks.multiplicities(200))
    ips = out["validation"]["IPS"]
    assert out["estimates"]["IPS"]["ci_high"] < out["on_policy"]["value"]
    assert not ips["interval_contains_on_policy"] and not ips["validated"]
    assert ips["ratio_to_on_policy"]["value"] == pytest.approx(
        low.reward.mean() / high.reward.mean()
    )
    assert out["rounds"] == 6000


def test_a_block_without_rows_of_a_log_keeps_its_place() -> None:
    log = _log(3)
    late = _logged(log[obd_blocks.blocks(log) >= 500])
    draws = np.zeros((1, 1008))
    draws[0, 600] = 1008
    out = obd_blocks.check(late, late, draws)
    in_block = log[obd_blocks.blocks(log) == 600]["click"].mean()
    assert out["on_policy"]["ci_low"] == pytest.approx(in_block)


def test_empirical_target_is_recomputed_in_every_resample() -> None:
    random_log, bts_log = _log(4), _log(5)
    draws = obd_blocks.multiplicities(100)
    out = obd_blocks.empirical_target(random_log, bts_log, UNIFORM, draws)
    ratio, change = out["ratio_to_on_policy"], out["difference_from_fixed_prior"]
    clicks = random_log["click"].to_numpy()
    shown = np.zeros_like(UNIFORM)
    np.add.at(shown, (bts_log["item_id"].to_numpy(), bts_log["position"].to_numpy() - 1), 1)
    empirical = np.mean(obd._weights(random_log, shown / shown.sum(axis=0)) * clicks)
    assert ratio["value"] == pytest.approx(empirical / bts_log["click"].mean())
    assert change["value"] == pytest.approx(
        ratio["value"] - clicks.mean() / bts_log["click"].mean()
    )
    assert ratio["ci_low"] < 1 < ratio["ci_high"]
    # The first 48 blocks, 21 times each: target, estimate and on-policy value come from them
    some = np.zeros((1, 1008))
    some[0, :48] = 21
    r, b = (log[obd_blocks.blocks(log) < 48] for log in (random_log, bts_log))
    shown = np.zeros_like(UNIFORM)
    np.add.at(shown, (b["item_id"].to_numpy(), b["position"].to_numpy() - 1), 1)
    expected = np.mean(obd._weights(r, shown / shown.sum(axis=0)) * r["click"].to_numpy())
    got = obd_blocks.empirical_target(random_log, bts_log, UNIFORM, some)["ratio_to_on_policy"]
    assert got["ci_low"] == pytest.approx(expected / b["click"].mean())
