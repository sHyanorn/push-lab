import numpy as np
import pandas as pd
import pytest

from push_lab.data import fixtures
from push_lab.data.schemas import OBD_MEN_ACTIONS
from push_lab.ope import obd


def test_prior_matches_the_pinned_file() -> None:
    assert len(obd.ALPHA) == len(obd.BETA) == OBD_MEN_ACTIONS
    assert (sum(obd.ALPHA), sum(obd.BETA)) == (1557, 408523)
    assert obd.ALPHA[13] == 518 and obd.BETA[13] == 111326


def test_bts_distribution() -> None:
    dist = obd.bts_distribution(draws=20_000)
    np.testing.assert_allclose(dist.sum(axis=0), 1)
    assert (dist.sum(axis=1) <= 1 + 1e-12).all()
    finer = obd.bts_distribution(draws=200_000, seed=1)
    assert np.abs(dist - finer).max() < 0.02
    # A prior that almost surely ranks item 0 first and item 1 second
    sure = obd.bts_distribution((1000, 500, 1, 1), (1, 500, 1000, 1000), draws=2_000)
    assert sure[0, 0] == 1 and sure[1, 1] > 0.99
    np.testing.assert_allclose(sure[2:, 2].sum(), 1)


def _logs(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Synthetic random and BTS logs with clicks from a known rate per item and position."""
    rng = np.random.default_rng(seed)
    items = fixtures.obd_items(rng)
    dist = obd.bts_distribution(draws=20_000)
    rate = rng.uniform(0.01, 0.2, OBD_MEN_ACTIONS)[:, None] * np.array([1.0, 0.8, 0.6])[None, :]

    def log(policy: str) -> pd.DataFrame:
        frame = fixtures.obd_log(rng, n, policy="random")
        position = rng.integers(1, 4, n)
        if policy == "random":
            item = rng.integers(0, OBD_MEN_ACTIONS, n)
            propensity = np.full(n, 1 / OBD_MEN_ACTIONS)
        else:
            cum = np.cumsum(dist[:, position - 1].T, axis=1)
            item = (rng.random(n)[:, None] > cum).sum(axis=1)
            propensity = dist[item, position - 1]
        frame["item_id"], frame["position"] = item, position
        frame["propensity_score"] = propensity
        frame["click"] = (rng.random(n) < rate[item, position - 1]).astype(int)
        return frame

    return log("random"), log("bts"), items, dist


def test_features_use_the_given_item() -> None:
    random_log, _, items, _ = _logs(200, 0)
    item = np.full(200, 5)
    x = obd.features(random_log, items, item)
    np.testing.assert_array_equal(x["own_affinity"], random_log["user_item_affinity_5"])
    assert (x["item"] == 5).all()
    np.testing.assert_array_equal(x["position"], random_log["position"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_validation_recovers_the_bts_value() -> None:
    random_log, bts_log, items, dist = _logs(20_000, 1)
    out = obd.validate(random_log, bts_log, items, dist, samples=200)
    ips = out["estimates"]["IPS"]
    assert ips["ci_low"] <= out["on_policy"]["value"] <= ips["ci_high"]
    assert out["validation"]["IPS"]["validated"]
    assert out["validation"]["DR"]["relative_error_median"] < 0.2
    assert out["random_rows"] == 20_000 and out["max_weight"] > 1


def test_impressions_group_rows_shown_together() -> None:
    log = pd.DataFrame(
        {
            "timestamp": [
                "2019-11-24 00:00:00.000000+00:00",
                "2019-11-24 00:00:00.200000+00:00",
                "2019-11-24 00:00:00.400000+00:00",
                "2019-11-24 00:00:00.500000+00:00",
                "2019-11-24 00:00:05.000000+00:00",
            ],
            "user_feature_0": ["a", "a", "a", "b", "a"],
        }
    )
    np.testing.assert_array_equal(obd.impressions(log), [0, 0, 0, 1, 2])


def test_impressions_split_a_user_at_one_second() -> None:
    log = pd.DataFrame(
        {
            "timestamp": [
                "2019-11-24 00:00:00.000000+00:00",
                "2019-11-24 00:00:00.999999+00:00",
                "2019-11-24 00:00:05.999999+00:00",
            ],
            "user_feature_0": ["a", "a", "a"],
        }
    )
    np.testing.assert_array_equal(obd.impressions(log), [0, 0, 1])


def test_empirical_target_recovers_the_bts_value() -> None:
    random_log, bts_log, _, dist = _logs(20_000, 4)
    out = obd.empirical_target(random_log, bts_log, dist, samples=200)
    ratio, change = out["ratio_to_on_policy"], out["difference_from_fixed_prior"]
    assert ratio["ci_low"] <= 1 <= ratio["ci_high"]
    # The BTS logs are drawn from ``dist``, so the two targets nearly coincide and their paired
    # difference is far tighter than the ratio itself
    clicks = random_log["click"].to_numpy()
    fixed = np.mean(obd._weights(random_log, dist) * clicks) / bts_log["click"].mean()
    assert change["value"] == pytest.approx(ratio["value"] - fixed)
    assert change["ci_high"] - change["ci_low"] < (ratio["ci_high"] - ratio["ci_low"]) / 2


def test_weights_use_the_position_of_each_row() -> None:
    # The target shows item 0, 1 and 2 at positions 1, 2 and 3; only item 2 at position 3 gets
    # clicks, at rate 0.5, so the target's click rate per row is 0.5 / 3
    rng = np.random.default_rng(3)
    n = 60_000
    log = pd.DataFrame(
        {
            "item_id": rng.integers(0, OBD_MEN_ACTIONS, n),
            "position": rng.integers(1, 4, n),
            "propensity_score": np.full(n, 1 / OBD_MEN_ACTIONS),
        }
    )
    dist = np.zeros((OBD_MEN_ACTIONS, 3))
    dist[0, 0] = dist[1, 1] = dist[2, 2] = 1.0
    rate = np.zeros((OBD_MEN_ACTIONS, 3))
    rate[2, 2] = 0.5
    click = rng.random(n) < rate[log["item_id"], log["position"] - 1]
    assert float(np.mean(obd._weights(log, dist) * click)) == pytest.approx(0.5 / 3, abs=0.03)
