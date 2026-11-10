import numpy as np
import pytest

from push_lab.policy import compare


def test_bootstrap_means_equal_explicit_resamples() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 50))
    boot = compare.Bootstrap(50, samples=2_500, seed=4)
    means = boot.means(values)
    assert means.shape == (2_500, 3)
    np.testing.assert_allclose(boot.weights.sum(axis=1), 50)
    np.testing.assert_allclose(means[7], boot.weights[7] @ values.T / 50)
    np.testing.assert_allclose(means.mean(axis=0), values.mean(axis=1), atol=0.02)
    se = values.std(axis=1) / np.sqrt(50)
    np.testing.assert_allclose(means.std(axis=0), se, rtol=0.1)


def test_verdict() -> None:
    assert compare.verdict(0.3, 0.01) == "win"
    assert compare.verdict(-0.3, 0.01) == "loss"
    assert compare.verdict(0.3, 0.013) == "neither"


@pytest.mark.parametrize(
    ("average", "rejected", "wins", "losses", "expected"),
    [
        (0.1, True, 50, 0, "beats in most sampled settings"),
        (0.1, True, 50, 5, "beats in most sampled settings"),
        (0.1, True, 30, 5, "depends on the setting"),
        (0.1, True, 30, 3, "better on average"),
        (-0.1, True, 0, 48, "loses in most sampled settings"),
        (-0.1, True, 2, 20, "worse on average"),
        (0.1, False, 25, 25, "depends on the setting"),
        (0.1, False, 50, 0, "no detectable difference"),
    ],
)
def test_labels_follow_the_registered_order(
    average: float, rejected: bool, wins: int, losses: int, expected: str
) -> None:
    assert compare.label(average, rejected, wins, losses) == expected


def test_band_classes() -> None:
    assert compare.band_class(-0.01, 0.0005) == "below"
    assert compare.band_class(0.001, 0.004) == "inside"
    assert compare.band_class(0.006, 0.01) == "above"
    assert compare.band_class(0.0005, 0.002) == "straddling"


def test_sweep_average_covers_a_known_mean() -> None:
    """Settings whose true differences have mean 0.2; the two-level interval covers it."""
    rng = np.random.default_rng(2)
    covered = 0
    for trial in range(40):
        truth = 0.2 + rng.normal(scale=0.3, size=64)
        users = truth[:, None] + rng.normal(scale=1.0, size=(64, 400))
        boot = compare.Bootstrap(400, samples=1_000, seed=trial)
        boots = boot.means(users).T
        est, _, _ = compare.sweep_average(boots, users.mean(axis=1), seed=trial)
        covered += est["ci_low"] <= 0.2 <= est["ci_high"]
    assert covered >= 34
