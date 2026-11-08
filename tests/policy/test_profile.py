import numpy as np

from push_lab.policy import profile
from push_lab.sim import inputs


def _hourly() -> np.ndarray:
    return np.random.default_rng(0).poisson(2.0, (200, 24)).astype(np.float64)


def test_no_perturbation_reproduces_the_policy_profile() -> None:
    hourly = _hourly()
    expected = inputs.smoothed_profile(inputs.windows_from_hourly(hourly))
    np.testing.assert_array_equal(profile.perturbed(hourly, 0.0, 0.0, seed=1), expected)


def test_a_full_day_shift_is_the_identity_and_shifts_interpolate() -> None:
    hourly = _hourly()
    np.testing.assert_array_equal(profile.shift_hours(hourly, 24.0), hourly)
    np.testing.assert_array_equal(profile.shift_hours(hourly, 1.0)[:, 5], hourly[:, 4])
    np.testing.assert_allclose(
        profile.shift_hours(hourly, -1.5)[:, 5], 0.5 * hourly[:, 7] + 0.5 * hourly[:, 6]
    )
    np.testing.assert_allclose(profile.shift_hours(hourly, 2.3).sum(axis=1), hourly.sum(axis=1))


def test_a_one_hour_shift_moves_activity_into_the_next_window() -> None:
    hourly = np.zeros((1, 24))
    hourly[0, 10] = 5.0  # last hour of the 08:00 window
    shifted = inputs.windows_from_hourly(profile.shift_hours(hourly, 2.0))
    np.testing.assert_array_equal(shifted, [[0.0, 5.0, 0.0, 0.0]])


def test_noise_keeps_shares_and_is_seeded() -> None:
    hourly = _hourly()
    a = profile.perturbed(hourly, 0.0, 1.0, seed=3)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    np.testing.assert_array_equal(a, profile.perturbed(hourly, 0.0, 1.0, seed=3))
    clean = profile.perturbed(hourly, 0.0, 0.0, seed=3)
    agree = (a.argmax(axis=1) == clean.argmax(axis=1)).mean()
    assert agree < 0.8
