import numpy as np
import pytest

from push_lab.policy import budget


def _concave(rng: np.random.Generator, units: int, sends: int) -> np.ndarray:
    return -np.sort(-rng.uniform(-0.2, 1.0, (units, sends)), axis=1)


@pytest.mark.parametrize("seed", range(20))
def test_allocation_matches_brute_force_with_non_increasing_gains(seed: int) -> None:
    rng = np.random.default_rng(seed)
    gains = _concave(rng, units=4, sends=3)
    # Ties between units, as with equal scores
    if seed % 4 == 0:
        gains[1] = gains[0]
    for b in range(0, 13):
        x, lam = budget.allocate(gains, b)
        assert x.sum() <= b and lam >= 0
        assert budget.total_gain(gains, x) == pytest.approx(budget.brute_force(gains, b))


def test_ties_at_the_multiplier_fill_the_budget_by_index() -> None:
    gains = np.array([[0.5], [0.5], [0.5], [0.9]])
    x, lam = budget.allocate(gains, 3)
    assert lam == 0.5
    np.testing.assert_array_equal(x, [1, 1, 0, 1])


def test_negative_gains_are_never_sent() -> None:
    gains = np.array([[0.3, -0.1], [-0.2, -0.3]])
    x, lam = budget.allocate(gains, 4)
    assert lam == 0.0
    np.testing.assert_array_equal(x, [1, 0])


def test_rising_gains_can_make_the_multiplier_fall_short() -> None:
    # Unit 0 gains little from one send and much from two (fatigue can break concavity):
    # with a budget of 2 the optimum sends both to unit 0 (0.1 + 1.0) but the multiplier
    # allocation cannot split the step and sends to unit 1 only
    gains = np.array([[0.1, 1.0], [0.6, 0.0]])
    x, _ = budget.allocate(gains, 2)
    assert budget.total_gain(gains, x) < budget.brute_force(gains, 2)


def test_top_k_takes_the_highest_eligible_scores_with_ties_by_index() -> None:
    scores = np.array([0.2, 0.9, 0.5, 0.5, 0.5, -1.0])
    eligible = np.array([True, False, True, True, True, True])
    np.testing.assert_array_equal(
        budget.top_k(scores, eligible, 2), [False, False, True, True, False, False]
    )
    np.testing.assert_array_equal(
        budget.top_k(scores, eligible, 4), [True, False, True, True, True, False]
    )
    assert budget.top_k(scores, eligible, 10).sum() == eligible.sum()


def test_top_k_equals_the_allocator_with_one_positive_gain_per_unit() -> None:
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.01, 1.0, 50)
    x, _ = budget.allocate(scores[:, None], 20)
    chosen = budget.top_k(scores, np.ones(50, dtype=bool), 20)
    np.testing.assert_array_equal(x.astype(bool), chosen)
