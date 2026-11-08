import numpy as np
import numpy.typing as npt
import pytest
from sklift import metrics as sk

from push_lab.uplift import curves

# scikit-uplift 0.5.1 calls a scikit-learn helper deprecated in 1.8 (D005)
pytestmark = pytest.mark.filterwarnings("ignore:Function stable_cumsum:FutureWarning")

Data = tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]


def _data(n: int, seed: int, distinct: int | None = None) -> Data:
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < 0.6).astype(np.int64)
    score = rng.normal(size=n)
    if distinct is not None:
        score = np.round(score * distinct / 6)
    y = (rng.random(n) < 0.2 + 0.15 * t * (score > 0)).astype(np.int64)
    return score, t, y


def test_hand_computed_curves() -> None:
    score = np.array([0.9, 0.7, 0.7, 0.2])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 1, 1])
    c = curves.counts(curves.exact_groups(score), t, y)
    x, q = curves.qini_curve(c)
    # Groups: {0.9}, {0.7, 0.7}, {0.2}
    np.testing.assert_allclose(x, [0, 1, 3, 4])
    # k=1: N_C = 0 -> Q = 1; k=3: 2 - 0 * 2/1 = 2; k=4: 2 - 1 * 2/2 = 1
    np.testing.assert_allclose(q, [0, 1, 2, 1])
    _, u = curves.uplift_curve(c)
    # k=1: (1/1 - 0) * 1; k=3: (2/2 - 0/1) * 3; k=4: (2/2 - 1/2) * 4
    np.testing.assert_allclose(u, [0, 1, 3, 2])
    assert curves.area(x, q) == pytest.approx(0.5 + 3 + 1.5)


@pytest.mark.parametrize("distinct", [None, 5])
def test_coefficients_match_scikit_uplift(distinct: int | None) -> None:
    score, t, y = _data(2_000, seed=1, distinct=distinct)
    c = curves.counts(curves.exact_groups(score), t, y)
    assert curves.qini_coefficient(c) == pytest.approx(
        sk.qini_auc_score(y, score, t, negative_effect=True), abs=1e-12
    )
    assert curves.auuc(c) == pytest.approx(sk.uplift_auc_score(y, score, t), abs=1e-12)
    x, q = curves.qini_curve(c)
    sx, sq = sk.qini_curve(y, score, t)
    np.testing.assert_allclose(x, sx)
    np.testing.assert_allclose(q, sq)


def test_perfect_uplift_order_follows_class_sizes() -> None:
    # More control responders than treated non-responders flips the tail of the oracle
    t = np.array([1, 1, 0, 0, 0, 0])
    y = np.array([1, 0, 1, 1, 1, 0])
    c = curves.counts(curves.exact_groups(np.arange(6.0)), t, y)
    assert curves.auuc(c) == pytest.approx(sk.uplift_auc_score(y, np.arange(6.0), t), abs=1e-12)


def test_binned_equals_exact_when_bins_hold_one_score() -> None:
    _, t, y = _data(5_000, seed=2)
    # Five scores, each far above 1/200 of the rows, so every bin is one score
    score = np.random.default_rng(2).integers(0, 5, len(t)).astype(np.float64)
    exact = curves.counts(curves.exact_groups(score), t, y)
    binned = curves.counts(curves.quantile_groups(score, bins=200), t, y)
    assert curves.qini_coefficient(binned) == pytest.approx(curves.qini_coefficient(exact))
    assert curves.auuc(binned) == pytest.approx(curves.auuc(exact))


def test_quantile_groups_keep_ties_together_and_order_by_score() -> None:
    score = np.r_[np.full(500, 1.0), np.linspace(0, 0.5, 500)]
    groups = curves.quantile_groups(score, bins=10)
    assert len(np.unique(groups[:500])) == 1
    assert groups[0] == 0
    assert np.all(np.diff(groups[500:]) <= 0)


def test_random_scores_give_coefficients_near_zero() -> None:
    _, t, y = _data(50_000, seed=3)
    noise = np.random.default_rng(4).random(len(t))
    c = curves.counts(curves.exact_groups(noise), t, y)
    assert abs(curves.qini_coefficient(c)) < 0.05
    assert abs(curves.auuc(c)) < 0.05


def test_uplift_at_budget_interpolates_the_curve() -> None:
    score = np.array([4.0, 3.0, 2.0, 1.0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 0, 0])
    c = curves.counts(curves.exact_groups(score), t, y)
    # U at k = 1, 2, 3, 4: 1, 2, 1.5, 2
    got = curves.uplift_at(c, np.array([0.25, 0.5, 0.625, 1.0]))
    np.testing.assert_allclose(got, [1.0, 2.0, 1.75, 2.0])


def test_cell_bootstrap_matches_row_bootstrap() -> None:
    score, t, y = _data(600, seed=5)
    groups = curves.exact_groups(score)
    samples = 2_000

    rng = np.random.default_rng(6)
    by_row = []
    for _ in range(samples):
        idx = rng.integers(0, len(score), len(score))
        by_row.append(curves.qini_coefficient(curves.counts(groups[idx], t[idx], y[idx])))

    cell = curves.cells([curves.quantile_groups(score, bins=10)], t, y)
    binned_rows = curves.quantile_groups(score, bins=10)
    by_row_binned = []
    rng = np.random.default_rng(7)
    for _ in range(samples):
        idx = rng.integers(0, len(score), len(score))
        by_row_binned.append(
            curves.qini_coefficient(curves.counts(binned_rows[idx], t[idx], y[idx]))
        )
    by_cell = [
        curves.qini_coefficient(curves.counts(cell.groups[0], cell.t, cell.y, w))
        for w in curves.resampled_weights(cell.count, samples, seed=8)
    ]
    # Same statistic, two resampling schemes: percentile intervals agree within Monte Carlo error
    np.testing.assert_allclose(
        np.percentile(by_cell, [2.5, 50, 97.5]),
        np.percentile(by_row_binned, [2.5, 50, 97.5]),
        atol=0.015,
    )
    # Exact rows as cells reproduce the row bootstrap too
    exact = curves.cells([groups], t, y)
    by_exact_cell = [
        curves.qini_coefficient(curves.counts(exact.groups[0], exact.t, exact.y, w))
        for w in curves.resampled_weights(exact.count, samples, seed=9)
    ]
    np.testing.assert_allclose(
        np.percentile(by_exact_cell, [2.5, 50, 97.5]),
        np.percentile(by_row, [2.5, 50, 97.5]),
        atol=0.015,
    )


def test_denominator_is_recomputed_per_resample() -> None:
    score, t, y = _data(400, seed=10)
    cell = curves.cells([curves.exact_groups(score)], t, y)
    w = next(curves.resampled_weights(cell.count, 1, seed=0))
    c = curves.counts(cell.groups[0], cell.t, cell.y, w)
    # Expanding the weights back into rows gives the same coefficient
    rows = np.repeat(np.arange(len(w)), w.astype(np.int64))
    expanded = curves.counts(cell.groups[0][rows], cell.t[rows], cell.y[rows])
    assert curves.qini_coefficient(c) == pytest.approx(curves.qini_coefficient(expanded))
    assert curves.qini_coefficient(c) == pytest.approx(
        sk.qini_auc_score(cell.y[rows], -cell.groups[0][rows].astype(float), cell.t[rows])
    )


def test_cells_with_two_groupings_preserve_counts() -> None:
    rng = np.random.default_rng(11)
    a, b = rng.integers(0, 777, 20_000), rng.integers(0, 913, 20_000)
    t, y = rng.integers(0, 2, 20_000), rng.integers(0, 2, 20_000)
    cell = curves.cells([a, b], t, y)
    assert cell.count.sum() == 20_000
    for rows, groups in ((a, cell.groups[0]), (b, cell.groups[1])):
        np.testing.assert_array_equal(
            curves.counts(groups, cell.t, cell.y, cell.count, 913).y_c,
            curves.counts(rows, t, y, None, 913).y_c,
        )
