from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from push_lab.uplift import curves
from push_lab.uplift import forest as forest_module
from push_lab.uplift.forest import CausalForest
from push_lab.uplift.learners import SLearner, TLearner, XLearner

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _sigmoid(z: FloatArray) -> FloatArray:
    out: FloatArray = 1 / (1 + np.exp(-z))
    return out


def _trial(
    n: int, seed: int, effect: bool = True
) -> tuple[FloatArray, IntArray, IntArray, FloatArray]:
    """Randomized trial with a known effect tau(x) on the outcome probability."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    t = (rng.random(n) < 0.5).astype(np.int64)
    base = -1 + x[:, 0]
    lift = (0.5 + 0.8 * x[:, 1]) if effect else np.zeros(n)
    tau = _sigmoid(base + lift) - _sigmoid(base)
    y = (rng.random(n) < _sigmoid(base + t * lift)).astype(np.int64)
    return x, t, y, tau


Fit = Callable[[FloatArray, IntArray, IntArray], SLearner | TLearner | XLearner | CausalForest]
LEARNERS: dict[str, Fit] = {
    "s": SLearner.fit,
    "t": TLearner.fit,
    "x": XLearner.fit,
    "forest": CausalForest.fit,
}


@pytest.fixture(scope="module")
def trial() -> tuple[FloatArray, IntArray, IntArray, FloatArray, FloatArray]:
    x, t, y, _ = _trial(20_000, seed=0)
    x_test, _, _, tau_test = _trial(5_000, seed=1)
    return x, t, y, x_test, tau_test


@pytest.mark.parametrize("name", list(LEARNERS))
def test_learner_recovers_known_effect(
    name: str, trial: tuple[FloatArray, IntArray, IntArray, FloatArray, FloatArray]
) -> None:
    x, t, y, x_test, tau = trial
    got = LEARNERS[name](x, t, y).effect(x_test)
    assert np.corrcoef(got, tau)[0, 1] > 0.7
    assert abs(got.mean() - tau.mean()) < 0.02


@pytest.mark.parametrize("name", ["t", "forest"])
def test_fit_is_deterministic(
    name: str, trial: tuple[FloatArray, IntArray, IntArray, FloatArray, FloatArray]
) -> None:
    x, t, y, x_test, _ = trial
    first = LEARNERS[name](x[:5_000], t[:5_000], y[:5_000]).effect(x_test)
    second = LEARNERS[name](x[:5_000], t[:5_000], y[:5_000]).effect(x_test)
    np.testing.assert_array_equal(first, second)


def test_t_learner_response_is_the_treated_model(
    trial: tuple[FloatArray, IntArray, IntArray, FloatArray, FloatArray],
) -> None:
    x, t, y, x_test, _ = trial
    model = TLearner.fit(x, t, y)
    response = model.response(x_test)
    assert np.all((response > 0) & (response < 1))
    np.testing.assert_allclose(model.effect(x_test), response - model.control.predict(x_test))


def test_no_effect_gives_qini_interval_containing_zero() -> None:
    x, t, y, _ = _trial(20_000, seed=2, effect=False)
    x_test, t_test, y_test, _ = _trial(20_000, seed=3, effect=False)
    score = TLearner.fit(x, t, y).effect(x_test)
    cell = curves.cells([curves.exact_groups(score)], t_test, y_test)
    boot = [
        curves.qini_coefficient(curves.counts(cell.groups[0], cell.t, cell.y, w))
        for w in curves.resampled_weights(cell.count, 300, seed=0)
    ]
    low, high = np.percentile(boot, [2.5, 97.5])
    assert low < 0 < high


def test_forest_chunks_match_one_pass(
    trial: tuple[FloatArray, IntArray, IntArray, FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forest_module, "CHUNK_ROWS", 512)
    x, t, y, x_test, _ = trial
    forest = CausalForest.fit(x[:5_000], t[:5_000], y[:5_000])
    whole = np.asarray(forest.model.effect(x_test), dtype=np.float64).ravel()
    np.testing.assert_array_equal(forest.effect(x_test), whole)
