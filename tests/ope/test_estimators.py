from typing import Any

import numpy as np
import pytest

from push_lab.ope import estimators as est

ACTIONS = 3


def _world(n: int, seed: int, exact_q: bool) -> tuple[est.Logged, float]:
    """A contextual bandit with a known target value; logging is a softmax, the target picks
    the action with the largest true reward."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    p = 1 / (1 + np.exp(-(np.stack([x, -x, 0.5 * x + 0.3], axis=1) - 1.0)))
    logits = np.stack([0.5 * x, -0.5 * x, np.zeros(n)], axis=1)
    b = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    a = (rng.random(n)[:, None] > np.cumsum(b, axis=1)).sum(axis=1)
    r = (rng.random(n) < p[np.arange(n), a]).astype(float)
    target = p.argmax(axis=1)
    weight = (a == target) / b[np.arange(n), a]
    q = p if exact_q else np.clip(p + 0.1, 0, 1)
    logged = est.Logged(r, weight, q[np.arange(n), a], q[np.arange(n), target], np.arange(n))
    # Expected value over the context distribution, by a large separate sample
    xs = np.random.default_rng(10_000 + seed).standard_normal(200_000)
    ps = 1 / (1 + np.exp(-(np.stack([xs, -xs, 0.5 * xs + 0.3], axis=1) - 1.0)))
    return logged, float(ps.max(axis=1).mean())


def test_ips_and_dr_are_unbiased_and_cover() -> None:
    hits = {name: 0 for name in est.ESTIMATORS}
    errors: dict[str, list[float]] = {name: [] for name in est.ESTIMATORS}
    seeds = 100
    for seed in range(seeds):
        logged, truth = _world(2_000, seed, exact_q=False)
        out: Any = est.evaluate(logged, samples=300, seed=seed)
        for name, e in out["estimates"].items():
            hits[name] += e["ci_low"] <= truth <= e["ci_high"]
            errors[name].append(e["value"] - truth)
    for name in ("IPS", "DR", "SNIPS"):
        bias = np.mean(errors[name])
        se = np.std(errors[name]) / np.sqrt(seeds)
        assert abs(bias) < 3 * se + 1e-3, name
        assert hits[name] / seeds >= 0.85, name
    # The misspecified model biases DM upward by about 0.1
    assert np.mean(errors["DM"]) > 0.05


def test_dm_is_exact_with_the_true_model() -> None:
    logged, _ = _world(5_000, 1, exact_q=True)
    values = est.point(logged)
    expected = logged.q_target.mean()
    assert values["DM"] == pytest.approx(expected)


def test_on_policy_weights_give_the_on_policy_mean() -> None:
    rng = np.random.default_rng(0)
    r = (rng.random(500) < 0.3).astype(float)
    ones = np.ones(500)
    logged = est.Logged(r, ones, np.full(500, 0.3), np.full(500, 0.3), np.arange(500))
    values = est.point(logged)
    assert values["IPS"] == pytest.approx(r.mean())
    assert values["SNIPS"] == pytest.approx(r.mean())
    assert est.effective_size(ones) == pytest.approx(500)


def test_snips_ignores_a_common_weight_scale() -> None:
    logged, _ = _world(1_000, 2, exact_q=False)
    scaled = est.Logged(
        logged.reward, 7 * logged.weight, logged.q_logged, logged.q_target, logged.cluster
    )
    assert est.point(scaled)["SNIPS"] == pytest.approx(est.point(logged)["SNIPS"])
    assert est.effective_size(scaled.weight) == pytest.approx(est.effective_size(logged.weight))


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    # Two clusters with constant rewards 0 and 1: every resample mean is 0, 0.5 or 1
    r = np.array([0.0, 0.0, 1.0, 1.0])
    logged = est.Logged(r, np.ones(4), r, r, np.array([5, 5, 9, 9]))
    boot = est.bootstrap(logged, samples=200, seed=0)
    assert set(np.round(boot[:, est.ESTIMATORS.index("IPS")], 6)) <= {0.0, 0.5, 1.0}
    np.testing.assert_array_equal(boot, est.bootstrap(logged, samples=200, seed=0))


def test_report_fields() -> None:
    logged, _ = _world(500, 3, exact_q=False)
    out: Any = est.evaluate(logged, samples=50)
    assert set(out["estimates"]) == set(est.ESTIMATORS)
    assert out["rounds"] == 500
    assert out["max_weight"] == pytest.approx(logged.weight.max())


def test_disjoint_intervals_are_inconclusive() -> None:
    logged, _ = _world(2_000, 4, exact_q=False)
    out: Any = est.evaluate(logged, samples=200)
    # The misspecified model pushes DM well above the weighted estimators
    assert out["inconclusive"] is True
    exact, _ = _world(2_000, 4, exact_q=True)
    assert est.evaluate(exact, samples=200)["inconclusive"] is False
