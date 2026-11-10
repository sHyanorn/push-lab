import math

import numpy as np
import pytest

from push_lab.rank import metrics


def test_recall_at_k() -> None:
    retrieved = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    positives = [{2, 4, 99}, {8}, set()]
    got = metrics.recall_at_k(retrieved, positives, 2)
    assert got[0] == pytest.approx(1 / 3)
    assert got[1] == 0.0
    assert math.isnan(got[2])


def test_user_auc_with_ties_and_single_class() -> None:
    users = np.array([1, 1, 1, 1, 2, 2, 3, 3])
    scores = np.array([0.9, 0.5, 0.5, 0.1, 0.3, 0.4, 0.2, 0.2])
    labels = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    got = metrics.user_auc(users, scores, labels, np.array([1, 2, 3]))
    # User 1: pairs (0.9>0.5)=1, (0.9>0.1)=1, (0.5 tie 0.5)=0.5, (0.5>0.1)=1 -> 3.5/4
    assert got[0] == pytest.approx(0.875)
    assert math.isnan(got[1])
    assert got[2] == pytest.approx(0.5)


def test_user_ndcg() -> None:
    users = np.array([1, 1, 1, 2, 2])
    scores = np.array([0.3, 0.2, 0.1, 0.5, 0.4])
    labels = np.array([0, 1, 1, 0, 0], dtype=float)
    got = metrics.user_ndcg(users, scores, labels, np.array([1, 2]), k=5)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 + 1 / math.log2(3)
    assert got[0] == pytest.approx(dcg / ideal)
    assert math.isnan(got[1])


def test_estimate_and_paired_comparison() -> None:
    a = np.array([0.6, 0.7, np.nan, 0.8])
    b = np.array([0.5, 0.6, 0.4, 0.7])
    idx = metrics.resample_indices(4, samples=200)
    est = metrics.estimate(a, idx)
    assert est.value == pytest.approx(0.7) and est.users == 3
    cmp = metrics.compare(a, b, idx)
    assert cmp.difference.value == pytest.approx(0.1)
    assert cmp.p_value == 0.0


def test_impression_weighted_estimate() -> None:
    values = np.array([1.0, 0.0, np.nan])
    weights = np.array([3.0, 1.0, 5.0])
    idx = metrics.resample_indices(3, samples=50)
    assert metrics.estimate(values, idx, weights).value == pytest.approx(0.75)


def test_interval_brackets_the_estimate_and_pairing_narrows_it() -> None:
    rng = np.random.default_rng(1)
    b = rng.random(400)
    a = b + 0.01 + rng.normal(0.0, 0.001, 400)
    idx = metrics.resample_indices(400, samples=200)
    est = metrics.estimate(a, idx)
    assert est.low <= est.value <= est.high
    diff = metrics.compare(a, b, idx).difference
    assert diff.low <= diff.value <= diff.high
    assert diff.value == pytest.approx(0.01, abs=0.001)
    # Paired differences vary by about 0.001 while the levels vary by about 0.3
    assert diff.high - diff.low < 0.002
