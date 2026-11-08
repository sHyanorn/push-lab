import math

import numpy as np
import torch

from push_lab.data import fixtures
from push_lab.recall import two_tower
from push_lab.recall.index import ItemIndex, overlap


def _hand_loss(
    u: list[list[float]], v: list[list[float]], ids: list[int], log_q: list[float], tau: float
) -> float:
    total = 0.0
    for b in range(len(u)):
        logits = []
        for j in range(len(v)):
            if j != b and ids[j] == ids[b]:
                continue
            dot = sum(x * y for x, y in zip(u[b], v[j], strict=True))
            logits.append((j, dot / tau - log_q[j]))
        own = dict(logits)[b]
        total += -(own - math.log(sum(math.exp(s) for _, s in logits)))
    return total / len(u)


U = [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]
V = [[0.8, 0.6], [0.0, 1.0], [1.0, 0.0]]


def test_loss_matches_hand_computation() -> None:
    log_q = [math.log(0.5), math.log(0.3), math.log(0.2)]
    got = two_tower.logq_softmax_loss(
        torch.tensor(U), torch.tensor(V), torch.tensor([7, 8, 9]), torch.tensor(log_q), 0.5
    )
    assert math.isclose(float(got), _hand_loss(U, V, [7, 8, 9], log_q, 0.5), rel_tol=1e-6)


def test_equal_q_is_plain_in_batch_softmax() -> None:
    log_q = torch.full((3,), math.log(1 / 3))
    got = two_tower.logq_softmax_loss(
        torch.tensor(U), torch.tensor(V), torch.tensor([1, 2, 3]), log_q, 0.5
    )
    plain = torch.nn.functional.cross_entropy(
        torch.tensor(U) @ torch.tensor(V).T / 0.5, torch.arange(3)
    )
    assert math.isclose(float(got), float(plain), rel_tol=1e-6)


def test_duplicate_item_is_masked() -> None:
    log_q = [0.0, 0.0, 0.0]
    got = two_tower.logq_softmax_loss(
        torch.tensor(U), torch.tensor(V), torch.tensor([5, 5, 6]), torch.tensor(log_q), 1.0
    )
    assert math.isclose(float(got), _hand_loss(U, V, [5, 5, 6], log_q, 1.0), rel_tol=1e-6)
    unmasked = _hand_loss(U, V, [5, 6, 7], log_q, 1.0)
    assert not math.isclose(float(got), unmasked, rel_tol=1e-3)


def _model(seed: int) -> two_tower.TwoTower:
    tables = fixtures.processed_tables(n=3_000)
    items = two_tower.item_side_features(tables["kuairand", "videos_basic"])
    window = tables["kuairand", "log_standard_early"]
    config = two_tower.Config(epochs=3, batch=128, seed=seed, threads=1)
    return two_tower.TwoTower(tables["kuairand", "users"], items, window, config)


def test_training_runs_and_is_deterministic() -> None:
    a, b = _model(0), _model(0)
    losses = a.fit()
    assert b.fit() == losses
    ids = np.arange(fixtures.N_VIDEOS, dtype=np.int64)
    np.testing.assert_array_equal(a.item_vectors(ids), b.item_vectors(ids))
    unknown = a.user_vectors(np.array([10**9], dtype=np.int64))
    assert np.isclose(np.linalg.norm(unknown), 1.0)


def test_hnsw_matches_exact_search() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(5_000, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = vectors[rng.choice(5_000, 200, replace=False)]
    ids = np.arange(5_000, dtype=np.int64) + 100
    hnsw = ItemIndex(ids, vectors).search(queries, 200)
    exact = ItemIndex(ids, vectors, exact=True).search(queries, 200)
    assert overlap(hnsw, exact) >= 0.99
