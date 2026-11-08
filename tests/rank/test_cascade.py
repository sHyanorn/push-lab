import numpy as np

from push_lab.rank import cascade


def _vectors(n_items: int, n_users: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    item_ids = np.arange(n_items, dtype=np.int64) * 3 + 10
    items = rng.normal(size=(n_items, 8)).astype(np.float32)
    users = rng.normal(size=(n_users, 8)).astype(np.float32)
    return item_ids, items, users


def test_pool_appends_unseen_trending_items() -> None:
    personal = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    pool = cascade.build_pool(personal, np.array([3, 7, 4], dtype=np.int64))
    assert pool.tolist() == [[1, 2, 3, 7, 4], [4, 5, 6, 3, 7]]


def test_coarse_keeps_highest_scores_and_skips_padding() -> None:
    item_ids, items, users = _vectors(6, 2)
    pool = np.array([item_ids, [*item_ids[:3], -1, -1, -1]], dtype=np.int64)
    kept = cascade.coarse(pool, users, item_ids, items, 4)
    expected = item_ids[np.argsort(-(items @ users[0]))[:4]]
    assert kept[0].tolist() == expected.tolist()
    assert kept[1, 3] == cascade.PAD and cascade.PAD not in kept[1, :3]


def test_full_cascade_equals_catalog_reference() -> None:
    item_ids, items, users = _vectors(50, 5)
    rng = np.random.default_rng(1)
    fine = rng.random((5, 50))
    user_ids = np.arange(5, dtype=np.int64)
    row_of = {int(i): k for k, i in enumerate(item_ids)}

    def scorer(u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return np.array([fine[a, row_of[int(b)]] for a, b in zip(u, i, strict=True)])

    pool = np.tile(item_ids, (5, 1))
    kept = cascade.coarse(pool, users, item_ids, items, len(item_ids))
    got = cascade.top1(user_ids, kept, scorer)
    assert got.tolist() == item_ids[np.argmax(fine, axis=1)].tolist()


def test_survival_is_monotone_in_n() -> None:
    item_ids, items, users = _vectors(40, 6)
    pool = np.tile(item_ids, (6, 1))
    positives = [set(item_ids[[k, k + 7, 30]].tolist()) for k in range(6)]
    previous = np.zeros(6)
    for n in (5, 10, 20, 40):
        s = cascade.survival(cascade.coarse(pool, users, item_ids, items, n), positives)
        assert (s >= previous).all()
        previous = s
    assert (previous == 1.0).all()
