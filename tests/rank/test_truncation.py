import itertools

import numpy as np

from push_lab.rank import truncation


def test_full_pool_agrees_with_catalog_and_metrics_move_with_n() -> None:
    rng = np.random.default_rng(0)
    n_items, n_users = 60, 8
    item_ids = np.arange(n_items, dtype=np.int64) + 1_000
    items = rng.normal(size=(n_items, 4)).astype(np.float32)
    users = rng.normal(size=(n_users, 4)).astype(np.float32)
    pool = np.tile(item_ids, (n_users, 1))
    positives = [set(rng.choice(item_ids, 3, replace=False).tolist()) for _ in range(n_users)]
    sample = np.array([0, 2, 5], dtype=np.int64)
    catalog = rng.random((len(sample), n_items))

    results = truncation.study(
        pool, users, item_ids, items, positives, sample, catalog, [5, 20, n_items]
    )
    full = results[-1]
    assert (full.agreement == 1.0).all()
    assert np.allclose(full.regret, 0.0)
    assert (full.survival == 1.0).all()
    for small, large in itertools.pairwise(results):
        assert (large.survival >= small.survival).all()
        assert (large.regret <= small.regret + 1e-12).all()
