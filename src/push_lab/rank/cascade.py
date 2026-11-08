"""Candidate pool, coarse ranking and the top-1 cascade (METHODS Section 4).

Candidate sets are padded 2-D arrays of item ids, one row per user, with ``PAD`` in
unused slots.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

PAD = -1
PERSONAL_K = 500
TRENDING_K = 100
# Users scored at once in the coarse stage, to bound memory
CHUNK_USERS = 1024

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
# Scores a flat batch of (user id, item id) pairs
PairScorer = Callable[[IntArray, IntArray], npt.NDArray[np.float64]]


def build_pool(personal: IntArray, trending: IntArray) -> IntArray:
    """Each user's personal items, then the trending items not already among them."""
    rows = []
    for items in personal:
        seen = set(items.tolist())
        extra = [i for i in trending.tolist() if i not in seen]
        rows.append(np.concatenate([items, np.array(extra, dtype=np.int64)]))
    width = max(len(r) for r in rows)
    pool = np.full((len(rows), width), PAD, dtype=np.int64)
    for k, row in enumerate(rows):
        pool[k, : len(row)] = row
    return pool


def coarse(
    pool: IntArray,
    user_vectors: FloatArray,
    item_ids: IntArray,
    item_vectors: FloatArray,
    n: int,
) -> IntArray:
    """Keep the top ``n`` items of each user's pool by two-tower score."""
    lookup = np.full(int(item_ids.max()) + 1, -1, dtype=np.int64)
    lookup[item_ids] = np.arange(len(item_ids))
    kept = np.full((len(pool), min(n, pool.shape[1])), PAD, dtype=np.int64)
    for start in range(0, len(pool), CHUNK_USERS):
        block = pool[start : start + CHUNK_USERS]
        pad = block == PAD
        rows = lookup[np.where(pad, item_ids[0], block)]
        scores = np.einsum(
            "ud,upd->up", user_vectors[start : start + CHUNK_USERS], item_vectors[rows]
        )
        scores[pad] = -np.inf
        # Stable sort so ties keep pool order
        order = np.argsort(-scores, axis=1, kind="stable")[:, :n]
        chosen = np.take_along_axis(block, order, axis=1)
        chosen[np.isneginf(np.take_along_axis(scores, order, axis=1))] = PAD
        kept[start : start + CHUNK_USERS] = chosen
    return kept


def top1(user_ids: IntArray, candidates: IntArray, scorer: PairScorer) -> IntArray:
    """The highest-scoring candidate per user; ties go to the earlier candidate."""
    valid = candidates != PAD
    users = np.repeat(user_ids, valid.sum(axis=1))
    scores = np.full(candidates.shape, -np.inf)
    scores[valid] = scorer(users, candidates[valid])
    best = np.argmax(scores, axis=1)
    return candidates[np.arange(len(candidates)), best]


def survival(candidates: IntArray, positives: list[set[int]]) -> npt.NDArray[np.float64]:
    """Per user, the share of positive items that are among the candidates (NaN if none)."""
    out = np.full(len(candidates), np.nan)
    for k, (row, pos) in enumerate(zip(candidates, positives, strict=True)):
        if pos:
            out[k] = len(pos & set(row.tolist())) / len(pos)
    return out
