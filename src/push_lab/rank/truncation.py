"""Truncation study: what the coarse stage's top-N costs (METHODS Section 4)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from push_lab.rank import cascade

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Truncation:
    """Per-user results at one N. Agreement and regret cover only the sampled users."""

    n: int
    survival: FloatArray
    agreement: FloatArray
    regret: FloatArray


def study(
    pool: IntArray,
    user_vectors: npt.NDArray[np.float32],
    item_ids: IntArray,
    item_vectors: npt.NDArray[np.float32],
    positives: list[set[int]],
    sample: IntArray,
    catalog_scores: FloatArray,
    sizes: list[int],
) -> list[Truncation]:
    """Evaluate the cascade at each N in ``sizes``.

    ``sample`` holds row positions of the users whose fine scores over the whole catalog are
    in ``catalog_scores`` (one row per sampled user, columns in ``item_ids`` order).
    """
    column = np.full(int(item_ids.max()) + 1, -1, dtype=np.int64)
    column[item_ids] = np.arange(len(item_ids))
    best = catalog_scores.max(axis=1)
    reference = item_ids[np.argmax(catalog_scores, axis=1)]
    results = []
    for n in sizes:
        kept = cascade.coarse(pool, user_vectors, item_ids, item_vectors, n)
        rows = kept[sample]
        cols = column[np.where(rows == cascade.PAD, item_ids[0], rows)]
        if (cols < 0).any():
            raise ValueError("candidates contain items outside the scored catalog")
        scores = np.take_along_axis(catalog_scores, cols, axis=1)
        scores[rows == cascade.PAD] = -np.inf
        choice = rows[np.arange(len(rows)), np.argmax(scores, axis=1)]
        results.append(
            Truncation(
                n=n,
                survival=cascade.survival(kept, positives),
                agreement=(choice == reference).astype(np.float64),
                regret=best - scores.max(axis=1),
            )
        )
    return results
