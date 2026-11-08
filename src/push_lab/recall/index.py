"""Nearest-neighbour search over item vectors with faiss (METHODS Section 3)."""

import faiss
import numpy as np
import numpy.typing as npt

HNSW_M = 32
EF_CONSTRUCTION = 200
# Must be well above the largest K searched (500); see D018
EF_SEARCH = 1024


class ItemIndex:
    """Inner-product search returning item ids; ``exact`` uses brute force as the reference."""

    def __init__(
        self,
        item_ids: npt.NDArray[np.int64],
        vectors: npt.NDArray[np.float32],
        *,
        exact: bool = False,
    ) -> None:
        self.item_ids = item_ids
        dim = vectors.shape[1]
        self.index: faiss.Index
        if exact:
            self.index = faiss.IndexFlatIP(dim)
        else:
            hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = EF_CONSTRUCTION
            hnsw.hnsw.efSearch = EF_SEARCH
            self.index = hnsw
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: npt.NDArray[np.float32], k: int) -> npt.NDArray[np.int64]:
        """Top-``k`` item ids for each query row, best first."""
        _, rows = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        ids: npt.NDArray[np.int64] = self.item_ids[rows]
        return ids


def overlap(a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]) -> float:
    """Mean share of each row of ``b`` that also appears in the same row of ``a``."""
    shares = [len(set(x) & set(y)) / len(y) for x, y in zip(a, b, strict=True)]
    return float(np.mean(shares))
