"""Open-probability model and T-learner gate fitted on exploration logs (METHODS 12.3)."""

from dataclasses import dataclass
from typing import Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from push_lab.policy.features import Rows
from push_lab.uplift import learners

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

Kind = Literal["S", "G", "G2"]
KINDS: tuple[Kind, ...] = ("S", "G", "G2")


def _fit(x: FloatArray, y: npt.NDArray[np.bool_], threads: int) -> lgb.Booster:
    params = {**learners.PARAMS, "objective": "binary", "num_threads": threads}
    return lgb.train(params, lgb.Dataset(x, y.astype(np.float64)), learners.ROUNDS)


@dataclass(frozen=True)
class Models:
    """p: open given a delivery; treated, control: same-day visit with and without one."""

    p: lgb.Booster
    treated: lgb.Booster
    control: lgb.Booster
    threads: int = 1

    @classmethod
    def fit(cls, rows: Rows, threads: int = 1) -> Models:
        on = rows.treated
        return cls(
            _fit(rows.x[on], rows.opened[on], threads),
            _fit(rows.x[on], rows.visited[on], threads),
            _fit(rows.x[~on], rows.visited[~on], threads),
            threads,
        )

    def _predict(self, booster: lgb.Booster, x: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(booster.predict(x, num_threads=self.threads))
        return out

    def scores(self, f: FloatArray, kind: Kind) -> tuple[IntArray, FloatArray]:
        """Slot and user-day score from features [user, slot, feature]: every kind sends at
        the slot with the highest predicted open probability."""
        n, slots, width = f.shape
        if n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        p = self._predict(self.p, f.reshape(-1, width)).reshape(n, slots)
        slot = p.argmax(axis=1).astype(np.int64)
        p_best = p[np.arange(n), slot]
        if kind == "S":
            return slot, p_best
        x = f[np.arange(n), slot]
        control = self._predict(self.control, x)
        if kind == "G":
            return slot, self._predict(self.treated, x) - control
        return slot, p_best * (1 - control)
