"""S-, T- and X-learners on LightGBM (METHODS Section 8)."""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Labels = npt.NDArray[np.integer]

ROUNDS = 200
PARAMS: dict[str, Any] = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 0,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def fit_booster(x: FloatArray, target: FloatArray, objective: str) -> lgb.Booster:
    return lgb.train({**PARAMS, "objective": objective}, lgb.Dataset(x, target), ROUNDS)


def predict(booster: lgb.Booster, x: FloatArray) -> FloatArray:
    out: FloatArray = np.asarray(booster.predict(x), dtype=np.float64)
    return out


def _with_treatment(x: FloatArray, value: float) -> FloatArray:
    return np.column_stack([x, np.full(len(x), value)])


@dataclass(frozen=True)
class SLearner:
    model: lgb.Booster

    @classmethod
    def fit(cls, x: FloatArray, t: Labels, y: Labels) -> SLearner:
        xt = np.column_stack([x, t.astype(np.float64)])
        return cls(fit_booster(xt, y.astype(np.float64), "binary"))

    def effect(self, x: FloatArray) -> FloatArray:
        return predict(self.model, _with_treatment(x, 1.0)) - predict(
            self.model, _with_treatment(x, 0.0)
        )


@dataclass(frozen=True)
class TLearner:
    treated: lgb.Booster
    control: lgb.Booster

    @classmethod
    def fit(cls, x: FloatArray, t: Labels, y: Labels) -> TLearner:
        on = t == 1
        target = y.astype(np.float64)
        return cls(
            fit_booster(x[on], target[on], "binary"), fit_booster(x[~on], target[~on], "binary")
        )

    def response(self, x: FloatArray) -> FloatArray:
        """Predicted outcome if treated: the response-model baseline."""
        return predict(self.treated, x)

    def effect(self, x: FloatArray) -> FloatArray:
        return predict(self.treated, x) - predict(self.control, x)


@dataclass(frozen=True)
class XLearner:
    """Künzel et al. (2019) with the constant propensity of a randomized trial as g(x)."""

    tau_treated: lgb.Booster
    tau_control: lgb.Booster
    propensity: float

    @classmethod
    def fit(cls, x: FloatArray, t: Labels, y: Labels, stage1: TLearner | None = None) -> XLearner:
        first = TLearner.fit(x, t, y) if stage1 is None else stage1
        on = t == 1
        target = y.astype(np.float64)
        imputed_treated = target[on] - predict(first.control, x[on])
        imputed_control = predict(first.treated, x[~on]) - target[~on]
        return cls(
            fit_booster(x[on], imputed_treated, "regression"),
            fit_booster(x[~on], imputed_control, "regression"),
            float(on.mean()),
        )

    def effect(self, x: FloatArray) -> FloatArray:
        e = self.propensity
        return e * predict(self.tau_control, x) + (1 - e) * predict(self.tau_treated, x)
