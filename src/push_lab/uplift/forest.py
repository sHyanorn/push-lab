"""Causal forest from EconML (METHODS Section 8)."""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from econml.dml import CausalForestDML
from sklearn.dummy import DummyClassifier

from push_lab.uplift.learners import PARAMS, ROUNDS, FloatArray, Labels

TREES = 100
# Rows drawn from the Criteo training sample for the forest (SPEC Section 10)
CRITEO_FOREST_ROWS = 200_000
ROUND_DECIMALS = 12


def _outcome_model() -> lgb.LGBMRegressor:
    p = PARAMS
    return lgb.LGBMRegressor(
        n_estimators=ROUNDS,
        num_leaves=p["num_leaves"],
        learning_rate=p["learning_rate"],
        min_child_samples=p["min_data_in_leaf"],
        colsample_bytree=p["feature_fraction"],
        subsample=p["bagging_fraction"],
        subsample_freq=p["bagging_freq"],
        random_state=p["seed"],
        n_jobs=p["num_threads"],
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


@dataclass(frozen=True)
class CausalForest:
    model: Any

    @classmethod
    def fit(cls, x: FloatArray, t: Labels, y: Labels, seed: int = 0) -> CausalForest:
        model = CausalForestDML(
            model_y=_outcome_model(),
            # Randomized assignment: the propensity is the constant treated share
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            cv=2,
            n_estimators=TREES,
            random_state=seed,
            n_jobs=PARAMS["num_threads"],
        )
        model.fit(y.astype(np.float64), t.astype(np.int64), X=x)
        return cls(model)

    def effect(self, x: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(self.model.effect(x), dtype=np.float64).ravel()
        # Trees are averaged across threads, so the last bit can differ between runs; rounding
        # keeps bin membership, and with it every bootstrap draw, reproducible
        rounded: FloatArray = np.round(out, ROUND_DECIMALS)
        return rounded
