"""LightGBM fine ranker with click and long-view heads (METHODS Section 5)."""

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab.rank.features import CATEGORICAL, FeatureBuilder

TARGETS = ("is_click", "long_view")
FUSION_WEIGHT = 0.5

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "num_leaves": 63,
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
MAX_ROUNDS = 2000
EARLY_STOPPING = 50

FloatArray = npt.NDArray[np.float64]


@dataclass
class FineRanker:
    boosters: dict[str, lgb.Booster] = field(default_factory=dict)
    best_rounds: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        builder: FeatureBuilder,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> FineRanker:
        """Fit one booster per target; features come from ``builder``'s cutoff."""
        params = {**PARAMS, **(params or {})}
        x_train, x_valid = builder.build_rows(train), builder.build_rows(valid)
        ranker = cls()
        for target in TARGETS:
            d_train = lgb.Dataset(
                x_train, train[target].to_numpy(), categorical_feature=CATEGORICAL
            )
            d_valid = lgb.Dataset(
                x_valid,
                valid[target].to_numpy(),
                categorical_feature=CATEGORICAL,
                reference=d_train,
            )
            booster = lgb.train(
                params,
                d_train,
                num_boost_round=MAX_ROUNDS,
                valid_sets=[d_valid],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            ranker.boosters[target] = booster
            ranker.best_rounds[target] = booster.best_iteration
        return ranker

    def predict(self, x: pd.DataFrame) -> dict[str, FloatArray]:
        out: dict[str, FloatArray] = {}
        for target, booster in self.boosters.items():
            pred = booster.predict(x, num_iteration=booster.best_iteration)
            out[target] = np.asarray(pred, dtype=np.float64)
        return out


def fuse(pred: dict[str, FloatArray], w: float = FUSION_WEIGHT) -> FloatArray:
    """r = w * p_click + (1 - w) * p_long."""
    fused: FloatArray = w * pred["is_click"] + (1 - w) * pred["long_view"]
    return fused
