from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from push_lab.data import obd, schemas


def _write(raw: Path, *, propensity: float = 1 / 34) -> None:
    rng = np.random.default_rng(0)
    n = 20
    for policy in obd.POLICIES:
        log = pd.DataFrame(
            {
                "timestamp": ["2019-11-24 00:03:13+00:00"] * n,
                "item_id": rng.integers(0, schemas.OBD_MEN_ACTIONS, size=n),
                "position": rng.integers(1, 4, size=n),
                "click": rng.integers(0, 2, size=n),
                "propensity_score": [propensity] * n,
                **{f"user_feature_{i}": ["a"] * n for i in range(4)},
                **{f"user-item_affinity_{i}": np.zeros(n) for i in range(schemas.OBD_MEN_ACTIONS)},
            }
        )
        log.to_csv(raw / f"{policy}_men.csv")
        items = pd.DataFrame(
            {
                "item_id": range(schemas.OBD_MEN_ACTIONS),
                "item_feature_0": 0.5,
                **{f"item_feature_{i}": "h" for i in range(1, 4)},
            }
        )
        items.to_csv(raw / f"{policy}_men_item_context.csv")


def test_convert_and_load(tmp_path: Path) -> None:
    _write(tmp_path)
    obd.convert(tmp_path, tmp_path / "out")
    log = obd.load(tmp_path / "out", "bts")
    assert list(log.columns) == list(schemas.OBD_LOG)
    assert len(log) == 20
    items = obd.load(tmp_path / "out", "random", items=True)
    assert list(items.columns) == list(schemas.OBD_ITEMS)


def test_rejects_invalid_propensity(tmp_path: Path) -> None:
    _write(tmp_path, propensity=0.0)
    with pytest.raises(ValueError, match="propensity"):
        obd.convert(tmp_path, tmp_path / "out")
