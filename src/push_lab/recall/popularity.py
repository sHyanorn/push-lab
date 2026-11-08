"""Popularity and trending sources (METHODS Section 2)."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab.rank import splits

TRENDING_DAYS = 3


def _ranked(counts: pd.DataFrame, keys: list[str]) -> npt.NDArray[np.int64]:
    order = counts.sort_values(keys, ascending=[False] * (len(keys) - 1) + [True])
    return order["video_id"].to_numpy(dtype=np.int64)


def _clicks(log: pd.DataFrame, name: str) -> pd.DataFrame:
    return log.groupby("video_id")["is_click"].sum().rename(name).reset_index()


def popularity(window: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Item ids by clicks in the window, most clicked first; ties by item id."""
    return _ranked(_clicks(window, "is_click"), ["is_click", "video_id"])


def trending(window: pd.DataFrame, cutoff: int) -> npt.NDArray[np.int64]:
    """Item ids by clicks in the last days before ``cutoff``; ties by window clicks, then id."""
    recent_start = splits.shift(cutoff, -TRENDING_DAYS)
    recent = _clicks(window[window["date"] >= recent_start], "recent")
    counts = _clicks(window, "is_click").merge(recent, on="video_id", how="left")
    counts["recent"] = counts["recent"].fillna(0)
    return _ranked(counts, ["recent", "is_click", "video_id"])
