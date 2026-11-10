"""Cutoffs, fitting windows and the random-exposure user split (METHODS Section 1)."""

from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd

FIRST_DATE = 20220409
# Features for the fine ranker's training rows
CUTOFF_TRAIN = 20220415
# Retrieval, coarse and fine ranking at evaluation
CUTOFF_EVAL = 20220422
# Fine-ranker labels: train on the first range, early-stop on the second
LABELS_TRAIN = (20220415, 20220419)
LABELS_VALID = (20220420, 20220421)
SPLIT_SEED = 0


def to_date(yyyymmdd: int) -> date:
    return date(yyyymmdd // 10000, yyyymmdd // 100 % 100, yyyymmdd % 100)


def from_date(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def shift(yyyymmdd: int, days: int) -> int:
    return from_date(to_date(yyyymmdd) + timedelta(days=days))


def window_days(cutoff: int) -> int:
    """Number of days in the fitting window before ``cutoff``."""
    return (to_date(cutoff) - to_date(FIRST_DATE)).days


def window(standard: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Standard-log rows a model fitted at ``cutoff`` may use."""
    return standard[standard["date"] < cutoff].reset_index(drop=True)


def between(log: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """Rows dated from ``first`` to ``last``, both included."""
    return log[log["date"].between(first, last)].reset_index(drop=True)


def user_halves(
    random_log: pd.DataFrame, seed: int = SPLIT_SEED
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Split random-exposure users into the evaluation half and the simulator half."""
    users = np.sort(random_log["user_id"].unique()).astype(np.int64)
    permuted = np.random.default_rng(seed).permutation(users)
    half = len(permuted) // 2
    return np.sort(permuted[:half]), np.sort(permuted[half:])
