"""Synthetic stand-ins for every processed table, generated from the shared schemas.

CI never downloads data. Tests and later milestones write this tree with
``write_processed`` and read it with the same loaders as the real data.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from push_lab.data import kuairand, paths, schemas

Generator = Callable[[np.random.Generator, int], pd.DataFrame]

N_USERS = 50
N_VIDEOS = 80
N_AUTHORS = 20


def _fill(schema: schemas.Schema, rng: np.random.Generator, n: int) -> dict[str, object]:
    """Type-correct default columns; table generators override the ones that matter."""
    columns: dict[str, object] = {}
    for name, dtype in schema.items():
        if pa.types.is_floating(dtype):
            columns[name] = rng.random(n)
        elif pa.types.is_integer(dtype):
            columns[name] = rng.integers(0, 2, size=n)
        elif pa.types.is_boolean(dtype):
            columns[name] = rng.random(n) < 0.5
        else:
            columns[name] = [f"{name}_{i % 3}" for i in range(n)]
    return columns


def _frame(schema: schemas.Schema, columns: dict[str, object]) -> pd.DataFrame:
    table = pa.Table.from_pydict(columns, schema=schemas.to_arrow(schema))
    df: pd.DataFrame = table.to_pandas()
    return df


def kuairand_log(rng: np.random.Generator, n: int, *, random_exposure: bool) -> pd.DataFrame:
    cols = _fill(schemas.KUAIRAND_LOG, rng, n)
    cols.update(
        user_id=rng.integers(0, N_USERS, size=n),
        video_id=rng.integers(0, N_VIDEOS, size=n),
        date=rng.choice([20220422, 20220423, 20220424], size=n),
        hourmin=rng.integers(0, 24, size=n) * 100,
        time_ms=np.sort(rng.integers(1_650_000_000_000, 1_651_000_000_000, size=n)),
        is_click=(rng.random(n) < 0.2).astype(np.int64),
        long_view=(rng.random(n) < 0.3).astype(np.int64),
        play_time_ms=rng.integers(0, 60_000, size=n),
        duration_ms=rng.integers(5_000, 120_000, size=n),
        is_rand=np.full(n, int(random_exposure)),
        tab=np.where(rng.random(n) < (0.99 if random_exposure else 0.6), 1, 0),
    )
    return _frame(schemas.KUAIRAND_LOG, cols)


def kuairand_users(rng: np.random.Generator) -> pd.DataFrame:
    cols = _fill(schemas.KUAIRAND_USERS, rng, N_USERS)
    cols["user_id"] = np.arange(N_USERS)
    return _frame(schemas.KUAIRAND_USERS, cols)


def kuairand_videos_basic(rng: np.random.Generator) -> pd.DataFrame:
    cols = _fill(schemas.KUAIRAND_VIDEOS_BASIC, rng, N_VIDEOS)
    cols.update(
        video_id=np.arange(N_VIDEOS),
        author_id=rng.integers(0, N_AUTHORS, size=N_VIDEOS),
        video_duration=rng.integers(5_000, 120_000, size=N_VIDEOS).astype(float),
        tag=[str(t) for t in rng.integers(0, 10, size=N_VIDEOS)],
    )
    return _frame(schemas.KUAIRAND_VIDEOS_BASIC, cols)


def kuairand_videos_statistic(rng: np.random.Generator) -> pd.DataFrame:
    cols = _fill(schemas.KUAIRAND_VIDEOS_STATISTIC, rng, N_VIDEOS)
    cols["video_id"] = np.arange(N_VIDEOS)
    return _frame(schemas.KUAIRAND_VIDEOS_STATISTIC, cols)


def hillstrom(rng: np.random.Generator, n: int) -> pd.DataFrame:
    cols = _fill(schemas.HILLSTROM_PROCESSED, rng, n)
    treatment = (rng.random(n) < 0.5).astype(np.int64)
    cols.update(
        segment=np.where(treatment == 1, "Womens E-Mail", "No E-Mail").tolist(),
        treatment=treatment,
        visit=(rng.random(n) < 0.1 + 0.05 * treatment).astype(np.int64),
        conversion=(rng.random(n) < 0.01).astype(np.int64),
        spend=np.zeros(n),
    )
    return _frame(schemas.HILLSTROM_PROCESSED, cols)


def criteo(rng: np.random.Generator, n: int) -> pd.DataFrame:
    cols = _fill(schemas.CRITEO_PROCESSED, rng, n)
    # Exactly the published treatment share, so checks against it do not depend on the seed
    treatment = rng.permutation(np.arange(n) < round(0.85 * n))
    visit = rng.random(n) < 0.04 + 0.01 * treatment
    cols.update(
        treatment=treatment.astype(np.int8),
        visit=visit.astype(np.int8),
        conversion=(visit & (rng.random(n) < 0.06)).astype(np.int8),
        exposure=(treatment & (rng.random(n) < 0.03)).astype(np.int8),
        train=rng.random(n) < 0.1,
    )
    return _frame(schemas.CRITEO_PROCESSED, cols)


def obd_log(rng: np.random.Generator, n: int, *, policy: str) -> pd.DataFrame:
    actions = schemas.OBD_MEN_ACTIONS
    cols = _fill(schemas.OBD_LOG, rng, n)
    propensity = np.full(n, 1 / actions) if policy == "random" else rng.uniform(0.001, 0.7, size=n)
    cols.update(
        timestamp=["2019-11-24 00:00:00+00:00"] * n,
        item_id=rng.integers(0, actions, size=n),
        position=rng.integers(1, 4, size=n),
        click=(rng.random(n) < 0.005).astype(np.int64),
        propensity_score=propensity,
    )
    return _frame(schemas.OBD_LOG, cols)


def obd_items(rng: np.random.Generator) -> pd.DataFrame:
    cols = _fill(schemas.OBD_ITEMS, rng, schemas.OBD_MEN_ACTIONS)
    cols["item_id"] = np.arange(schemas.OBD_MEN_ACTIONS)
    return _frame(schemas.OBD_ITEMS, cols)


def processed_tables(seed: int = 0, n: int = 500) -> dict[tuple[str, str], pd.DataFrame]:
    """Every processed table, keyed by (dataset, table name)."""
    rng = np.random.default_rng(seed)
    return {
        ("kuairand", "log_standard_early"): kuairand_log(rng, n, random_exposure=False),
        ("kuairand", "log_standard_late"): kuairand_log(rng, n, random_exposure=False),
        ("kuairand", "log_random"): kuairand_log(rng, n, random_exposure=True),
        ("kuairand", "users"): kuairand_users(rng),
        ("kuairand", "videos_basic"): kuairand_videos_basic(rng),
        ("kuairand", "videos_statistic"): kuairand_videos_statistic(rng),
        ("hillstrom", "hillstrom"): hillstrom(rng, n),
        ("criteo", "criteo"): criteo(rng, n),
        ("obd", "random_men"): obd_log(rng, n, policy="random"),
        ("obd", "random_men_items"): obd_items(rng),
        ("obd", "bts_men"): obd_log(rng, n, policy="bts"),
        ("obd", "bts_men_items"): obd_items(rng),
    }


SCHEMAS: dict[tuple[str, str], schemas.Schema] = {
    **{("kuairand", name): schema for name, (_, schema) in kuairand.TABLES.items()},
    ("hillstrom", "hillstrom"): schemas.HILLSTROM_PROCESSED,
    ("criteo", "criteo"): schemas.CRITEO_PROCESSED,
    ("obd", "random_men"): schemas.OBD_LOG,
    ("obd", "random_men_items"): schemas.OBD_ITEMS,
    ("obd", "bts_men"): schemas.OBD_LOG,
    ("obd", "bts_men_items"): schemas.OBD_ITEMS,
}


def write_processed(root: Path, seed: int = 0, n: int = 500) -> None:
    """Write a synthetic ``data/processed`` tree under ``root``."""
    for (dataset, name), df in processed_tables(seed, n).items():
        out = paths.processed_dir(root, dataset)
        out.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, schema=schemas.to_arrow(SCHEMAS[dataset, name]))
        pq.write_table(table, out / f"{name}.parquet")
