"""Open Bandit Dataset, Men's campaign, under the random and BTS policies: the small sample,
and optionally the full campaign from the full archive."""

import zipfile
from pathlib import Path
from typing import IO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from push_lab.data import schemas

POLICIES = ("random", "bts")
POSITIONS = (1, 2, 3)


def _read(csv: Path | IO[bytes], raw: schemas.Schema, processed: schemas.Schema) -> pd.DataFrame:
    table = schemas.read_csv(csv, raw).drop_columns([""]).rename_columns(list(processed))
    df: pd.DataFrame = table.to_pandas()
    return df


def _validate(log: pd.DataFrame, name: str) -> None:
    p = log["propensity_score"]
    if not ((p > 0) & (p <= 1)).all():
        raise ValueError(f"{name}: propensity scores outside (0, 1]")
    if not log["position"].isin(POSITIONS).all():
        raise ValueError(f"{name}: positions outside {POSITIONS}")
    if not log["item_id"].between(0, schemas.OBD_MEN_ACTIONS - 1).all():
        raise ValueError(f"{name}: item ids outside the Men's campaign")
    if not log["click"].isin((0, 1)).all():
        raise ValueError(f"{name}: clicks are not binary")


def _write(out_dir: Path, policy: str, log: pd.DataFrame, items: pd.DataFrame) -> list[Path]:
    _validate(log, f"{policy}_men")
    written = []
    for name, df, schema in (
        (f"{policy}_men", log, schemas.OBD_LOG),
        (f"{policy}_men_items", items, schemas.OBD_ITEMS),
    ):
        target = out_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(df, schema=schemas.to_arrow(schema), preserve_index=False)
        pq.write_table(table, target)
        written.append(target)
    return written


def convert_full(archive: Path, out_dir: Path) -> list[Path]:
    """The Men's campaign of the full archive, read without extracting the other campaigns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(archive) as z:
        for policy in POLICIES:
            folder = f"open_bandit_dataset/{policy}/men"
            with z.open(f"{folder}/men.csv") as f:
                log = _read(f, schemas.OBD_LOG_RAW, schemas.OBD_LOG)
            with z.open(f"{folder}/item_context.csv") as f:
                items = _read(f, schemas.OBD_ITEMS_RAW, schemas.OBD_ITEMS)
            written += _write(out_dir, policy, log, items)
    return written


def convert(raw_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for policy in POLICIES:
        log = _read(raw_dir / f"{policy}_men.csv", schemas.OBD_LOG_RAW, schemas.OBD_LOG)
        items = _read(
            raw_dir / f"{policy}_men_item_context.csv", schemas.OBD_ITEMS_RAW, schemas.OBD_ITEMS
        )
        written += _write(out_dir, policy, log, items)
    return written


def load(processed: Path, policy: str, *, items: bool = False) -> pd.DataFrame:
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}")
    suffix = "_items" if items else ""
    return pd.read_parquet(processed / f"{policy}_men{suffix}.parquet")
