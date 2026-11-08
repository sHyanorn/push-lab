"""Open Bandit Dataset small sample, Men's campaign, under the random and BTS policies."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from push_lab.data import schemas

POLICIES = ("random", "bts")
POSITIONS = (1, 2, 3)


def _read(csv: Path, raw: schemas.Schema, processed: schemas.Schema) -> pd.DataFrame:
    table = pacsv.read_csv(
        csv, convert_options=pacsv.ConvertOptions(column_types=schemas.to_arrow(raw))
    )
    if table.column_names != list(raw):
        raise ValueError(f"{csv.name}: columns {table.column_names} do not match the schema")
    table = table.drop_columns([""]).rename_columns(list(processed))
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


def convert(raw_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for policy in POLICIES:
        log = _read(raw_dir / f"{policy}_men.csv", schemas.OBD_LOG_RAW, schemas.OBD_LOG)
        _validate(log, f"{policy}_men")
        items = _read(
            raw_dir / f"{policy}_men_item_context.csv", schemas.OBD_ITEMS_RAW, schemas.OBD_ITEMS
        )
        for name, df, schema in (
            (f"{policy}_men", log, schemas.OBD_LOG),
            (f"{policy}_men_items", items, schemas.OBD_ITEMS),
        ):
            target = out_dir / f"{name}.parquet"
            table = pa.Table.from_pandas(df, schema=schemas.to_arrow(schema), preserve_index=False)
            pq.write_table(table, target)
            written.append(target)
    return written


def load(processed: Path, policy: str, *, items: bool = False) -> pd.DataFrame:
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}")
    suffix = "_items" if items else ""
    return pd.read_parquet(processed / f"{policy}_men{suffix}.parquet")
