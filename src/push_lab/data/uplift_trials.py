"""Randomized uplift trials: Hillstrom e-mail and Criteo advertising."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from push_lab.data import schemas

# Hillstrom contrast (D004): women's e-mail against no e-mail
HILLSTROM_TREATED = "Womens E-Mail"
HILLSTROM_CONTROL = "No E-Mail"

CRITEO_TRAIN_FRACTION = 0.1
CRITEO_SEED = 0
CRITEO_BLOCK_BYTES = 16 << 20


def _check_columns(names: list[str], schema: schemas.Schema, source: Path) -> None:
    if names != list(schema):
        raise ValueError(f"{source.name}: columns {names} do not match the schema")


def convert_hillstrom(csv: Path, out: Path) -> Path:
    table = pacsv.read_csv(
        csv,
        convert_options=pacsv.ConvertOptions(column_types=schemas.to_arrow(schemas.HILLSTROM)),
    )
    _check_columns(table.column_names, schemas.HILLSTROM, csv)
    df = table.to_pandas()
    df = df[df["segment"].isin([HILLSTROM_TREATED, HILLSTROM_CONTROL])].reset_index(drop=True)
    df["treatment"] = (df["segment"] == HILLSTROM_TREATED).astype("int64")
    out.parent.mkdir(parents=True, exist_ok=True)
    arrow = pa.Table.from_pandas(df, schema=schemas.to_arrow(schemas.HILLSTROM_PROCESSED))
    pq.write_table(arrow, out)
    return out


def convert_criteo(
    csv_gz: Path,
    out: Path,
    *,
    seed: int = CRITEO_SEED,
    block_bytes: int = CRITEO_BLOCK_BYTES,
) -> Path:
    """Stream the gzipped CSV into one Parquet file with a seeded ``train`` flag.

    Uniform draws are taken row by row from one generator, so the flag does not depend on
    how the file is split into batches.
    """
    rng = np.random.default_rng(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    target = schemas.to_arrow(schemas.CRITEO_PROCESSED)
    reader = pacsv.open_csv(
        # pyarrow-stubs 20 types the argument narrower than pyarrow 26 accepts
        pa.input_stream(csv_gz, compression="gzip"),  # type: ignore[arg-type]
        read_options=pacsv.ReadOptions(block_size=block_bytes),
        convert_options=pacsv.ConvertOptions(column_types=schemas.to_arrow(schemas.CRITEO)),
    )
    _check_columns(reader.schema.names, schemas.CRITEO, csv_gz)
    with pq.ParquetWriter(out, target) as writer:
        for batch in reader:
            train = pa.array(rng.random(batch.num_rows) < CRITEO_TRAIN_FRACTION)
            columns = [*batch.columns, train]
            writer.write_batch(pa.RecordBatch.from_arrays(columns, schema=target))
    return out


def load_hillstrom(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_criteo(path: Path, *, train: bool | None = None) -> pd.DataFrame:
    """Load Criteo; ``train=True`` gives the training sample, ``False`` the evaluation rows."""
    if train is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, filters=[("train", "==", train)])
