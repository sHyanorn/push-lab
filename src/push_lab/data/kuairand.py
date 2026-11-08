"""KuaiRand-Pure: extraction, conversion to Parquet and loading.

``is_click`` means a click in the two-column UI (tab 1) and a valid play in the
single-column UI, so loaders return ``tab == 1`` rows unless asked otherwise (D004).
"""

import tarfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from push_lab.data import schemas
from push_lab.data.schemas import Schema

DEFAULT_TAB = 1

TABLES: dict[str, tuple[str, Schema]] = {
    "log_standard_early": ("log_standard_4_08_to_4_21_pure.csv", schemas.KUAIRAND_LOG),
    "log_standard_late": ("log_standard_4_22_to_5_08_pure.csv", schemas.KUAIRAND_LOG),
    "log_random": ("log_random_4_22_to_5_08_pure.csv", schemas.KUAIRAND_LOG),
    "users": ("user_features_pure.csv", schemas.KUAIRAND_USERS),
    "videos_basic": ("video_features_basic_pure.csv", schemas.KUAIRAND_VIDEOS_BASIC),
    "videos_statistic": (
        "video_features_statistic_pure.csv",
        schemas.KUAIRAND_VIDEOS_STATISTIC,
    ),
}
LOGS = ("log_standard_early", "log_standard_late", "log_random")


def extract(archive: Path, dest: Path) -> Path:
    """Extract the archive safely and return the directory holding the CSV files."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # Rejects absolute paths, "..", links outside dest and special files
        tar.extractall(dest, filter="data")
    return dest / "KuaiRand-Pure" / "data"


def convert(archive: Path, work_dir: Path, out_dir: Path) -> list[Path]:
    """Extract ``archive`` into ``work_dir`` and write one Parquet file per table."""
    csv_dir = extract(archive, work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (filename, schema) in TABLES.items():
        target = out_dir / f"{name}.parquet"
        pq.write_table(schemas.read_csv(csv_dir / filename, schema), target)
        written.append(target)
    return written


def load(processed: Path, name: str, *, tab: int | None = DEFAULT_TAB) -> pd.DataFrame:
    """Load a processed table; log tables are filtered to ``tab`` unless it is None."""
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}")
    path = processed / f"{name}.parquet"
    if name in LOGS and tab is not None:
        return pd.read_parquet(path, filters=[("tab", "==", tab)])
    return pd.read_parquet(path)
