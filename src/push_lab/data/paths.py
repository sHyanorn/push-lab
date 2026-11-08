"""Default locations for raw data, processed tables and committed results."""

from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
RESULTS = Path("results")


def raw_dir(root: Path, dataset: str) -> Path:
    return root / RAW / dataset


def processed_dir(root: Path, dataset: str) -> Path:
    return root / PROCESSED / dataset
