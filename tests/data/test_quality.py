import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from push_lab.data import fixtures, paths, quality, schemas


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    fixtures.write_processed(tmp_path, n=5_000)
    return tmp_path


def test_clean_synthetic_data_has_no_errors(tree: Path) -> None:
    report = quality.build(tree, published=False)
    assert report.errors == []
    payload = json.loads(quality.to_json(report))
    assert payload["datasets"]["kuairand"]["random_rows"] == 5_000
    assert set(payload["datasets"]["obd"]) == {"random", "bts"}


def test_report_is_deterministic(tree: Path) -> None:
    a = quality.to_json(quality.build(tree, published=False))
    b = quality.to_json(quality.build(tree, published=False))
    assert a == b


def test_published_figures_are_checked(tree: Path) -> None:
    errors = quality.build(tree).errors
    assert any("kuairand.log_random: 5000 rows, published" in e for e in errors)
    assert any("criteo.criteo: 5000 rows, published" in e for e in errors)
    # Synthetic Criteo is drawn with the published treatment share
    assert not any(e.startswith("criteo: treatment share") for e in errors)


def test_missing_table_is_an_error(tree: Path) -> None:
    (paths.processed_dir(tree, "hillstrom") / "hillstrom.parquet").unlink()
    errors = quality.build(tree, published=False).errors
    assert any(e.startswith("hillstrom.hillstrom: missing") for e in errors)


def test_non_uniform_random_propensity_is_an_error(tree: Path) -> None:
    path = paths.processed_dir(tree, "obd") / "random_men.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "propensity_score"] = 0.5
    table = pa.Table.from_pandas(df, schema=schemas.to_arrow(schemas.OBD_LOG))
    pq.write_table(table, path)
    errors = quality.build(tree, published=False).errors
    assert errors == ["obd.random: propensity is not uniform over the Men's actions"]


def test_schema_drift_is_an_error(tree: Path) -> None:
    path = paths.processed_dir(tree, "hillstrom") / "hillstrom.parquet"
    df = pd.read_parquet(path)
    df["visit"] = df["visit"].astype("float64")
    df.to_parquet(path, index=False)
    errors = quality.build(tree, published=False).errors
    assert "hillstrom.hillstrom: Parquet schema differs from the shared schema" in errors


def test_main_writes_report_and_fails_on_errors(tree: Path) -> None:
    assert quality.main(["--root", str(tree)]) == 1
    assert (tree / "results" / "data_quality.json").exists()
