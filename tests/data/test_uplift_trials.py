import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from push_lab.data import schemas, uplift_trials


def _write_criteo(path: Path, n: int) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
    for flag in ("treatment", "conversion", "visit", "exposure"):
        df[flag] = rng.integers(0, 2, size=n)
    with gzip.open(path, "wt", newline="") as f:
        df.to_csv(f, index=False)


def test_criteo_train_flag_does_not_depend_on_batching(tmp_path: Path) -> None:
    src = tmp_path / "criteo.csv.gz"
    _write_criteo(src, 20_000)
    small = uplift_trials.convert_criteo(src, tmp_path / "small.parquet", block_bytes=1 << 14)
    large = uplift_trials.convert_criteo(src, tmp_path / "large.parquet", block_bytes=1 << 26)
    a = uplift_trials.load_criteo(small)
    b = uplift_trials.load_criteo(large)
    assert a["train"].tolist() == b["train"].tolist()
    expected = np.random.default_rng(uplift_trials.CRITEO_SEED).random(20_000) < 0.1
    assert a["train"].to_numpy().tolist() == expected.tolist()
    assert list(a.columns) == list(schemas.CRITEO_PROCESSED)


def test_criteo_train_and_eval_partition_rows(tmp_path: Path) -> None:
    src = tmp_path / "criteo.csv.gz"
    _write_criteo(src, 5_000)
    out = uplift_trials.convert_criteo(src, tmp_path / "c.parquet")
    train = uplift_trials.load_criteo(out, train=True)
    evaluation = uplift_trials.load_criteo(out, train=False)
    assert len(train) + len(evaluation) == 5_000
    assert 0.07 < len(train) / 5_000 < 0.13


def test_hillstrom_keeps_womens_contrast(tmp_path: Path) -> None:
    header = ",".join(schemas.HILLSTROM)
    rows = [
        "10,2) $100 - $200,142.44,1,0,Surburban,0,Phone,Womens E-Mail,0,0,0",
        "6,3) $200 - $350,329.08,1,1,Rural,1,Web,No E-Mail,1,0,0",
        "7,2) $100 - $200,180.65,0,1,Surburban,1,Web,Mens E-Mail,0,0,0",
    ]
    csv = tmp_path / "h.csv"
    csv.write_text("\n".join([header, *rows]) + "\n")
    df = uplift_trials.load_hillstrom(uplift_trials.convert_hillstrom(csv, tmp_path / "h.parquet"))
    assert df["segment"].tolist() == ["Womens E-Mail", "No E-Mail"]
    assert df["treatment"].tolist() == [1, 0]
