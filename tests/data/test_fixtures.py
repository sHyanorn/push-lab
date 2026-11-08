from pathlib import Path

import pandas as pd
import pyarrow as pa

from push_lab.data import fixtures, kuairand, obd, paths, schemas, uplift_trials


def test_every_table_matches_its_schema() -> None:
    tables = fixtures.processed_tables(seed=0, n=50)
    assert tables.keys() == fixtures.SCHEMAS.keys()
    for key, df in tables.items():
        schema = schemas.to_arrow(fixtures.SCHEMAS[key])
        assert list(df.columns) == schema.names, key
        # Raises if a column cannot be represented with its schema type
        pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def test_same_seed_same_data() -> None:
    a = fixtures.processed_tables(seed=3, n=30)
    b = fixtures.processed_tables(seed=3, n=30)
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key])


def test_loaders_read_the_synthetic_tree(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=200)
    kr = paths.processed_dir(tmp_path, "kuairand")
    log = kuairand.load(kr, "log_random")
    assert (log["tab"] == 1).all() and len(log) > 0
    assert len(kuairand.load(kr, "users")) == fixtures.N_USERS

    hill = uplift_trials.load_hillstrom(
        paths.processed_dir(tmp_path, "hillstrom") / "hillstrom.parquet"
    )
    assert set(hill["treatment"]) == {0, 1}
    crit = paths.processed_dir(tmp_path, "criteo") / "criteo.parquet"
    assert uplift_trials.load_criteo(crit, train=True)["train"].all()

    for policy in obd.POLICIES:
        obd._validate(obd.load(paths.processed_dir(tmp_path, "obd"), policy), policy)
