"""Checks against the live sources and the real processed data; deselected by default."""

import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from push_lab.data import fixtures, paths, schemas
from push_lab.data.download import fetch
from push_lab.data.sources import HILLSTROM, OBD_MEN, SOURCES, Source

pytestmark = pytest.mark.network

ALL_SOURCES = [s for group in SOURCES.values() for s in group]


@pytest.mark.parametrize("source", ALL_SOURCES, ids=lambda s: s.filename)
def test_source_reachable_with_expected_size(source: Source) -> None:
    request = urllib.request.Request(source.url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status == 206:
            total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        else:
            total = int(response.headers["Content-Length"])
    assert total == source.size


@pytest.mark.parametrize("source", [HILLSTROM, *OBD_MEN], ids=lambda s: s.filename)
def test_small_sources_match_checksums(source: Source, tmp_path: Path) -> None:
    assert fetch(source, tmp_path).stat().st_size == source.size


@pytest.mark.parametrize("key", list(fixtures.SCHEMAS), ids=lambda k: "/".join(k))
def test_real_processed_tables_match_schemas(key: tuple[str, str]) -> None:
    dataset, name = key
    path = paths.processed_dir(Path.cwd(), dataset) / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not prepared")
    assert pq.read_schema(path).remove_metadata().equals(schemas.to_arrow(fixtures.SCHEMAS[key]))


def test_kuairand_time_zone_is_utc8() -> None:
    from push_lab.data import kuairand
    from push_lab.sim import inputs

    processed = paths.processed_dir(Path.cwd(), "kuairand")
    if not (processed / "log_random.parquet").exists():
        pytest.skip("KuaiRand is not prepared")
    log = kuairand.load(processed, "log_random")
    assert int(inputs.hourly_shares(log).argmin()) < 6
    agreement = inputs.date_agreement(log, offsets=tuple(range(-12, 15)))
    assert max(agreement, key=lambda h: agreement[h]) == inputs.UTC_OFFSET_HOURS
