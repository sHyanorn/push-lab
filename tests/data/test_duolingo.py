import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from push_lab.data import duolingo, fixtures
from push_lab.data.sources import DUOLINGO

ROWS = 2_500
ONE_EACH = {"train": ("train-part-1",), "test": ("test-part-1",)}


def _convert(raw: Path, out: Path, rows: dict[str, int]) -> list[Path]:
    return duolingo.convert(raw, out, rows, ONE_EACH)


def test_sources_are_the_six_published_archives_and_the_readme() -> None:
    archives = [s for s in DUOLINGO if s.filename.endswith(".tar.gz")]
    assert sorted(s.filename.removesuffix(".tar.gz") for s in archives) == sorted(
        duolingo.ARCHIVES["train"] + duolingo.ARCHIVES["test"]
    )
    assert sum(s.size for s in archives) == 9_348_915_292
    assert len({s.url for s in DUOLINGO}) == 7
    assert all(s.algorithm == "md5" and len(s.checksum) == 32 for s in DUOLINGO)


def _expected(table: pa.Table) -> dict[str, np.ndarray]:
    """Recency, j, days since the previous notification and history length, row by row from the
    raw history."""
    k = len(duolingo.TEMPLATES)
    out: dict[str, list[Any]] = {"recency": [], "j": [], "since": [], "length": []}
    histories: list[Any] = table["history"].to_pylist()
    times: list[Any] = table["datetime"].to_pylist()
    for when, history in zip(times, histories, strict=True):
        history = history or []
        recency: list[Any] = [np.inf] * k
        for entry in history:
            i = duolingo.TEMPLATES.index(entry["template"])
            recency[i] = min(recency[i], np.float32(entry["n_days"]))
        frac = when - np.floor(when)
        out["recency"].append(recency)
        out["j"].append(sum(np.float64(np.float32(e["n_days"])) < frac for e in history))
        out["since"].append(min(recency))
        out["length"].append(len(history))
    return {key: np.array(value) for key, value in out.items()}


def test_conversion_keeps_every_row_of_every_member_in_order(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, ROWS, row_group=700)
    written = _convert(raw, out, rows)
    report = json.loads((out / duolingo.REPORT).read_text("utf-8"))
    assert report["train"]["rows"] == rows["train"] == 2 * ROWS
    summary = report["train"]["archives"][0]
    assert len(summary["members"]) == 2 and summary["members"][0]["row_groups"] == 4
    assert sorted(Path(s).name for s in summary["skipped"]) == [
        ".DS_Store",
        "._.DS_Store",
        "._part-00000.snappy.parquet",
        "._part-00001.snappy.parquet",
        ".part-00000.snappy.parquet.crc",
        ".part-00001.snappy.parquet.crc",
        "logs",
    ]
    assert len([p for p in written if p.suffix == ".parquet"]) == 4
    assert not (raw / "tmp" / "train-part-1.member.parquet").exists()

    with tarfile.open(raw / "train-part-1.tar.gz") as tar:
        member = tar.extractfile("logs/part-00001.snappy.parquet")
        assert member is not None
        source = pq.read_table(io.BytesIO(member.read()))
    converted = pq.read_table(duolingo.files(out, "train")[1])
    assert converted.schema.equals(duolingo.OUTPUT)
    np.testing.assert_array_equal(converted["datetime"], source["datetime"])
    expected = _expected(source)
    recency = np.column_stack([converted[c].to_numpy() for c in duolingo.RECENCY])
    np.testing.assert_array_equal(recency, expected["recency"])
    np.testing.assert_array_equal(converted["j"], expected["j"])
    np.testing.assert_array_equal(converted["since_previous"], expected["since"])
    np.testing.assert_array_equal(converted["history_length"], expected["length"])
    assert expected["j"].max() >= 2 and (expected["length"] == 0).any()
    letters = np.array(list(duolingo.TEMPLATES))
    eligible = converted["eligible"].to_numpy()
    sets_of: list[Any] = source["eligible_templates"].to_pylist()
    for i, sets in enumerate(sets_of[:200]):
        assert sorted(sets) == list(letters[(eligible[i] >> np.arange(12)) & 1 == 1])
    np.testing.assert_array_equal(
        letters[converted["selected"].to_numpy()], source["selected_template"].to_numpy()
    )
    assert converted["ui_language"].to_pylist() == source["ui_language"].to_pylist()


def test_a_finished_archive_is_not_converted_again(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, 300)
    _convert(raw, out, rows)
    (raw / "train-part-1.tar.gz").unlink()
    (out / duolingo.REPORT).unlink()
    _convert(raw, out, rows)
    assert (out / duolingo.REPORT).exists()


def _archive(
    tmp_path: Path, table: pa.Table, extra: tuple[tarfile.TarInfo, bytes] | None = None
) -> Path:
    raw = tmp_path / "raw"
    fixtures.duolingo_archive(raw / "train-part-1.tar.gz", [table], extra=extra)
    fixtures.duolingo_archive(
        raw / "test-part-1.tar.gz", [fixtures.duolingo_table(np.random.default_rng(1), 50, (3, 4))]
    )
    return raw


def _stops(raw: Path, out: Path, message: str) -> None:
    with pytest.raises(duolingo.ConversionError, match=message):
        _convert(raw, out, {"train": 100, "test": 50})
    assert not (out / duolingo.REPORT).exists()
    with pytest.raises(duolingo.ConversionError):
        duolingo.files(out, "train")


def _table(rows: int = 100) -> pa.Table:
    return fixtures.duolingo_table(np.random.default_rng(0), rows, (0.0, 2.0))


def test_refuses_a_schema_with_another_column(tmp_path: Path) -> None:
    table = _table().append_column("user_id", pa.array(np.arange(100)))
    _stops(_archive(tmp_path, table), tmp_path / "out", "unexpected schema")


def test_refuses_a_member_path_outside_the_target_folder(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    pq.write_table(_table(), buffer)
    info = tarfile.TarInfo("../outside.parquet")
    info.size = len(buffer.getvalue())
    raw = _archive(tmp_path, _table(), extra=(info, buffer.getvalue()))
    _stops(raw, tmp_path / "out", "leaves the target folder")
    assert not (tmp_path / "raw" / "outside.parquet").exists()


def test_refuses_a_link_named_like_parquet(tmp_path: Path) -> None:
    info = tarfile.TarInfo("logs/part-00009.snappy.parquet")
    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
    _stops(_archive(tmp_path, _table(), extra=(info, b"")), tmp_path / "out", "unexpected member")


def _replace(table: pa.Table, name: str, values: pa.Array[Any]) -> pa.Table:
    i = table.schema.get_field_index(name)
    return table.set_column(i, table.schema.field(i), values)


def test_stops_on_a_selected_template_outside_its_eligible_set(tmp_path: Path) -> None:
    table = _table()
    eligible: list[Any] = table["eligible_templates"].to_pylist()
    letters = [next(t for t in duolingo.TEMPLATES if t not in e) for e in eligible[:1]]
    selected = letters + table["selected_template"].to_pylist()[1:]
    table = _replace(table, "selected_template", pa.array(selected))
    _stops(_archive(tmp_path, table), tmp_path / "out", "outside its eligible set")


def test_stops_on_a_letter_outside_a_to_l(tmp_path: Path) -> None:
    table = _table()
    history: list[Any] = table["history"].to_pylist()
    owner = next(i for i, h in enumerate(history) if h)
    history[owner][0]["template"] = "M"
    table = _replace(table, "history", pa.array(history, table.schema.field("history").type))
    _stops(_archive(tmp_path, table), tmp_path / "out", "outside A to L")


@pytest.mark.parametrize(
    ("column", "value"), [("datetime", np.nan), ("n_days", np.nan), ("n_days", np.inf)]
)
def test_stops_on_a_nan_or_infinite_time(tmp_path: Path, column: str, value: float) -> None:
    """Arrow does not count NaN as null, so the null checks of D070 let it through."""
    table = _table()
    if column == "datetime":
        when = table["datetime"].to_numpy().copy()
        when[0] = value
        table = _replace(table, "datetime", pa.array(when))
    else:
        history: list[Any] = table["history"].to_pylist()
        owner = next(i for i, h in enumerate(history) if h)
        history[owner][0]["n_days"] = value
        table = _replace(table, "history", pa.array(history, table.schema.field("history").type))
    _stops(_archive(tmp_path, table), tmp_path / "out", "NaN or infinite")


def test_stops_on_a_repeated_letter(tmp_path: Path) -> None:
    """Two L's carry past the twelve bits of the eligible mask."""
    table = _table()
    eligible: list[Any] = table["eligible_templates"].to_pylist()
    selected: list[Any] = table["selected_template"].to_pylist()
    eligible[0], selected[0] = ["L", "L"], "L"
    table = _replace(table, "eligible_templates", pa.array(eligible, table.schema.field(2).type))
    table = _replace(table, "selected_template", pa.array(selected))
    _stops(_archive(tmp_path, table), tmp_path / "out", "repeated letter")


def test_an_archive_converted_by_other_code_is_converted_again(tmp_path: Path) -> None:
    """A stored summary without the current commit, here one that records none, as earlier code
    wrote them, and a count that a reconversion replaces."""
    raw, out = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, 300)
    _convert(raw, out, rows)
    j_sum = json.loads((out / duolingo.REPORT).read_text("utf-8"))["train"]["j_sum"]
    stored = out / "train" / "train-part-1" / duolingo.SUMMARY
    summary = json.loads(stored.read_text("utf-8"))
    summary.pop("commit", None)
    stored.write_text(json.dumps(summary | {"j_sum": -1}), encoding="utf-8")
    _convert(raw, out, rows)
    report = json.loads((out / duolingo.REPORT).read_text("utf-8"))
    assert report["train"]["j_sum"] == j_sum
    assert report["train"]["archives"][0]["commit"] == report["train"]["commit"] == ""


def test_a_finished_archive_is_reused_only_at_its_commit(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "processed"
    rows = fixtures.duolingo_raw(raw, 300)
    duolingo.convert(raw, out, rows, ONE_EACH, commit="a")
    stored = out / "train" / "train-part-1" / duolingo.SUMMARY
    summary = json.loads(stored.read_text("utf-8"))
    stored.write_text(json.dumps(summary | {"j_sum": -1}), encoding="utf-8")
    duolingo.convert(raw, out, rows, ONE_EACH, commit="a")
    assert json.loads((out / duolingo.REPORT).read_text("utf-8"))["train"]["j_sum"] == -1
    duolingo.convert(raw, out, rows, ONE_EACH, commit="b")
    report = json.loads((out / duolingo.REPORT).read_text("utf-8"))
    assert report["train"]["j_sum"] == summary["j_sum"] and report["train"]["commit"] == "b"


def test_stops_on_a_row_count_other_than_the_readme(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    rows = fixtures.duolingo_raw(raw, 300)
    _convert(raw, out, rows)
    # A failed rerun leaves no report from the earlier run
    with pytest.raises(duolingo.ConversionError, match="the readme gives"):
        _convert(raw, out, rows | {"test": rows["test"] + 1})
    assert not (out / duolingo.REPORT).exists()


def test_skips_other_regular_files(tmp_path: Path) -> None:
    info = tarfile.TarInfo("logs/_SUCCESS")
    info.size = 0
    raw = _archive(tmp_path, _table(), extra=(info, b""))
    _convert(raw, tmp_path / "out", {"train": 100, "test": 50})
    report = json.loads((tmp_path / "out" / duolingo.REPORT).read_text("utf-8"))
    assert "logs/_SUCCESS" in report["train"]["archives"][0]["skipped"]


def test_stops_before_a_member_that_would_fill_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    fixtures.duolingo_raw(raw, 100)
    monkeypatch.setattr(duolingo, "FREE_MARGIN", 10**18)
    with pytest.raises(duolingo.ConversionError, match="must remain"):
        _convert(raw, tmp_path / "out", {"train": 200, "test": 200})
    assert not list((tmp_path / "out").glob("*/*/part-*.parquet"))
