import io
import tarfile
from pathlib import Path

import pytest

from push_lab.data import kuairand
from push_lab.data.schemas import Schema


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return f'"{text}"' if "," in text else text


def _csv(schema: Schema, rows: list[list[object]]) -> bytes:
    lines = [",".join(schema)] + [",".join(_cell(v) for v in r) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def _row(schema: Schema, **values: object) -> list[object]:
    return [values.get(name, 0) for name in schema]


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _archive(path: Path) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, (filename, schema) in kuairand.TABLES.items():
            if name in kuairand.LOGS:
                rows = [_row(schema, user_id=1, tab=1, is_click=1), _row(schema, user_id=2, tab=0)]
            else:
                rows = [_row(schema, user_active_degree="high_active", tag="1,2", upload_dt="x")]
            _add(tar, f"KuaiRand-Pure/data/{filename}", _csv(schema, rows))
    return path


def test_convert_and_load_filters_tab(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "pure.tar.gz")
    out = tmp_path / "processed"
    written = kuairand.convert(archive, tmp_path / "work", out)
    assert {p.stem for p in written} == set(kuairand.TABLES)

    log = kuairand.load(out, "log_random")
    assert log["tab"].tolist() == [1]
    assert len(kuairand.load(out, "log_random", tab=None)) == 2
    assert list(kuairand.load(out, "users").columns) == list(kuairand.TABLES["users"][1])


def test_convert_rejects_unexpected_columns(tmp_path: Path) -> None:
    with tarfile.open(tmp_path / "bad.tar.gz", "w:gz") as tar:
        for name, (filename, schema) in kuairand.TABLES.items():
            header = [*schema, "extra"] if name == "users" else list(schema)
            _add(tar, f"KuaiRand-Pure/data/{filename}", (",".join(header) + "\n").encode())
    with pytest.raises(ValueError, match="do not match"):
        kuairand.convert(tmp_path / "bad.tar.gz", tmp_path / "work", tmp_path / "out")


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add(tar, "../evil.txt", b"x")
    with pytest.raises(tarfile.FilterError):
        kuairand.extract(archive, tmp_path / "work")
    assert not (tmp_path / "evil.txt").exists()
