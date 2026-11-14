"""Duolingo notification logs (METHODS 15, D067), converted to Parquet.

Each tar.gz archive is read as a stream, one member at a time: a Parquet member is copied to a
temporary file under the raw folder, converted row group by row group to
``data/processed/duolingo/<split>/<archive>/`` and deleted. Every row keeps the columns METHODS 15
needs: the time, the language, the eligible set as a bit mask, the selected template, the
reward, each template's recency, j_t, the days since the previous notification and the history
length. An archive is converted into a partial folder and renamed when done, so a rerun skips
finished archives; ``conversion.json`` is written only after the row counts of both splits
match the readme, and the analysis reads nothing without it.
"""

import json
import os
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

IntArray = npt.NDArray[np.int64]

TEMPLATES = "ABCDEFGHIJKL"
LETTERS = pa.array(list(TEMPLATES))
ARCHIVES = {
    "train": ("train-part-1", "train-part-2", "train-part-3"),
    "test": ("test-part-1", "test-part-2", "test-part-3"),
}
# Row counts the dataset's readme gives
ROWS = {"train": 87_665_839, "test": 114_486_159}
HISTORY_ENTRY = pa.struct([pa.field("template", pa.string()), pa.field("n_days", pa.float32())])
# The six columns of the archives, as read from training part 1 (METHODS 15)
_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("datetime", pa.float64()),
    ("ui_language", pa.string()),
    ("eligible_templates", pa.list_(pa.field("element", pa.string()))),
    ("history", pa.list_(pa.field("element", HISTORY_ENTRY))),
    ("selected_template", pa.string()),
    ("session_end_completed", pa.bool_()),
]
SCHEMA = pa.schema(_COLUMNS)
RECENCY = tuple(f"recency_{t}" for t in TEMPLATES)
OUTPUT = pa.schema(
    [
        pa.field("datetime", pa.float64()),
        pa.field("ui_language", pa.dictionary(pa.int32(), pa.string())),
        # Bit i is set when template i (A = 0) is eligible
        pa.field("eligible", pa.uint16()),
        pa.field("selected", pa.int8()),
        pa.field("completed", pa.bool_()),
        # Smallest days-before of each template in the history, inf if absent
        *[pa.field(name, pa.float32()) for name in RECENCY],
        # History entries earlier on the same day, i.e. days-before below the time of day
        pa.field("j", pa.int32()),
        pa.field("since_previous", pa.float32()),
        pa.field("history_length", pa.int32()),
    ]
)
POPCOUNT = np.array([bin(i).count("1") for i in range(1 << len(TEMPLATES))])
# Free space kept after a member and its output (D067)
FREE_MARGIN = 5 * 10**9
CHUNK = 1 << 20
SUMMARY = "summary.json"
REPORT = "conversion.json"


class ConversionError(Exception):
    """Input that stops the conversion."""


def _letters(values: Any, what: str) -> IntArray:
    """Template numbers of letters; a null or a letter outside A to L stops the conversion."""
    index = pc.index_in(values, value_set=LETTERS)
    if index.null_count:
        raise ConversionError(f"{what}: {index.null_count} letters outside A to L or null")
    out: IntArray = np.asarray(index.to_numpy(zero_copy_only=False), dtype=np.int64)
    return out


def _no_nulls(column: pa.ChunkedArray[Any], what: str) -> None:
    if column.null_count:
        raise ConversionError(f"{what}: {column.null_count} nulls")


def convert_table(table: pa.Table) -> tuple[pa.Table, dict[str, Any]]:
    """One row group in the output schema, with the counts the report needs."""
    n = table.num_rows
    for name in ("datetime", "eligible_templates", "selected_template", "session_end_completed"):
        _no_nulls(table[name], name)
    when = table["datetime"].to_numpy()

    eligible = table["eligible_templates"].combine_chunks()
    sizes = pc.list_value_length(eligible).to_numpy(zero_copy_only=False)
    letters = _letters(pc.list_flatten(eligible), "eligible_templates")
    parents = pc.list_parent_indices(eligible).to_numpy()
    mask = np.bincount(parents, weights=1 << letters, minlength=n).astype(np.int64)
    if (POPCOUNT[mask] != sizes).any() or (sizes == 0).any():
        raise ConversionError("eligible_templates: a repeated letter or an empty set")
    selected = _letters(table["selected_template"], "selected_template")
    if not ((mask >> selected) & 1).all():
        raise ConversionError("selected_template: a template outside its eligible set")

    history = table["history"].combine_chunks()
    entries = cast(pa.StructArray, pc.list_flatten(history))
    if entries.null_count:
        raise ConversionError(f"history: {entries.null_count} null entries")
    days = entries.field("n_days")
    if days.null_count:
        raise ConversionError(f"history: {days.null_count} null n_days")
    template = _letters(entries.field("template"), "history template")
    days_np = days.to_numpy(zero_copy_only=False)
    owner = pc.list_parent_indices(history).to_numpy()
    flat = np.full(n * len(TEMPLATES), np.inf, dtype=np.float32)
    np.minimum.at(flat, owner * len(TEMPLATES) + template, days_np)
    recency = flat.reshape(n, len(TEMPLATES))
    time_of_day = when - np.floor(when)
    same_day = days_np.astype(np.float64) < time_of_day[owner]
    j = np.bincount(owner[same_day], minlength=n)
    length = pc.fill_null(pc.list_value_length(history), 0).to_numpy(zero_copy_only=False)

    columns: dict[str, Any] = {
        "datetime": when,
        "ui_language": pc.dictionary_encode(table["ui_language"].combine_chunks()),
        "eligible": mask.astype(np.uint16),
        "selected": selected.astype(np.int8),
        "completed": table["session_end_completed"].to_numpy(),
        **{name: recency[:, i] for i, name in enumerate(RECENCY)},
        "j": j.astype(np.int32),
        "since_previous": recency.min(axis=1),
        "history_length": length.astype(np.int32),
    }
    languages = pc.value_counts(table["ui_language"])
    counts = {
        "rows": n,
        "datetime_min": float(when.min()),
        "datetime_max": float(when.max()),
        "j_sum": int(j.sum()),
        "j_positive": int((j > 0).sum()),
        "one_eligible": int((sizes == 1).sum()),
        "null_language": table["ui_language"].null_count,
        "null_history": history.null_count,
        "history_entries": len(days_np),
        "history_length_max": int(length.max()) if n else 0,
        "n_days_min": float(days_np.min()) if len(days_np) else None,
        "n_days_max": float(days_np.max()) if len(days_np) else None,
        "n_days_outside_0_30": int(((days_np < 0) | (days_np > 30)).sum()),
        "languages": dict(
            zip(
                map(str, languages.field("values").to_pylist()),
                languages.field("counts").to_pylist(),
                strict=True,
            )
        ),
    }
    return pa.table(columns, schema=OUTPUT), counts


def combine(counts: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts of several row groups, members or archives as one."""
    out: dict[str, Any] = {}
    for c in counts:
        for key, value in c.items():
            if value is None:
                out.setdefault(key, None)
            elif key == "languages":
                merged = out.setdefault(key, {})
                for lang, k in value.items():
                    merged[lang] = merged.get(lang, 0) + k
            elif key.endswith("_min"):
                out[key] = value if out.get(key) is None else min(out[key], value)
            elif key.endswith("_max"):
                out[key] = value if out.get(key) is None else max(out[key], value)
            else:
                out[key] = out.get(key, 0) + value
    return out


def convert_member(source: Path, target: Path) -> dict[str, Any]:
    """A Parquet member, row group by row group, after checking its schema and row count."""
    # Closed before the caller deletes the member, which Windows refuses for an open file
    with pq.ParquetFile(source) as file:
        if not file.schema_arrow.equals(SCHEMA, check_metadata=False):
            raise ConversionError(f"unexpected schema:\n{file.schema_arrow}")
        counts = []
        with pq.ParquetWriter(target, OUTPUT) as writer:
            for i in range(file.num_row_groups):
                table, c = convert_table(file.read_row_group(i))
                writer.write_table(table, row_group_size=max(1, table.num_rows))
                counts.append(c)
        rows = file.metadata.num_rows
        out = combine(counts) | {"row_groups": file.num_row_groups}
    if out.get("rows", 0) != rows:
        raise ConversionError(f"{source.name}: read {out.get('rows')} of {rows}")
    return out


def member_kind(member: tarfile.TarInfo, folder: Path) -> str:
    """ "parquet" for a regular Parquet file, "skip" for a folder or any other regular file
    (macOS and checksum side files, Spark's ``_SUCCESS``); a path that would leave ``folder``,
    or a link or other special member, stops the conversion. Skipped members are reported, and
    the row counts of D067 catch any data left in them."""
    path = PurePosixPath(member.name)
    inside = (folder / member.name).resolve().is_relative_to(folder.resolve())
    if path.is_absolute() or ".." in path.parts or not inside:
        raise ConversionError(f"member path leaves the target folder: {member.name!r}")
    if member.isfile():
        return "parquet" if path.name.endswith(".parquet") else "skip"
    if member.isdir():
        return "skip"
    raise ConversionError(f"unexpected member {member.name!r}")


def check_disk(folder: Path, size: int) -> None:
    """Stop if a member of ``size`` bytes and an output of at most that size would leave less
    than ``FREE_MARGIN`` free."""
    free = shutil.disk_usage(folder).free
    if free - 2 * size < FREE_MARGIN:
        raise ConversionError(
            f"stopped before a member of {size:,} bytes: {free:,} bytes free, "
            f"{FREE_MARGIN:,} must remain"
        )


def convert_archive(archive: Path, split: str, out_dir: Path, scratch: Path) -> dict[str, Any]:
    """One archive into ``out_dir/split/<name>/``; returns its summary, read back if done."""
    name = archive.name.removesuffix(".tar.gz")
    done = out_dir / split / name
    if (done / SUMMARY).exists():
        summary: dict[str, Any] = json.loads((done / SUMMARY).read_text("utf-8"))
        return summary
    shutil.rmtree(done, ignore_errors=True)
    partial = out_dir / ".partial" / name
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    scratch.mkdir(parents=True, exist_ok=True)
    members: list[dict[str, Any]] = []
    skipped: list[str] = []
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if member_kind(member, scratch) == "skip":
                skipped.append(member.name)
                continue
            check_disk(scratch, member.size)
            temp = scratch / f"{name}.member.parquet"
            source = tar.extractfile(member)
            assert source is not None
            with temp.open("wb") as f:
                shutil.copyfileobj(source, f, CHUNK)
            try:
                c = convert_member(temp, partial / f"part-{len(members):03d}.parquet")
            finally:
                temp.unlink()
            members.append({"name": member.name, "bytes": member.size} | c)
    if not members:
        raise ConversionError(f"{archive.name}: no Parquet member")
    summary = {"archive": archive.name, "split": split, "members": members, "skipped": skipped}
    summary |= combine(
        [{k: v for k, v in m.items() if k not in ("name", "bytes")} for m in members]
    )
    (partial / SUMMARY).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    done.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, done)
    return summary


def convert(
    raw_dir: Path,
    out_dir: Path,
    rows: Mapping[str, int] = ROWS,
    archives: Mapping[str, Sequence[str]] = ARCHIVES,
) -> list[Path]:
    """Every archive, then the row counts of each split against the readme; the report is
    written only if they match."""
    # A report left by an earlier run would vouch for files this run may not finish
    (out_dir / REPORT).unlink(missing_ok=True)
    report: dict[str, Any] = {}
    for split, names in archives.items():
        summaries = [
            convert_archive(raw_dir / f"{n}.tar.gz", split, out_dir, raw_dir / "tmp") for n in names
        ]
        total = combine(
            [
                {k: s[k] for k in s if k not in ("archive", "split", "members", "skipped")}
                for s in summaries
            ]
        )
        if total["rows"] != rows[split]:
            raise ConversionError(
                f"{split}: {total['rows']:,} rows, the readme gives {rows[split]:,}"
            )
        report[split] = total | {"archives": summaries}
    for split in report:
        r = report[split]
        r["j_mean"] = r["j_sum"] / r["rows"]
        r["j_positive_share"] = r["j_positive"] / r["rows"]
        r["one_eligible_share"] = r["one_eligible"] / r["rows"]
    target = out_dir / REPORT
    target.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return [target, *sorted(out_dir.glob("*/*/part-*.parquet"))]


def files(processed: Path, split: str) -> list[Path]:
    """The converted files of ``split`` in a fixed order; refuses an incomplete conversion."""
    if not (processed / REPORT).exists():
        raise ConversionError(f"no {REPORT}: the conversion did not finish")
    return sorted((processed / split).glob("*/part-*.parquet"))
