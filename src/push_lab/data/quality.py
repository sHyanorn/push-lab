"""Data quality report over the processed tables (thresholds in D016).

Errors make the command exit with status 1; the report is written either way.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from push_lab.data import fixtures, paths, schemas
from push_lab.data.sources import SOURCES

# Published figures for the real data (D016)
PUBLISHED: dict[str, float] = {
    "kuairand.log_standard_early.rows": 1_141_112,
    "kuairand.log_standard_late.rows": 295_497,
    "kuairand.log_random.rows": 1_186_059,
    "kuairand.users.rows": 27_285,
    "criteo.criteo.rows": 13_979_592,
    "criteo.treatment_share": 0.85,
}
SHARE_TOLERANCE = 0.005
TRAIN_SHARE = 0.1
TRAIN_TOLERANCE = 0.01
ARM_SHARE_RANGE = (0.45, 0.55)


# Columns the checks need; other tables are small enough to load whole
COLUMNS: dict[tuple[str, str], list[str]] = {
    ("criteo", "criteo"): ["treatment", "visit", "conversion", "train"],
}


@dataclass
class Report:
    tables: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.errors.append(message)


def _round(x: float) -> float:
    return round(float(x), 6)


def _table(
    report: Report, root: Path, key: tuple[str, str], published: bool
) -> pd.DataFrame | None:
    dataset, name = key
    path = paths.processed_dir(root, dataset) / f"{name}.parquet"
    label = f"{dataset}.{name}"
    if not path.exists():
        report.errors.append(f"{label}: missing {path}")
        return None
    expected = schemas.to_arrow(fixtures.SCHEMAS[key])
    report.check(
        pq.read_schema(path).remove_metadata().equals(expected),
        f"{label}: Parquet schema differs from the shared schema",
    )
    meta = pq.ParquetFile(path).metadata
    nulls = dict.fromkeys(expected.names, 0)
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        for c in range(group.num_columns):
            column = group.column(c)
            stats = column.statistics
            if stats is None or stats.null_count is None:
                report.errors.append(f"{label}: no null-count statistics in {path.name}")
                return None
            nulls[column.path_in_schema] += stats.null_count
    n_rows = meta.num_rows
    report.tables[label] = {
        "rows": n_rows,
        "missing_share": {c: _round(v / n_rows) for c, v in nulls.items() if v and n_rows},
    }
    df = pd.read_parquet(path, columns=COLUMNS.get(key))
    rows = PUBLISHED.get(f"{label}.rows")
    if published and rows is not None:
        report.check(n_rows == rows, f"{label}: {n_rows} rows, published {rows:.0f}")
    return df


def _kuairand(report: Report, logs: dict[str, pd.DataFrame]) -> None:
    summary: dict[str, Any] = {}
    for name, log in logs.items():
        summary[name] = {
            "click_rate_by_tab": {
                str(tab): _round(rate)
                for tab, rate in log.groupby("tab")["is_click"].mean().items()
            },
            "tab1_share": _round((log["tab"] == 1).mean()),
        }
        want_rand = int(name == "log_random")
        report.check(
            bool((log["is_rand"] == want_rand).all()),
            f"kuairand.{name}: is_rand is not {want_rand} on every row",
        )
    total = sum(len(log) for log in logs.values())
    summary["random_rows"] = len(logs["log_random"])
    summary["random_share_of_all_logs"] = _round(len(logs["log_random"]) / total)
    report.datasets["kuairand"] = summary


def _hillstrom(report: Report, df: pd.DataFrame) -> None:
    arms = df.groupby("treatment")
    share = _round(df["treatment"].mean())
    report.datasets["hillstrom"] = {
        "rows_by_treatment": {str(k): int(v) for k, v in arms.size().items()},
        "treatment_share": share,
        "visit_rate_by_treatment": {str(k): _round(v) for k, v in arms["visit"].mean().items()},
        "conversion_rate_by_treatment": {
            str(k): _round(v) for k, v in arms["conversion"].mean().items()
        },
    }
    low, high = ARM_SHARE_RANGE
    report.check(low <= share <= high, f"hillstrom: treatment share {share} outside {low}-{high}")


def _criteo(report: Report, df: pd.DataFrame, published: bool) -> None:
    share = _round(df["treatment"].mean())
    train = _round(df["train"].mean())
    report.datasets["criteo"] = {
        "treatment_share": share,
        "visit_rate": _round(df["visit"].mean()),
        "conversion_rate": _round(df["conversion"].mean()),
        "train_share": train,
        "train_rows": int(df["train"].sum()),
    }
    if published:
        expected = PUBLISHED["criteo.treatment_share"]
        report.check(
            abs(share - expected) <= SHARE_TOLERANCE,
            f"criteo: treatment share {share}, published {expected}",
        )
    report.check(
        abs(train - TRAIN_SHARE) <= TRAIN_TOLERANCE,
        f"criteo: train share {train} not within {TRAIN_TOLERANCE} of {TRAIN_SHARE}",
    )


def _obd(report: Report, logs: dict[str, pd.DataFrame]) -> None:
    summary: dict[str, Any] = {}
    for policy, log in logs.items():
        p = log["propensity_score"]
        summary[policy] = {
            "rows": len(log),
            "click_rate": _round(log["click"].mean()),
            "propensity_min": _round(p.min()),
            "propensity_max": _round(p.max()),
        }
        report.check(bool(((p > 0) & (p <= 1)).all()), f"obd.{policy}: propensity outside (0, 1]")
    if "random" in logs:
        uniform = 1 / schemas.OBD_MEN_ACTIONS
        report.check(
            bool((logs["random"]["propensity_score"] - uniform).abs().max() < 1e-9),
            "obd.random: propensity is not uniform over the Men's actions",
        )
    report.datasets["obd"] = summary


def build(root: Path, *, published: bool = True) -> Report:
    """Check every processed table under ``root``.

    ``published=False`` skips comparisons with published figures, for synthetic data.
    """
    report = Report()
    frames = {key: _table(report, root, key, published) for key in fixtures.SCHEMAS}

    logs = {
        name: df
        for (dataset, name), df in frames.items()
        if dataset == "kuairand" and name.startswith("log_") and df is not None
    }
    if len(logs) == 3:
        _kuairand(report, logs)
    hill = frames["hillstrom", "hillstrom"]
    if hill is not None:
        _hillstrom(report, hill)
    crit = frames["criteo", "criteo"]
    if crit is not None:
        _criteo(report, crit, published)
    obd_logs = {
        name.removesuffix("_men"): df
        for (dataset, name), df in frames.items()
        if dataset == "obd" and not name.endswith("_items") and df is not None
    }
    _obd(report, obd_logs)
    return report


def to_json(report: Report) -> str:
    sources = {
        s.filename: f"{s.algorithm}:{s.checksum}" for group in SOURCES.values() for s in group
    }
    payload = {
        "sources": sources,
        "tables": report.tables,
        "datasets": report.datasets,
        "errors": report.errors,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the data quality report.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    report = build(args.root)
    out = args.root / paths.RESULTS / "data_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(report), encoding="utf-8", newline="\n")
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)
    print(f"wrote {out} ({len(report.errors)} errors)")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
