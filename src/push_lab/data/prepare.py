"""Convert downloaded datasets to Parquet."""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from push_lab.data import kuairand, obd, paths, uplift_trials
from push_lab.data.sources import CRITEO_UPLIFT, HILLSTROM, KUAIRAND_PURE


def _kuairand(root: Path) -> list[Path]:
    raw = paths.raw_dir(root, "kuairand")
    return kuairand.convert(
        raw / KUAIRAND_PURE.filename, raw / "extracted", paths.processed_dir(root, "kuairand")
    )


def _hillstrom(root: Path) -> list[Path]:
    csv = paths.raw_dir(root, "hillstrom") / HILLSTROM.filename
    out = paths.processed_dir(root, "hillstrom") / "hillstrom.parquet"
    return [uplift_trials.convert_hillstrom(csv, out)]


def _criteo(root: Path) -> list[Path]:
    csv_gz = paths.raw_dir(root, "criteo") / CRITEO_UPLIFT.filename
    out = paths.processed_dir(root, "criteo") / "criteo.parquet"
    return [uplift_trials.convert_criteo(csv_gz, out)]


def _obd(root: Path) -> list[Path]:
    return obd.convert(paths.raw_dir(root, "obd"), paths.processed_dir(root, "obd"))


PREPARERS: dict[str, Callable[[Path], list[Path]]] = {
    "kuairand": _kuairand,
    "hillstrom": _hillstrom,
    "criteo": _criteo,
    "obd": _obd,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert downloaded datasets to Parquet.")
    parser.add_argument("--only", nargs="+", choices=sorted(PREPARERS), help="datasets to convert")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    for dataset in args.only or sorted(PREPARERS):
        for path in PREPARERS[dataset](args.root):
            print(f"{dataset}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
