"""Convert downloaded datasets to Parquet."""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from push_lab.data import kuairand, paths
from push_lab.data.sources import KUAIRAND_PURE


def _kuairand(root: Path) -> list[Path]:
    raw = paths.raw_dir(root, "kuairand")
    return kuairand.convert(
        raw / KUAIRAND_PURE.filename, raw / "extracted", paths.processed_dir(root, "kuairand")
    )


PREPARERS: dict[str, Callable[[Path], list[Path]]] = {
    "kuairand": _kuairand,
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
