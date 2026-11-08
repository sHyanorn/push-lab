"""Rerun the demo notebook and compare its outputs with the committed ones (D051).

``python -m push_lab.report.notebook`` checks; ``--update`` rewrites the stored outputs.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path("notebooks") / "demo.ipynb"
KEPT = ("text/plain", "text/html", "text/markdown", "image/svg+xml")


def execute(path: Path) -> Any:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    ).execute()
    return nb


def outputs(nb: Any) -> list[list[dict[str, Any]]]:
    """Each code cell's outputs without execution counts or metadata."""
    cells = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        kept = []
        for out in cell.outputs:
            if out.output_type == "stream":
                kept.append({"stream": out.name, "text": out.text})
            elif out.output_type in ("display_data", "execute_result"):
                kept.append({k: out.data[k] for k in KEPT if k in out.data})
            else:
                kept.append({"error": out.get("ename", out.output_type)})
        cells.append(kept)
    return cells


def strip_counts(nb: Any) -> None:
    """Keep the stored notebook free of run-specific fields."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.metadata = {}
            for out in cell.outputs:
                if "execution_count" in out:
                    out.execution_count = None
                if "metadata" in out:
                    out.metadata = {}
    nb.metadata.pop("widgets", None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or update the demo notebook's outputs.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--update", action="store_true", help="rewrite the stored outputs")
    args = parser.parse_args(argv)
    path = args.root / NOTEBOOK
    fresh = execute(path)
    if args.update:
        strip_counts(fresh)
        nbformat.write(fresh, path)
        print(f"wrote {path}")
        return 0
    stored = nbformat.read(path, as_version=4)
    if outputs(fresh) != outputs(stored):
        print(f"{path} outputs differ from a fresh run; run with --update and review the diff")
        return 1
    print(f"{path} matches a fresh run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
