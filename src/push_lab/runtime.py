"""Helpers shared by the command-line steps: result files, the code commit and threads."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def rounded(x: float) -> float:
    """Six decimals, the precision of the result files."""
    return round(float(x), 6)


def write_json(path: Path, data: Any, indent: int | None = 1) -> None:
    """UTF-8 JSON with a final newline and LF line endings on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8", newline="\n")


def one_thread_per_process() -> None:
    """Keep BLAS and OpenMP in worker processes to one thread, since workers fill the cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def code_commit(root: Path) -> str:
    """HEAD, refusing a working tree whose code differs from it."""

    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    if git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"):
        raise SystemExit("commit the code before running the sweep")
    return git("rev-parse", "HEAD")
