"""Helpers shared by the command-line steps: result files, the code commit, threads and stop
lines."""

import json
import multiprocessing as mp
import os
import subprocess
import time
from collections.abc import Iterator
from multiprocessing.pool import IMapIterator
from pathlib import Path
from typing import Any


def rounded(x: float) -> float:
    """Six decimals, the precision of the result files that round; the off-policy files keep
    most numbers at full precision."""
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
        raise SystemExit("commit the code before running")
    return git("rev-parse", "HEAD")


class RunStopped(Exception):
    """A run passed its stop line (D066, D067, D074)."""


class Deadline:
    """A stop line in seconds of wall-clock time, of which ``spent`` went to earlier steps."""

    def __init__(self, line: float, spent: float = 0.0) -> None:
        self.line, self.spent = line, spent
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return self.spent + time.perf_counter() - self.start

    def _stopped(self) -> RunStopped:
        return RunStopped(
            f"stopped after {self.elapsed():,.0f} s at the stop line of {self.line:,.0f} s"
        )

    def check(self) -> None:
        if self.elapsed() >= self.line:
            raise self._stopped()

    def results[T](self, items: IMapIterator[T]) -> Iterator[T]:
        """The results of a pool's ``imap``, waiting only until the line, so that a run stops
        while a worker is still busy."""
        while True:
            self.check()
            try:
                yield items.next(timeout=self.line - self.elapsed())
            except StopIteration:
                return
            except mp.TimeoutError:
                raise self._stopped() from None
