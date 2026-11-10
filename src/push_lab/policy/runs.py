"""Per-setting run files of the two sweeps: where they live, how a sweep writes and resumes
them (D031, D033)."""

import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from push_lab.data import paths
from push_lab.runtime import write_json

FloatArray = npt.NDArray[np.float64]
# Per-user metrics [policy, metric, user] of one setting and its run info
type RunSetting[C] = Callable[[C, dict[str, Any]], tuple[FloatArray, dict[str, Any]]]

POLICY_RUNS = Path("runs") / "policy"
TEMPLATE_RUNS = Path("runs") / "templates"
# Per-user metrics a setting's archive holds, in order
POLICY_METRICS = ("active_days", "active_week4", "opens_week1", "opens", "delivered", "opted_out")
TEMPLATE_METRICS = ("opens", "active_days", "opens_week1", "delivered", "repeats")


def setting_paths(root: Path, setting_id: int, runs: Path) -> tuple[Path, Path]:
    base = root / paths.RESULTS / runs / f"setting_{setting_id:03d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def write_setting(
    root: Path, setting_id: int, values: FloatArray, info: dict[str, Any], runs: Path
) -> None:
    npz, meta = setting_paths(root, setting_id, runs)
    write_json(meta, info)
    # The archive marks a finished setting, so it is written last and moved into place whole
    partial = npz.with_name(npz.stem + ".partial.npz")
    np.savez_compressed(partial, metrics=values.astype(np.float32))
    os.replace(partial, npz)


def finished(root: Path, setting_id: int, commit: str, runs: Path) -> bool:
    """A setting is done when both files exist; done under another commit is an error."""
    npz, meta = setting_paths(root, setting_id, runs)
    if not (npz.exists() and meta.exists()):
        return False
    ran = json.loads(meta.read_text("utf-8")).get("commit")
    if ran != commit:
        raise SystemExit(f"setting {setting_id} ran with commit {ran}, not {commit}")
    return True


@dataclass(frozen=True)
class _Worker[C]:
    root: Path
    commit: str
    runs: Path
    context: C
    run_setting: RunSetting[C]


_WORKER: _Worker[Any] | None = None


def _init[C](
    root: Path,
    commit: str,
    runs: Path,
    load_context: Callable[[Path], C],
    run_setting: RunSetting[C],
) -> None:
    global _WORKER
    _WORKER = _Worker(root, commit, runs, load_context(root), run_setting)


def _work(s: dict[str, Any]) -> tuple[int, float]:
    w = _WORKER
    assert w is not None
    start = time.perf_counter()
    values, info = w.run_setting(w.context, s)
    write_setting(w.root, s["id"], values, info | {"commit": w.commit}, w.runs)
    return s["id"], time.perf_counter() - start


def run_pending[C](
    root: Path,
    frozen: dict[str, Any],
    commit: str,
    ids: list[int] | None,
    processes: int,
    runs: Path,
    load_context: Callable[[Path], C],
    run_setting: RunSetting[C],
    first: Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    """Run the selected settings not on disk yet, ``first`` ordering them, in worker processes
    that each load their context once. Every setting already on disk must come from
    ``commit``, selected or not."""
    done = {s["id"] for s in frozen["settings"] if finished(root, s["id"], commit, runs)}
    todo = [
        s for s in frozen["settings"] if (ids is None or s["id"] in ids) and s["id"] not in done
    ]
    if first is not None:
        todo.sort(key=first)
    print(f"{len(todo)} settings to run at {commit}", flush=True)
    start = time.perf_counter()
    initargs = (root, commit, runs, load_context, run_setting)
    with mp.get_context("spawn").Pool(processes, _init, initargs) as pool:
        for count, (sid, seconds) in enumerate(pool.imap_unordered(_work, todo), 1):
            elapsed = time.perf_counter() - start
            print(
                f"{count}/{len(todo)} setting {sid}: {seconds:.0f} s, {elapsed:.0f} s", flush=True
            )
