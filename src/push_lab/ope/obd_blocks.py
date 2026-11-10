"""Post hoc: the OBD check of METHODS 14.2 with ten-minute blocks as resampling units.

``python -m push_lab.ope.obd_blocks`` recomputes every interval of the full Men's campaign in
``results/ope_obd.json`` and writes ``results/ope_obd_blocks.json``. Served lists are reused
across impressions and clicks cluster within minutes, so impressions are not independent. A
resample draws 1,008 ten-minute UTC blocks with replacement, and a block carries the rows of
both logs that fall in it (Section 5 of the registration in github.com/sHyanorn/ope-checks,
whose random stream is used). The registered file and its verdicts are unchanged.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab.data import obd as data
from push_lab.data import paths
from push_lab.ope import estimators as est
from push_lab.ope import obd
from push_lab.runtime import code_commit, write_json

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

WEEK_START_US = 1_574_553_600_000_000  # 2019-11-24T00:00:00Z, the first day of the logs
BLOCK_US = 600_000_000
BLOCKS = 1008
SAMPLES, SEED = 10_000, 20261110


def blocks(log: pd.DataFrame) -> IntArray:
    """Ten-minute UTC block of each row, from 0 at the start of the logged week."""
    stamp = pd.to_datetime(log["timestamp"], format="ISO8601")
    t = stamp.dt.as_unit("us").astype("int64").to_numpy(dtype=np.int64)
    block: IntArray = ((t - WEEK_START_US) // BLOCK_US).astype(np.int64)
    if block.min() < 0 or block.max() >= BLOCKS:
        raise ValueError("rows outside the logged week")
    return block


def multiplicities(samples: int = SAMPLES) -> FloatArray:
    """How often each block is drawn [sample, block]."""
    p = np.full(BLOCKS, 1 / BLOCKS)
    draws = [np.random.default_rng([SEED, 0, i]).multinomial(BLOCKS, p) for i in range(samples)]
    return np.stack(draws).astype(np.float64)


def _by_block(logged: est.Logged) -> FloatArray:
    """``Logged.sums`` with a row for every block, empty ones included."""
    out = np.zeros((BLOCKS, 5))
    out[np.unique(logged.cluster)] = logged.sums()
    return out


def check(random: est.Logged, bts: est.Logged, draws: FloatArray) -> dict[str, Any]:
    """The check of ``obd.validate`` when both logs, clustered by block, are resampled with the
    same ``draws``. ``bts`` has unit weights, so its IPS is the on-policy click rate."""
    ips = est.ESTIMATORS.index("IPS")
    boot = est.from_sums(draws @ _by_block(random))
    truth_boot = est.from_sums(draws @ _by_block(bts))[:, ips]
    truth = est.point(bts)["IPS"]
    report: Any = est.summarize(random, boot)
    low, high = np.percentile(truth_boot, [2.5, 97.5])
    return {
        "on_policy": {"value": truth, "ci_low": float(low), "ci_high": float(high)},
        **report,
        "validation": obd.verdicts(report, boot, truth, truth_boot),
    }


def validate(
    random_log: pd.DataFrame,
    bts_log: pd.DataFrame,
    items: pd.DataFrame,
    dist: FloatArray,
    draws: FloatArray,
) -> dict[str, Any]:
    random = replace(obd.logged(random_log, items, dist), cluster=blocks(random_log))
    clicks = bts_log["click"].to_numpy(dtype=np.float64)
    ones, zeros = np.ones(len(clicks)), np.zeros(len(clicks))
    return check(random, est.Logged(clicks, ones, zeros, zeros, blocks(bts_log)), draws)


def _cells(log: pd.DataFrame, weights: FloatArray | None = None) -> FloatArray:
    """Sums over the rows of each [block, item, position]."""
    shape = (BLOCKS, len(obd.ALPHA), obd.POSITIONS)
    index = (
        blocks(log),
        log["item_id"].to_numpy(dtype=np.int64),
        log["position"].to_numpy(dtype=np.int64) - 1,
    )
    flat = np.ravel_multi_index(index, shape)
    out: FloatArray = np.bincount(flat, weights, int(np.prod(shape))).reshape(shape).astype(float)
    return out


def empirical_target(
    random_log: pd.DataFrame, bts_log: pd.DataFrame, prior: FloatArray, draws: FloatArray
) -> dict[str, Any]:
    """The post hoc item of ``obd.empirical_target`` over blocks: IPS with the BTS logs' own
    (item, position) frequencies as the target, as a ratio to the on-policy value, and that
    ratio minus the one with the target ``prior``. Unlike there, the frequencies and the
    on-policy value are recomputed in every resample, from the same blocks as the estimate."""
    shown = _cells(bts_log)
    bts_clicks = _cells(bts_log, bts_log["click"].to_numpy(dtype=np.float64)).sum(axis=(1, 2))
    click = random_log["click"].to_numpy(dtype=np.float64)
    weighted = _cells(random_log, click / random_log["propensity_score"].to_numpy())
    rows = _cells(random_log).sum(axis=(1, 2))

    def ratios(m: FloatArray) -> tuple[FloatArray, FloatArray]:
        bts = np.tensordot(m, shown, 1)
        target = bts / bts.sum(axis=1, keepdims=True)
        clicks = np.tensordot(m, weighted, 1)
        scale = (m @ rows) * (m @ bts_clicks) / bts.sum(axis=(1, 2))
        empirical: FloatArray = (target * clicks).sum(axis=(1, 2)) / scale
        fixed: FloatArray = (prior * clicks).sum(axis=(1, 2)) / scale
        return empirical, empirical - fixed

    def summary(value: FloatArray, boot: FloatArray) -> dict[str, float]:
        low, high = np.percentile(boot, [2.5, 97.5])
        return {"value": float(value[0]), "ci_low": float(low), "ci_high": float(high)}

    point, boot = ratios(np.ones((1, BLOCKS))), ratios(draws)
    return {
        "ratio_to_on_policy": summary(point[0], boot[0]),
        "difference_from_fixed_prior": summary(point[1], boot[1]),
    }


def run(root: Path) -> dict[str, Any]:
    folder = paths.processed_dir(root, "obd_full")
    random_log, bts_log = data.load(folder, "random"), data.load(folder, "bts")
    items = data.load(folder, "random", items=True)
    dist, draws = obd.bts_distribution(), multiplicities()
    return {
        "note": "Post hoc: results/ope_obd.json with ten-minute blocks as resampling units",
        "registered": "results/ope_obd.json holds the registered result and verdicts, unchanged",
        "resampling": {
            "unit": "ten-minute UTC blocks, the same blocks in both logs",
            "blocks": BLOCKS,
            "samples": SAMPLES,
            "stream": f"numpy.random.default_rng([{SEED}, 0, i]).multinomial for resample i",
        },
        "rule": f"interval contains the on-policy value and median relative error below "
        f"{obd.MAX_RELATIVE_ERROR}",
        "full": validate(random_log, bts_log, items, dist, draws)
        | {"post_hoc": empirical_target(random_log, bts_log, dist, draws)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The OBD check with blocks as resampling units.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    target = args.root / paths.RESULTS / "ope_obd_blocks.json"
    write_json(target, {"commit": code_commit(args.root)} | run(args.root))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
