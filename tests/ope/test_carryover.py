import json
from typing import Any

import numpy as np
import pytest

from push_lab.ope import carryover
from push_lab.ope import simulator as sim
from push_lab.sim.world import Params


def test_grid_has_24_distinct_settings_around_the_reference() -> None:
    grid = carryover.settings()
    assert len(grid) == 24
    reference = [s for s in grid if s["values"] == Params().__dict__]
    assert len(reference) == 1
    assert sorted(g[0] for g in reference[0]["grid"]) == ["annoyance", "fatigue_elasticity"]
    for s in grid:
        Params(**s["values"])
    assert not set(carryover.LOG_SEEDS) & set(sim.LOG_SEEDS)


def test_bias_resamples_center_on_the_bias() -> None:
    rng = np.random.default_rng(0)
    boot = carryover.bias_resamples(0.6 + 0.01 * rng.standard_normal(10), np.full(100, 0.5), 1)
    assert boot.shape == (carryover.RESAMPLES,)
    assert abs(boot.mean() - 0.1) < 0.01


def _fake(bias_of: dict[str, float]) -> tuple[list[dict[str, Any]], dict[int, Any], dict[int, Any]]:
    """Every setting's estimates sit at truth + slope * annoyance for the focus targets."""
    grid = carryover.settings()
    rng = np.random.default_rng(0)
    logs: dict[int, Any] = {}
    truths: dict[int, Any] = {}
    for s in grid:
        a = s["values"]["annoyance"]
        truths[s["id"]] = {
            t: {"visit": list(0.5 + 0.001 * rng.standard_normal(100))} for t in sim.TARGETS
        }
        logs[s["id"]] = {}
        for seed in carryover.LOG_SEEDS:
            entry: dict[str, Any] = {}
            for t in sim.TARGETS:
                v = 0.5 + bias_of.get(t, 0.0) * (a - 0.06) + 0.001 * rng.standard_normal()
                e = {"value": v, "ci_low": v - 0.01, "ci_high": v + 0.01}
                entry[t] = {"visit": {"estimates": {"IPS": e, "DR": e}}}
            logs[s["id"]][seed] = entry
    return grid, logs, truths


def test_summarize_finds_a_flip_along_annoyance() -> None:
    grid, logs, truths = _fake({"T2": 1.0, "N": -1.0})
    out = carryover.summarize(grid, logs, truths)
    assert len(out["settings"]) == 24
    t2 = out["flips"]["annoyance"]["T2.DR"]
    assert t2["result"] == "flip"
    assert abs(t2["crossings"][0]["at"] - 0.06) < 0.005
    assert out["flips"]["opt_out"]["T2.DR"]["result"] != "flip"


def test_orderings_flag_a_reversed_difference(tmp_path: Any) -> None:
    truth = {"T": 0.50, "N": 0.52, "T_half": 0.51, "T2": 0.49}
    settings = []
    for i, a in enumerate((0.0, 0.1)):
        settings.append(
            {
                "values": {"annoyance": a},
                "grid": [["annoyance", i]],
                "targets": {t: {"truth": {"value": v, "se": 0.001}} for t, v in truth.items()},
            }
        )
        for k, seed in enumerate((1, 2, 3)):
            # The estimate of N - T is negative at the second point, against a positive truth
            est = dict(truth) | {"N": 0.52 - 0.05 * i + 0.001 * k}
            log = {
                t: {"visit": {"estimates": {"DR": {"value": v}, "IPS": {"value": v}}}}
                for t, v in est.items()
            }
            (tmp_path / f"{i}_{seed}.json").write_text(json.dumps(log), "utf-8")
    out = carryover.orderings({"settings": settings, "log_seeds": [1, 2, 3]}, tmp_path)
    pair = out["orderings"]["annoyance"]["N-T"]
    assert pair["reversed"]["DR"] == [False, True]
    assert pair["truth"][0]["value"] == pytest.approx(0.02)
    assert out["orderings"]["opt_out"]["values"] == []
