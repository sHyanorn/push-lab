import json
from typing import Any

import numpy as np
import pytest

from push_lab.policy import compare, design, sweep
from push_lab.sim import inputs
from push_lab.sim.world import DAYS

from .conftest import CANDIDATES, make_world

USERS = 700
Ran = tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, dict[str, Any]]]]


def _context() -> sweep.Context:
    pop, static = make_world(seed=4, users=USERS)
    rng = np.random.default_rng(5)
    hourly = rng.poisson(1.0, (USERS, 24)).astype(np.float64)
    lists = {
        name: inputs.CandidateList(
            np.tile(np.arange(CANDIDATES), (USERS, 1)), static.score, pop.p0 * factor
        )
        for name, factor in (("cascade", 1.0), ("two_tower", 0.9))
    }
    inp = inputs.Inputs(
        user_ids=np.arange(USERS),
        visit_rate=pop.visit_rate,
        profile_sim=pop.profile,
        profile_policy=inputs.smoothed_profile(inputs.windows_from_hourly(hourly)),
        hourly_policy=hourly,
        visit_rate_policy=static.visit_rate,
        lists=lists,
    )
    return sweep.Context(inp, np.arange(200), np.arange(200, 600), np.arange(600, USERS))


@pytest.fixture(scope="module")
def ran() -> Ran:
    ctx = _context()
    chosen = [s for s in design.settings() if s["id"] in (0, 64, 128)]
    chosen = [s | {"values": s["values"] | {"base_rate": 0.2}} for s in chosen]
    return chosen, {s["id"]: sweep.run_setting(ctx, s) for s in chosen}


def test_run_setting_outputs(
    ran: Ran,
) -> None:
    _, out = ran
    metrics, info = out[0]
    names = info["policies"]
    assert names == list(sweep.PRIMARY + sweep.SECONDARY)
    assert metrics.shape == (len(names), len(sweep.METRICS), 200)
    col = {m: i for i, m in enumerate(sweep.METRICS)}
    delivered = {n: metrics[i, col["delivered"]] for i, n in enumerate(names)}
    assert delivered["N"].sum() == 0
    assert (delivered["T"] <= DAYS).all() and (delivered["F"] <= DAYS).all()
    for k in ("S", "G", "G2"):
        assert delivered[k].sum() == pytest.approx(delivered["S"].sum(), rel=0.02)
        assert delivered[k].sum() <= 100 * DAYS
        assert delivered[f"{k}_b1"].sum() > delivered[k].sum()
    assert set(info["diagnostics"]) >= {
        "spearman_S",
        "spearman_G",
        "spearman_G2",
        "best_slot_share",
    }
    assert all(v >= 0 for v in info["multipliers"].values())
    json.dumps(info)


def test_compare_summarizes_the_settings(
    ran: Ran,
) -> None:
    chosen, out = ran
    boot = compare.Bootstrap(200, samples=500, seed=0)
    results = {
        sid: compare.evaluate_setting(m.astype(np.float64), info["policies"], boot)
        for sid, (m, info) in out.items()
    }
    frozen = {"settings": chosen, "bootstrap": {"settings_seed": 1}, "names": list(design.NAMES)}
    summary = compare.summarize(results, frozen, {sid: info for sid, (_, info) in out.items()})
    assert set(summary["primary"]) == {"main", "annoyance_free"}
    entry = summary["primary"]["main"]["H1"]
    assert entry["wins"] + entry["losses"] + entry["neither"] == 1
    assert "label" in entry and "note" in summary["primary"]["main"]["H2"]
    assert set(summary["primary_holm"]) == {
        f"{s}.{h}" for s in ("main", "annoyance_free") for h in compare.PRIMARY
    }
    assert sum(summary["stress_verdicts"]["H1"].values()) == 1
    json.dumps(summary)
