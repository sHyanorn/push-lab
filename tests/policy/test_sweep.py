import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from push_lab.policy import compare, design, runs, sweep
from push_lab.sim import env, inputs
from push_lab.sim.world import DAYS

from .conftest import CANDIDATES, make_world

USERS = 700
# The settings run, their metrics and info by id, and the users each setting simulated, in order
Ran = tuple[
    list[dict[str, Any]], dict[int, tuple[np.ndarray, dict[str, Any]]], dict[int, list[np.ndarray]]
]


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
        creator_clicks=np.zeros((USERS, CANDIDATES), dtype=np.int64),
        trend_rank=np.zeros((USERS, CANDIDATES), dtype=np.int64),
        lists=lists,
    )
    return sweep.Context(inp, np.arange(200), np.arange(200, 600), np.arange(600, USERS))


@pytest.fixture(scope="module")
def ran() -> Ran:
    ctx = _context()
    chosen = [s for s in design.settings() if s["id"] in (0, 64, 128)]
    chosen = [s | {"values": s["values"] | {"base_rate": 0.2}} for s in chosen]
    seen: list[np.ndarray] = []
    population = sweep.Context.population

    def spy(self: sweep.Context, users: np.ndarray, name: str = "cascade") -> env.Population:
        seen.append(np.asarray(users).copy())
        return population(self, users, name)

    out, simulated = {}, {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sweep.Context, "population", spy)
        for s in chosen:
            seen.clear()
            out[s["id"]] = sweep.run_setting(ctx, s)
            simulated[s["id"]] = list(seen)
    return chosen, out, simulated


def test_run_setting_outputs(
    ran: Ran,
) -> None:
    _, out, _ = ran
    metrics, info = out[0]
    names = info["policies"]
    assert names == list(sweep.PRIMARY + sweep.SECONDARY)
    assert metrics.shape == (len(names), len(runs.POLICY_METRICS), 200)
    col = {m: i for i, m in enumerate(runs.POLICY_METRICS)}
    delivered = {n: metrics[i, col["delivered"]] for i, n in enumerate(names)}
    assert delivered["N"].sum() == 0
    assert (delivered["T"] <= DAYS).all() and (delivered["F"] <= DAYS).all()
    for k in ("S", "G", "G2"):
        assert delivered[k].sum() == pytest.approx(delivered["S"].sum(), rel=0.02)
        assert delivered[k].sum() <= 100 * DAYS
        assert (metrics[names.index(f"{k}_lambda"), col["delivered"]] <= DAYS).all()
        assert delivered[f"{k}_lambda"].sum() <= delivered[k].sum() + 1e-9
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
    chosen, out, _ = ran
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
    assert sum(summary["stress_verdicts"]["counts"]["H1"].values()) == 1
    json.dumps(summary)


def test_a_partial_run_gets_no_labels(ran: Ran) -> None:
    chosen, out, _ = ran
    boot = compare.Bootstrap(200, samples=200, seed=0)
    results = {
        sid: compare.evaluate_setting(m.astype(np.float64), info["policies"], boot)
        for sid, (m, info) in out.items()
    }
    missing = chosen[0] | {"id": 999, "point": 1}
    frozen = {
        "settings": [*chosen, missing],
        "bootstrap": {"settings_seed": 1},
        "names": list(design.NAMES),
    }
    summary = compare.summarize(results, frozen, {sid: info for sid, (_, info) in out.items()})
    assert summary["primary"]["main"]["H1"]["label"] == "incomplete"
    assert summary["primary_holm"] == {}


def test_resume_refuses_settings_from_another_commit(tmp_path: Path) -> None:
    npz, meta = runs.setting_paths(tmp_path, 3, runs.POLICY_RUNS)
    npz.parent.mkdir(parents=True)
    assert not runs.finished(tmp_path, 3, "abc", runs.POLICY_RUNS)
    np.savez(npz, metrics=np.zeros(1))
    assert not runs.finished(tmp_path, 3, "abc", runs.POLICY_RUNS)
    meta.write_text(json.dumps({"commit": "abc"}))
    assert runs.finished(tmp_path, 3, "abc", runs.POLICY_RUNS)
    with pytest.raises(SystemExit, match="commit"):
        runs.finished(tmp_path, 3, "def", runs.POLICY_RUNS)


def test_policies_run_on_the_evaluation_users(ran: Ran) -> None:
    ctx = _context()
    _, out, simulated = ran
    groups = {"fit": ctx.fit, "diagnostic": ctx.diagnostic, "evaluation": ctx.evaluation}
    for sid, (metrics, _) in out.items():
        used = [next(k for k, g in groups.items() if np.array_equal(u, g)) for u in simulated[sid]]
        assert sorted(used) == ["diagnostic", "evaluation", "evaluation", "fit"]
        assert metrics.shape[2] == len(ctx.evaluation)
