import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from push_lab.data import paths
from push_lab.policy import design, sweep, template_compare, template_design, template_sweep
from push_lab.policy.templates import KINDS
from push_lab.sim import env, inputs
from push_lab.sim.world import DAYS, SLOTS

from .conftest import CANDIDATES, make_world

ROOT = Path(__file__).parents[2]
USERS = 300
FROZEN = template_design.build(ROOT, USERS)


def test_design_reuses_the_policy_design() -> None:
    counts = {
        sw: sum(s["sweep"] == sw for s in FROZEN["settings"])
        for sw in ("main", "grid", "replicate")
    }
    assert counts == {"main": 64, "grid": 20, "replicate": 10}
    source = {s["id"]: s for s in design.load(ROOT)["settings"]}
    for s in FROZEN["settings"]:
        origin = source[s["policy_setting"]]
        assert s["values"] == origin["values"]
        assert s["world_seed"] == origin["world_seed"]
        assert s["perturb_seed"] == origin["perturb_seed"]
        assert all(g[0] in template_design.GRID_NAMES for g in s["grid"])
    seeds = [r for s in FROZEN["settings"] if s["sweep"] == "replicate" for r in s["eval_seeds"]]
    assert len(set(seeds)) == 30 and not set(seeds) & {101, 102, 103}
    reference = [s for s in FROZEN["settings"] if s["values"] == FROZEN["reference"]]
    assert len(reference) == 11  # the grid's reference and its ten replicates
    assert [s["id"] for s in FROZEN["settings"]] == list(range(94))


def test_repeats_count_templates_seen_within_the_window() -> None:
    n = 2
    delivered = np.zeros((DAYS, SLOTS, n), dtype=bool)
    template = np.full((DAYS, SLOTS, n), -1, dtype=np.int16)
    for d, k in ((0, 3), (10, 3), (26, 3)):
        delivered[d, 0, 0], template[d, 0, 0] = True, k
    delivered[0, 0, 1], template[0, 0, 1] = True, 1
    delivered[1, 0, 1], template[1, 0, 1] = True, 2
    out = env.Outcome(
        send=delivered,
        delivered=delivered,
        opened=np.zeros_like(delivered),
        template=template,
        item=template,
        propensity=np.ones(delivered.shape, dtype=np.float32),
        opted_out_at=np.full(n, -1),
        visited=np.zeros((DAYS, n), dtype=bool),
    )
    # Days 0 -> 10 is a repeat within 15 days; 10 -> 26 is not
    np.testing.assert_array_equal(template_sweep.repeats(out, 5), [1, 0])


def test_score_diagnostics() -> None:
    copy = np.exp(np.linspace(-1, 1, 30))
    perfect = template_sweep.score_diagnostics(np.log(copy), copy)
    assert perfect == {"spearman": pytest.approx(1.0), "top_overlap": 1.0}
    assert template_sweep.score_diagnostics(np.zeros(30), np.ones(30))["spearman"] is None
    assert template_sweep.score_diagnostics(np.zeros(30), copy)["top_overlap"] is None


def _context() -> template_sweep.Context:
    pop, static = make_world(seed=4, users=USERS)
    rng = np.random.default_rng(5)
    hourly = rng.poisson(1.0, (USERS, 24)).astype(np.float64)
    shape = (USERS, CANDIDATES)
    lists = {
        name: inputs.CandidateList(np.tile(np.arange(CANDIDATES), (USERS, 1)), static.score, pop.p0)
        for name in inputs.LISTS
    }
    inp = inputs.Inputs(
        user_ids=np.arange(USERS),
        visit_rate=pop.visit_rate,
        profile_sim=pop.profile,
        profile_policy=inputs.smoothed_profile(inputs.windows_from_hourly(hourly)),
        hourly_policy=hourly,
        visit_rate_policy=static.visit_rate,
        creator_clicks=rng.integers(0, 2, shape),
        trend_rank=np.where(rng.random(shape) < 0.3, 5, 0),
        lists=lists,
    )
    return template_sweep.Context(inp, np.arange(60) % 6)


@pytest.fixture(scope="module")
def ran() -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, dict[str, Any]]]]:
    ctx = _context()
    chosen = [s for s in FROZEN["settings"] if s["id"] in (0, 64, 84)]
    chosen = [s | {"values": s["values"] | {"base_rate": 0.2}} for s in chosen]
    return chosen, {s["id"]: template_sweep.run_setting(ctx, s) for s in chosen}


def test_run_setting_outputs(ran: Any) -> None:
    chosen, out = ran
    for s in chosen:
        metrics, info = out[s["id"]]
        assert metrics.shape == (len(KINDS), len(template_sweep.METRICS), USERS)
        delivered = metrics[:, template_sweep.METRICS.index("delivered")]
        for row in delivered[1:]:
            np.testing.assert_array_equal(row, delivered[0])
        repeats = metrics[:, template_sweep.METRICS.index("repeats")]
        assert (repeats <= delivered).all()
        assert info["kinds"] == list(KINDS)
        assert sum(info["uniform_family_share"]) == pytest.approx(1, abs=1e-5)
        assert set(info["score_diagnostics"]) == {"RD", "D0", "RAW"}


def test_run_setting_is_deterministic(ran: Any) -> None:
    chosen, out = ran
    again, _ = template_sweep.run_setting(_context(), chosen[0])
    np.testing.assert_array_equal(again, out[chosen[0]["id"]][0])


def _fake_root(tmp_path: Path, effect: float, ids: list[int] | None = None) -> Path:
    """A design over 200 users with 200 resamples and synthetic runs where RD has ``effect``
    more opens than U and D0, plus noise."""
    frozen = json.loads(json.dumps(FROZEN))
    frozen["users"] = 200
    frozen["bootstrap"] = {"samples": 200, "seed": 0, "settings_seed": 1}
    results = tmp_path / paths.RESULTS
    results.mkdir(parents=True)
    (results / template_design.DESIGN_FILE).write_text(json.dumps(frozen))
    rng = np.random.default_rng(0)
    for s in frozen["settings"]:
        if ids is not None and s["id"] not in ids:
            continue
        m = np.zeros((len(KINDS), len(template_sweep.METRICS), 200))
        m[:, 0] = rng.poisson(1.0, 200)
        m[:, 3] = 28
        m[KINDS.index("RD"), 0] += effect + 0.05 * rng.standard_normal(200)
        m[KINDS.index("O"), 0] += 2 * effect
        info = {
            "kinds": list(KINDS),
            "commit": "abc",
            "uniform_family_share": [1 / 6] * 6,
            "timings_s": {},
            "score_diagnostics": {
                k: [{"spearman": 0.5, "top_overlap": 0.3}] for k in ("RD", "D0", "RAW")
            },
        }
        sweep.write_setting(tmp_path, s["id"], m, info, template_sweep.RUNS)
    return tmp_path


def test_compare_labels_a_clear_win(tmp_path: Path) -> None:
    out = template_compare.run(_fake_root(tmp_path, effect=0.5))
    assert out["settings_run"] == 94
    assert out["complete"] == {"grid": True, "main": True, "replicate": True}
    b1 = out["primary"]["B1"]
    assert b1["wins"] == 64 and b1["label"] == "beats in most sampled settings"
    assert out["primary"]["B2"]["holm_rejected"]
    assert out["share_of_oracle_gain"]["median"] == pytest.approx(0.5, abs=0.05)
    assert set(out["flips"]) == set(template_design.GRID_NAMES)
    assert out["seed_variability"]["B1"]["triples"] == 10
    assert out["note"] == template_compare.NOTE


def test_compare_gives_no_label_to_a_partial_sweep(tmp_path: Path) -> None:
    out = template_compare.run(_fake_root(tmp_path, effect=0.0, ids=list(range(10))))
    assert out["complete"]["main"] is False
    assert out["primary"]["B1"]["label"] == "incomplete"
    assert out["primary_holm"] == {}
