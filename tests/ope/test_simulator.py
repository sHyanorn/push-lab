from typing import Any

import numpy as np
import pytest

from push_lab.ope import simulator as sim_ope
from push_lab.sim import env
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import DAYS, SLOTS

from ..policy.conftest import make_world

USERS = 1_000


def _context(control: bool = True) -> sim_ope.Context:
    pop, static = make_world(seed=3, users=USERS)
    pop = env.Population(pop.visit_rate, pop.profile, pop.p0 * 0.5)
    return sim_ope.make_context(pop, static, control)


CTX = _context()


@pytest.mark.parametrize("target", sim_ope.TARGETS)
def test_target_probabilities_sum_to_one(target: str) -> None:
    user = np.arange(USERS)
    total = sum(
        sim_ope.target_probability(target, CTX, np.broadcast_to(v, (USERS, SLOTS)), user)
        for v in sim_ope.VECTORS
    )
    np.testing.assert_allclose(total, 1)


def test_logging_probabilities_match_the_product_and_the_logs() -> None:
    explorer = EpsilonGreedy(CTX.best, seed=4)
    user = np.arange(USERS)
    total = sum(
        sim_ope.logging_probability(explorer, np.broadcast_to(v, (USERS, SLOTS)), user)
        for v in sim_ope.VECTORS
    )
    np.testing.assert_allclose(total, 1)
    nothing = sim_ope.logging_probability(explorer, np.zeros((USERS, SLOTS), dtype=bool), user)
    np.testing.assert_allclose(nothing, 0.225 * 0.925**3)
    out = env.simulate(CTX.pop, sim_ope.CONTROL, explorer, env.draws(USERS, 4), 1)
    rows = sim_ope.user_days(out, CTX.static)
    share = (~rows.bits).all(axis=1).mean()
    assert share == pytest.approx(0.225 * 0.925**3, abs=0.01)


def test_user_days_align_with_the_outcome() -> None:
    explorer = EpsilonGreedy(CTX.best, seed=5)
    out = env.simulate(CTX.pop, sim_ope.CONTROL, explorer, env.draws(USERS, 5), 1)
    rows = sim_ope.user_days(out, CTX.static)
    assert len(rows.user) == DAYS * USERS
    day3 = slice(3 * USERS, 4 * USERS)
    np.testing.assert_array_equal(rows.reward["visit"][day3], out.visited[3])
    np.testing.assert_array_equal(rows.bits[day3], out.send[3].T)
    np.testing.assert_array_equal(rows.reward["opens"][day3], out.opened[3].sum(axis=0))
    # Day 0 has no history: visit share unknown
    assert np.isnan(rows.x[:USERS, sim_ope.CONTEXT.index("visit_share")]).all()


def test_two_slots_sends_twice_a_day() -> None:
    out = env.simulate(
        CTX.pop, sim_ope.CONTROL, sim_ope.TwoSlots(CTX.best, CTX.second, 1), env.draws(USERS, 1), 1
    )
    assert (out.delivered.sum(axis=1) == 2).all()


def test_truth_is_deterministic() -> None:
    a = sim_ope.truth(CTX, sim_ope.CONTROL, "R", 7)
    assert a == sim_ope.truth(CTX, sim_ope.CONTROL, "R", 7)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ips_and_dr_cover_the_truth_without_carryover() -> None:
    seeds = range(6)
    logs = {s: sim_ope.estimate_log(CTX, sim_ope.CONTROL, s) for s in seeds}
    truths = {
        t: [sim_ope.truth(CTX, sim_ope.CONTROL, t, 100 + s) for s in range(4)]
        for t in ("T", "N", "T2")
    }
    for target in truths:
        values = [t["visit"] for t in truths[target]]
        for name in ("IPS", "DR"):
            ests: list[Any] = [logs[s][target]["visit"]["estimates"][name] for s in seeds]
            check = sim_ope.validate(ests, float(np.mean(values)), float(np.std(values) / 2))
            assert abs(check["bias"]["value"]) < 0.02, (target, name)
            assert check["coverage"]["value"] >= 4 / 6, (target, name)


def test_wilson_interval() -> None:
    low, high = sim_ope.wilson(50, 50)
    assert low == pytest.approx(0.9287, abs=1e-3) and high == pytest.approx(1.0)
    low, high = sim_ope.wilson(40, 50)
    assert low < 0.8 < high


@pytest.mark.parametrize("target", sim_ope.TARGETS)
def test_exact_truth_matches_long_simulation(target: str) -> None:
    exact = sim_ope.exact_truth(CTX, sim_ope.CONTROL, target)
    runs = [sim_ope.truth(CTX, sim_ope.CONTROL, target, 500 + k) for k in range(30)]
    for reward in sim_ope.REWARDS:
        values = np.array([r[reward] for r in runs])
        se = values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean() - exact[reward]) < 4 * se + 1e-4, (target, reward)


def test_exact_truth_refuses_carryover() -> None:
    with pytest.raises(ValueError):
        sim_ope.exact_truth(CTX, sim_ope.SETTINGS["reference"], "T")
    with pytest.raises(ValueError):
        sim_ope.exact_truth(_context(control=False), sim_ope.CONTROL, "T")


def test_truth_seeds_are_independent_across_targets() -> None:
    seeds = [s for t in sim_ope.TARGETS for s in sim_ope.truth_seeds(t)]
    assert len(seeds) == len(set(seeds)) == len(sim_ope.TARGETS) * sim_ope.TRUTH_RUNS
    assert not set(seeds) & set(sim_ope.LOG_SEEDS) and sim_ope.WORLD_SEED not in seeds


def test_validate_uses_an_exact_truth() -> None:
    ests = [{"value": 0.5 + d, "ci_low": 0.49 + d, "ci_high": 0.51 + d} for d in (-0.001, 0.001)]
    check = sim_ope.validate(ests, 0.5, 0.0)
    assert check["bias"]["value"] == pytest.approx(0)
    assert check["coverage"]["value"] == 1
