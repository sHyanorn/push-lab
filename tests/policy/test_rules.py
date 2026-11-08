import numpy as np
import pytest
from scipy.stats import spearmanr

from push_lab.policy import rules
from push_lab.policy.features import History
from push_lab.policy.gate import KINDS, Kind
from push_lab.sim import env
from push_lab.sim.world import DAYS, SLOTS, Params

from .conftest import USERS, World

PARAMS = Params(base_rate=0.2, opt_out=0.002)


def _run(world: World, policy: env.Policy, seed: int = 5) -> env.Outcome:
    return env.simulate(world.pop, PARAMS, policy, env.draws(USERS, seed))


@pytest.mark.parametrize("kind", KINDS)
def test_scored_policies_send_the_budget_once_a_day(world: World, kind: Kind) -> None:
    policy = rules.Scored(world.models, world.static, kind, budget=0.3, run_seed=1)
    out = _run(world, policy)
    per_day = out.delivered.sum(axis=(1, 2))
    eligible = np.array([(~_opted_before(out, d)).sum() for d in range(DAYS)])
    np.testing.assert_array_equal(per_day, np.minimum(policy.cap, eligible))
    assert out.delivered.sum(axis=1).max() == 1


def _opted_before(out: env.Outcome, day: int) -> np.ndarray:
    at = out.opted_out_at
    return (at >= 0) & (at < day * SLOTS)


def test_the_three_scores_send_equal_volume(world: World) -> None:
    volumes = {
        k: _run(world, rules.Scored(world.models, world.static, k, 0.5, 1)).delivered.sum()
        for k in KINDS
    }
    assert len(set(volumes.values())) == 1 or max(volumes.values()) - min(volumes.values()) <= int(
        0.01 * USERS * DAYS
    )


def test_a_multiplier_never_exceeds_the_cap(world: World) -> None:
    policy = rules.Scored(world.models, world.static, "S", 0.2, 1, multiplier=0.0)
    out = _run(world, policy)
    assert out.delivered.sum(axis=(1, 2)).max() <= policy.cap
    assert policy.cap_binding_days == DAYS
    strict = rules.Scored(world.models, world.static, "S", 0.2, 1, multiplier=1.0)
    assert not _run(world, strict).delivered.any()


def test_all_policies_share_templates(world: World) -> None:
    daily = _run(world, rules.Daily(np.zeros(USERS, dtype=np.int64), run_seed=4))
    scored = _run(world, rules.Scored(world.models, world.static, "G", 0.5, 4))
    both = daily.send & scored.send
    assert both.any()
    np.testing.assert_array_equal(daily.template[both], scored.template[both])


def test_daily_and_half_daily(world: World) -> None:
    slot = np.random.default_rng(0).integers(0, SLOTS, USERS)
    full = _run(world, rules.Daily(slot, run_seed=2))
    half = _run(world, rules.Daily(slot, run_seed=2, half=True))
    assert (full.send.sum(axis=1) == 1).all()
    assert (full.send.argmax(axis=1) == slot).all()
    assert not (half.send & ~full.send).any()
    assert half.send.mean() == pytest.approx(full.send.mean() / 2, rel=0.05)
    never = _run(world, rules.Never(USERS, 2))
    assert not never.send.any()
    random = _run(world, rules.RandomSlots(USERS, 2))
    assert random.send.mean() == pytest.approx(0.25, abs=0.01)


def test_fixed_seeds_reproduce(world: World) -> None:
    a = _run(world, rules.Scored(world.models, world.static, "G2", 0.5, 3))
    b = _run(world, rules.Scored(world.models, world.static, "G2", 0.5, 3))
    np.testing.assert_array_equal(a.delivered, b.delivered)
    np.testing.assert_array_equal(a.visited, b.visited)


def test_open_model_ranks_true_open_probability(world: World) -> None:
    """Sanity check of the fitted model on fresh logs with the simulator's truth."""
    policy = rules.Daily(world.static.profile.argmax(axis=1), run_seed=9)
    out = env.simulate(world.pop, PARAMS, policy, env.draws(USERS, 9), truth=True)
    assert out.open_prob is not None
    history = History(USERS)
    f = history.features(world.static)
    slot, p_hat = world.models.scores(f, "S")
    true = out.open_prob[0, slot, np.arange(USERS)]
    assert spearmanr(p_hat, true).statistic > 0.3
