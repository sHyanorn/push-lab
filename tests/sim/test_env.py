from dataclasses import dataclass, replace

import numpy as np
import pytest

from push_lab.sim import env, world
from push_lab.sim.env import Action, Observation
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import DAYS, SLOTS, Params

N = 3_000
REFERENCE = Params()
QUIET = Params(fatigue_elasticity=0.0, annoyance=0.0, novelty=0.0, opt_out=0.0)


def _population(seed: int = 0, constant_items: bool = True) -> env.Population:
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0.05, 0.6, size=(N, 1))
    return env.Population(
        visit_rate=rng.uniform(0.1, 0.9, N),
        profile=rng.dirichlet(np.ones(SLOTS), N),
        p0=np.repeat(p0, 120, axis=1) if constant_items else rng.uniform(0.05, 0.6, (N, 120)),
    )


@dataclass
class Fixed:
    """Deterministic policy: send at the given slots with a template chosen by ``template``."""

    slots: tuple[int, ...]
    template: int | None = None  # None rotates templates by day

    def act(self, obs: Observation) -> Action:
        send = np.full(N, obs.slot in self.slots)
        k = obs.day % world.TEMPLATES if self.template is None else self.template
        return Action(send, np.full(N, k, dtype=np.int64), np.ones(N))


def _run(policy: env.Policy, params: Params = REFERENCE, seed: int = 1) -> env.Outcome:
    return env.simulate(_population(), params, policy, env.draws(N, seed))


def test_no_sends_leave_only_organic_visits() -> None:
    pop, run = _population(), env.draws(N, 1)
    out = env.simulate(pop, Params(), Fixed(slots=()), run)
    assert not out.delivered.any()
    assert (out.opted_out_at == -1).all()
    np.testing.assert_array_equal(out.visited, run.visit < pop.visit_rate)


def test_more_sends_never_lower_opens_or_visits_without_fatigue() -> None:
    few, many = _run(Fixed(slots=(0,)), QUIET), _run(Fixed(slots=(0, 2, 3)), QUIET)
    assert not (few.opened & ~many.opened).any()
    assert not (few.visited & ~many.visited).any()
    assert many.opened.sum() > few.opened.sum()


def test_template_does_not_matter_without_copy_effects() -> None:
    params = replace(Params(), copy_spread=0.0, novelty=0.0)
    a, b = _run(Fixed(slots=(1,), template=3), params), _run(Fixed(slots=(1,)), params)
    np.testing.assert_array_equal(a.opened, b.opened)
    np.testing.assert_array_equal(a.visited, b.visited)
    np.testing.assert_array_equal(a.opted_out_at, b.opted_out_at)


def test_repeating_a_template_is_penalized() -> None:
    params = replace(QUIET, copy_spread=0.0, novelty=0.9, novelty_half_life=30.0)
    repeat, rotate = _run(Fixed(slots=(1,), template=0), params), _run(Fixed(slots=(1,)), params)
    assert not (repeat.opened & ~rotate.opened).any()
    assert repeat.opened.sum() < rotate.opened.sum()


def test_fatigue_lowers_opens_on_the_same_draws() -> None:
    tired = replace(QUIET, fatigue_elasticity=0.8)
    assert _run(Fixed(slots=(0, 1, 2, 3)), tired).opened.sum() < (
        _run(Fixed(slots=(0, 1, 2, 3)), QUIET).opened.sum()
    )


def test_opted_out_users_receive_nothing_more() -> None:
    out = _run(Fixed(slots=(0, 1, 2, 3)), replace(Params(), opt_out=0.01))
    flat = out.delivered.reshape(DAYS * SLOTS, N)
    left = np.flatnonzero(out.opted_out_at >= 0)
    assert len(left) > 0
    for u in left[:50]:
        assert not flat[out.opted_out_at[u] + 1 :, u].any()


def test_fixed_seeds_reproduce() -> None:
    a = _run(EpsilonGreedy(np.zeros(N, dtype=np.int64), seed=3))
    b = _run(EpsilonGreedy(np.zeros(N, dtype=np.int64), seed=3))
    for field in ("send", "delivered", "opened", "template", "item", "visited", "opted_out_at"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_items_follow_the_list_without_repeats() -> None:
    out = _run(Fixed(slots=(0, 2)), QUIET)
    items = out.item.reshape(DAYS * SLOTS, N)[:, 0]
    sent = items[items >= 0]
    np.testing.assert_array_equal(sent, np.arange(len(sent)))


def test_reference_volume_is_the_steady_state_sum() -> None:
    for h in (3.0, 7.0, 14.0):
        explicit = sum(0.5 ** (k / h) for k in range(1, 5_000))
        assert world.reference_volume(h) == pytest.approx(explicit)
        p = Params(fatigue_half_life=h)
        assert world.fatigue_rate(p) * world.reference_volume(h) == pytest.approx(
            p.fatigue_elasticity
        )
        assert world.annoyance_rate(p) * world.reference_volume(h) == pytest.approx(p.annoyance)


@pytest.mark.parametrize("lift", [0.0, 0.02, 0.1, 0.25])
def test_eta_reproduces_the_timing_lift(lift: float) -> None:
    profile = _population().profile
    eta = world.solve_eta(profile, lift)
    assert world.timing_lift(profile, eta) == pytest.approx(lift, abs=1e-8)
    np.testing.assert_allclose(world.timing_multipliers(profile, eta).mean(axis=1), 1.0)


def test_copy_effects_have_mean_one_and_the_given_spread() -> None:
    c = world.copy_effects(0.2, seed=0, templates=60_000, families=6_000)
    assert c.mean() == pytest.approx(1.0, abs=0.01)
    assert np.log(c).std() == pytest.approx(0.2, abs=0.01)
    np.testing.assert_array_equal(world.copy_effects(0.0, seed=0), np.ones(world.TEMPLATES))


def test_exploration_propensities_and_frequencies() -> None:
    best = np.random.default_rng(0).integers(0, SLOTS, N)
    policy = EpsilonGreedy(best, seed=0)
    sends = np.zeros(SLOTS)
    counts = np.zeros(world.TEMPLATES)
    days = 40
    for d in range(days):
        for s in range(SLOTS):
            a = policy.act(Observation(d, s, np.zeros(N, dtype=bool)))
            p_send = policy.send_probability(s)
            expected = np.where(a.send, p_send / world.TEMPLATES, 1 - p_send)
            np.testing.assert_allclose(a.propensity, expected)
            sends[s] += a.send[best == s].mean()
            counts += np.bincount(a.template[a.send], minlength=world.TEMPLATES)
    assert policy.send_probability(0)[best == 0][0] == pytest.approx(0.775)
    assert policy.send_probability(0)[best != 0][0] == pytest.approx(0.075)
    np.testing.assert_allclose(sends / days, 0.775, atol=0.01)
    share = counts / counts.sum()
    np.testing.assert_allclose(share, 1 / world.TEMPLATES, atol=0.004)
    assert policy.min_propensity == pytest.approx(0.075 / world.TEMPLATES)


def test_exploration_sends_one_push_a_day_on_average() -> None:
    out = _run(EpsilonGreedy(np.random.default_rng(0).integers(0, SLOTS, N), seed=0), QUIET)
    assert out.send.sum() / (N * DAYS) == pytest.approx(1.0, abs=0.02)
