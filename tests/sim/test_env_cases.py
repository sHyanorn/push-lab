"""Single-user cases computed by hand from METHODS Sections 11.2-11.4."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from push_lab.sim import env, world
from push_lab.sim.env import Action, Observation
from push_lab.sim.world import DAYS, SLOT_HOURS, SLOTS, Params

P0 = 0.5
VISIT = 0.4
PARAMS = Params(timing_lift=0.0, copy_spread=0.0)


@dataclass
class Script:
    """Send template 0 at the listed (day, slot) pairs."""

    sends: frozenset[tuple[int, int]]

    def act(self, obs: Observation) -> Action:
        send = np.array([(obs.day, obs.slot) in self.sends])
        return Action(send, np.zeros(1, dtype=np.int64), np.ones(1))


def _run(
    sends: set[tuple[int, int]],
    params: Params = PARAMS,
    opened_at: set[tuple[int, int]] | None = None,
    opens: float = 1.0,
    opt_out: float = 1.0,
    visit: float = 1.0,
) -> env.Outcome:
    """Uniform draws are constant; ``opened_at`` forces opens there (draw 0)."""
    draws = env.Draws(
        open=np.full((DAYS, SLOTS, 1), opens),
        opt_out=np.full((DAYS, SLOTS, 1), opt_out),
        visit=np.full((DAYS, 1), visit),
    )
    for d, s in opened_at or set():
        draws.open[d, s, 0] = 0.0
    pop = env.Population(np.array([VISIT]), np.full((1, SLOTS), 0.25), np.full((1, 5), P0))
    return env.simulate(pop, params, Script(frozenset(sends)), draws)


def _second_push_probability(p: Params) -> float:
    # One earlier delivery exactly one day before; same template, one day ago
    f_minus = 0.5 ** (1 / p.fatigue_half_life)
    novelty = 1 - p.novelty * 0.5 ** (1 / p.novelty_half_life)
    return float(p.base_rate * P0 * np.exp(-world.fatigue_rate(p) * f_minus) * novelty)


@pytest.mark.parametrize("delta", [-1e-9, 1e-9])
def test_open_probability_uses_volume_before_the_push(delta: float) -> None:
    p = replace(PARAMS, base_rate=0.2, fatigue_elasticity=0.8, novelty=0.05)
    threshold = _second_push_probability(p)
    draws_open = threshold + delta
    out = _run({(0, 0), (1, 0)}, p, opens=draws_open)
    assert out.opened[1, 0, 0] == (delta < 0)


@pytest.mark.parametrize("delta", [-1e-12, 1e-12])
def test_opt_out_hazard_uses_volume_including_the_push(delta: float) -> None:
    p = replace(PARAMS, opt_out=0.004)
    f_plus = 0.5 ** (1 / p.fatigue_half_life) + 1
    hazard = p.opt_out * f_plus / (world.reference_volume(p.fatigue_half_life) + 1)
    # The first push must not trigger: its hazard is smaller
    out = _run({(0, 0), (1, 0)}, p, opt_out=hazard + delta)
    assert (out.opted_out_at[0] == SLOTS) == (delta < 0)


def test_hazard_is_kappa_at_one_push_a_day_in_steady_state() -> None:
    p = Params(fatigue_half_life=3.0)
    f_plus = sum(0.5 ** (k / p.fatigue_half_life) for k in range(0, 2_000))
    assert p.opt_out * f_plus / (world.reference_volume(p.fatigue_half_life) + 1) == (
        pytest.approx(p.opt_out)
    )


def test_return_boost_and_annoyance_set_organic_visits() -> None:
    p = replace(PARAMS, return_boost=0.1, return_half_life=1.0, annoyance=0.1)
    out = _run({(0, 0)}, p, opened_at={(0, 0)})
    delta = world.annoyance_rate(p)
    hour = SLOT_HOURS[0] / 24
    for day, returns in ((1, 1.0), (2, 0.5)):
        volume = 0.5 ** ((day - hour) / p.fatigue_half_life)
        organic = VISIT * np.exp(-delta * volume) + p.return_boost * returns
        for gap, expected in ((-1e-9, True), (1e-9, False)):
            again = _run({(0, 0)}, p, opened_at={(0, 0)}, visit=organic + gap)
            assert again.visited[day, 0] == expected
    # The day of the open counts as a visit whatever the organic draw
    assert out.visited[0, 0]
    assert not out.visited[3:, 0].any()


def test_opted_out_users_keep_visiting_organically() -> None:
    p = replace(PARAMS, opt_out=0.001, annoyance=0.0, return_boost=0.0)
    out = _run({(0, 0), (1, 0), (2, 0)}, p, opt_out=0.0, visit=VISIT - 1e-9)
    assert out.opted_out_at[0] == 0
    assert out.delivered[:, :, 0].sum() == 1
    assert out.visited[:, 0].all()


def test_simulation_does_not_change_the_draws() -> None:
    draws = env.draws(1, seed=5)
    before = draws.open.copy()
    pop = env.Population(np.array([VISIT]), np.full((1, SLOTS), 0.25), np.full((1, 5), P0))
    env.simulate(pop, PARAMS, Script(frozenset({(0, 0), (3, 2)})), draws)
    np.testing.assert_array_equal(draws.open, before)
    np.testing.assert_array_equal(env.draws(1, seed=5).open, before)
