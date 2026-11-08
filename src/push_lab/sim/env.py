"""Vectorized push simulator over users x days x slots (METHODS Section 11)."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from push_lab.sim import world
from push_lab.sim.world import DAYS, SLOT_HOURS, SLOTS, Params

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Population:
    """Per-user inputs: organic visit rate, simulator activity profile and the response
    model's click probability for each item of the user's candidate list, in list order."""

    visit_rate: FloatArray
    profile: FloatArray
    p0: FloatArray

    @property
    def users(self) -> int:
        return len(self.visit_rate)


@dataclass(frozen=True)
class Draws:
    """Uniform draws shared by every policy run with the same run seed."""

    open: FloatArray
    opt_out: FloatArray
    visit: FloatArray


def draws(users: int, seed: int) -> Draws:
    rng = np.random.default_rng(seed)
    return Draws(
        open=rng.random((DAYS, SLOTS, users)),
        opt_out=rng.random((DAYS, SLOTS, users)),
        visit=rng.random((DAYS, users)),
    )


@dataclass(frozen=True)
class Observation:
    day: int
    slot: int
    opted_out: BoolArray


@dataclass(frozen=True)
class Action:
    """Per user: whether to send, the template, and the probability the policy gave the
    action it took (1 for a deterministic policy)."""

    send: BoolArray
    template: IntArray
    propensity: FloatArray


class Policy(Protocol):
    def act(self, obs: Observation) -> Action: ...


@dataclass(frozen=True)
class Outcome:
    """Everything that happened, indexed [day, slot, user] or [day, user]."""

    send: BoolArray
    delivered: BoolArray
    opened: BoolArray
    template: npt.NDArray[np.int16]
    item: npt.NDArray[np.int16]
    propensity: npt.NDArray[np.float32]
    opted_out_at: IntArray
    visited: BoolArray


def simulate(
    pop: Population,
    params: Params,
    policy: Policy,
    run: Draws,
    world_seed: int = 0,
    templates: int = world.TEMPLATES,
) -> Outcome:
    n, candidates = pop.p0.shape
    p = params
    timing = world.timing_multipliers(pop.profile, world.solve_eta(pop.profile, p.timing_lift))
    copy = world.copy_effects(p.copy_spread, world_seed, templates)
    phi, delta = world.fatigue_rate(p), world.annoyance_rate(p)
    f_ref = world.reference_volume(p.fatigue_half_life)

    volume = np.zeros(n)  # decayed delivered volume at time `now`
    now = 0.0
    last_seen = np.full((n, templates), -np.inf)
    returns = np.zeros(n)
    pointer = np.zeros(n, dtype=np.int64)
    opted = np.zeros(n, dtype=bool)

    shape = (DAYS, SLOTS, n)
    out_send = np.zeros(shape, dtype=bool)
    out_delivered = np.zeros(shape, dtype=bool)
    out_opened = np.zeros(shape, dtype=bool)
    out_template = np.full(shape, -1, dtype=np.int16)
    out_item = np.full(shape, -1, dtype=np.int16)
    out_propensity = np.zeros(shape, dtype=np.float32)
    opted_out_at = np.full(n, -1, dtype=np.int64)
    visited = np.zeros((DAYS, n), dtype=bool)

    def advance(t: float) -> None:
        nonlocal now
        volume[:] *= 0.5 ** ((t - now) / p.fatigue_half_life)
        now = t

    for d in range(DAYS):
        advance(float(d))
        organic = np.minimum(
            1.0, pop.visit_rate * np.exp(-delta * volume) + p.return_boost * returns
        )
        opens_today = np.zeros(n, dtype=np.int64)
        for s, hour in enumerate(SLOT_HOURS):
            t = d + hour / 24
            advance(t)
            action = policy.act(Observation(d, s, opted.copy()))
            out_send[d, s] = action.send
            out_propensity[d, s] = action.propensity
            idx = np.flatnonzero(action.send & ~opted)
            if len(idx) == 0:
                continue
            k = action.template[idx]
            item = np.minimum(pointer[idx], candidates - 1)
            novelty = 1 - p.novelty * 0.5 ** ((t - last_seen[idx, k]) / p.novelty_half_life)
            prob = (
                p.base_rate
                * pop.p0[idx, item]
                * timing[idx, s]
                * copy[k]
                * np.exp(-phi * volume[idx])
                * novelty
            )
            opened = run.open[d, s, idx] < np.minimum(prob, 1.0)
            volume[idx] += 1.0
            hazard = p.opt_out * volume[idx] / (f_ref + 1.0)
            leaving = idx[run.opt_out[d, s, idx] < hazard]
            opted[leaving] = True
            opted_out_at[leaving] = d * SLOTS + s
            last_seen[idx, k] = t
            pointer[idx] += 1
            opens_today[idx] += opened
            out_delivered[d, s, idx] = True
            out_opened[d, s, idx] = opened
            out_template[d, s, idx] = k
            out_item[d, s, idx] = item
        visited[d] = (run.visit[d] < organic) | (opens_today > 0)
        returns = returns * 0.5 ** (1.0 / p.return_half_life) + opens_today
    return Outcome(
        send=out_send,
        delivered=out_delivered,
        opened=out_opened,
        template=out_template,
        item=out_item,
        propensity=out_propensity,
        opted_out_at=opted_out_at,
        visited=visited,
    )
