"""Delivery policies (METHODS Section 12.2).

Every policy draws templates from one stream keyed by (run seed, day, slot), so all
policies give a user the same template at the same (day, slot).
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from push_lab.policy import budget
from push_lab.policy.features import History, Static
from push_lab.policy.gate import Kind, Models
from push_lab.sim.env import Action, Observation
from push_lab.sim.world import COIN_STREAM, HALF_STREAM, SLOTS, TEMPLATE_STREAM, TEMPLATES, stream

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def templates(run_seed: int, day: int, slot: int, users: int, k: int = TEMPLATES) -> IntArray:
    out: IntArray = stream(TEMPLATE_STREAM, run_seed, day, slot).integers(0, k, users)
    return out


def _action(send: BoolArray, run_seed: int, obs: Observation) -> Action:
    n = len(send)
    return Action(send, templates(run_seed, obs.day, obs.slot, n), np.ones(n))


@dataclass
class Never:
    users: int
    run_seed: int

    def act(self, obs: Observation) -> Action:
        return _action(np.zeros(self.users, dtype=bool), self.run_seed, obs)


@dataclass
class RandomSlots:
    """Each slot with probability 1/4, independently; no daily cap."""

    users: int
    run_seed: int
    p: float = 1 / SLOTS

    def act(self, obs: Observation) -> Action:
        rng = stream(COIN_STREAM, self.run_seed, obs.day, obs.slot)
        send = rng.random(self.users) < self.p
        a = _action(send, self.run_seed, obs)
        return Action(a.send, a.template, np.where(send, self.p, 1 - self.p))


@dataclass
class Daily:
    """One push a day at each user's given slot; with ``half``, a seeded random half of the
    users each day."""

    slot: IntArray
    run_seed: int
    half: bool = False

    def act(self, obs: Observation) -> Action:
        send = self.slot == obs.slot
        if self.half:
            rng = stream(HALF_STREAM, self.run_seed, obs.day)
            send &= rng.random(len(self.slot)) < 0.5
        return _action(send, self.run_seed, obs)


@dataclass
class Scored:
    """S, G or G2: each day, score every eligible user at the slot with the highest predicted
    open probability and send to the ``floor(budget * n)`` highest scores. With a
    ``multiplier``, only scores above it are sent, within the same daily cap."""

    models: Models
    static: Static
    kind: Kind
    budget: float
    run_seed: int
    multiplier: float | None = None
    cap_binding_days: int = 0
    _history: History = field(init=False)
    _plan: IntArray = field(init=False)
    _delivered: BoolArray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.static.visit_rate)
        self._history = History(n)
        self._plan = np.full(n, -1, dtype=np.int64)
        self._delivered = np.zeros((SLOTS, n), dtype=bool)

    @property
    def cap(self) -> int:
        return int(np.floor(self.budget * len(self.static.visit_rate)))

    def _plan_day(self, obs: Observation) -> None:
        if obs.day > 0:
            opened, visited = obs.history()
            self._history.end_day(self._delivered, opened[obs.day - 1], visited[obs.day - 1])
        if self._history.days != obs.day:
            raise ValueError("a scored policy must see every day from day 0")
        self._delivered[:] = False
        n = len(self._plan)
        eligible = ~obs.opted_out
        users = np.flatnonzero(eligible)
        f = self._history.features(self.static)
        slot, score = self.models.scores(f[users], self.kind)
        full_slot = np.full(n, -1, dtype=np.int64)
        full_score = np.full(n, -np.inf)
        full_slot[users], full_score[users] = slot, score
        if self.multiplier is not None:
            above = eligible & (full_score > self.multiplier)
            self.cap_binding_days += int(above.sum() > self.cap)
            eligible = above
        chosen = budget.top_k(full_score, eligible, self.cap)
        self._plan = np.where(chosen, full_slot, -1)

    def act(self, obs: Observation) -> Action:
        if obs.slot == 0:
            self._plan_day(obs)
        send = self._plan == obs.slot
        self._delivered[obs.slot] = send & ~obs.opted_out
        return _action(send, self.run_seed, obs)
