"""Epsilon-greedy logging policy over send or skip, slot and template (METHODS Section 11.7)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from push_lab.sim.env import Action, Observation
from push_lab.sim.world import EXPLORE_STREAM, SLOTS, TEMPLATES, stream

EPSILON = 0.3
SEND_SHARE = 1 / SLOTS


@dataclass
class EpsilonGreedy:
    """Default: one push a day at the user's best slot under the policy profile; with
    probability epsilon the send decision is instead a coin with probability q.
    Templates are uniform. Draws depend only on (seed, day, slot), so reusing the object
    reproduces a run."""

    best_slot: npt.NDArray[np.int64]
    templates: int = TEMPLATES
    epsilon: float = EPSILON
    q: float = SEND_SHARE
    seed: int = 0

    def send_probability(self, slot: int) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = (1 - self.epsilon) * (
            self.best_slot == slot
        ) + self.epsilon * self.q
        return out

    @property
    def min_propensity(self) -> float:
        """Smallest probability of any logged send: off-slot send with one template."""
        return self.epsilon * self.q / self.templates

    def act(self, obs: Observation) -> Action:
        n = len(self.best_slot)
        p_send = self.send_probability(obs.slot)
        rng = stream(EXPLORE_STREAM, self.seed, obs.day, obs.slot)
        send = rng.random(n) < p_send
        template = rng.integers(0, self.templates, n)
        propensity = np.where(send, p_send / self.templates, 1 - p_send)
        return Action(send=send, template=template, propensity=propensity)
