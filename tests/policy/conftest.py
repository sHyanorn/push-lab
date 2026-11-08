"""A small synthetic world shared by the policy tests (no real data)."""

from dataclasses import dataclass

import numpy as np
import pytest

from push_lab.policy.features import Static, logged_rows
from push_lab.policy.gate import Models
from push_lab.sim import env
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import SLOTS, Params

USERS = 2_000
CANDIDATES = 120


@dataclass(frozen=True)
class World:
    pop: env.Population
    static: Static
    models: Models


def make_world(seed: int = 0, users: int = USERS) -> tuple[env.Population, Static]:
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.05, 0.6, users)
    p0 = quality[:, None] * (1 - 0.004 * np.arange(CANDIDATES))[None, :]
    profile = rng.dirichlet(np.ones(SLOTS) * 3, users)
    visit = rng.uniform(0.1, 0.9, users)
    pop = env.Population(visit, profile, p0)
    # The policy sees noisy versions of the truth, as in the sweep
    static = Static(
        profile=profile * np.exp(0.3 * rng.standard_normal(profile.shape)),
        visit_rate=np.clip(visit + 0.1 * rng.standard_normal(users), 0.01, 0.99),
        score=p0 + 0.02 * rng.standard_normal(p0.shape),
    )
    return pop, static


@pytest.fixture(scope="session")
def world() -> World:
    pop, static = make_world()
    params = Params(base_rate=0.2)
    explorer = EpsilonGreedy(static.profile.argmax(axis=1), seed=11)
    logs = env.simulate(pop, params, explorer, env.draws(USERS, 11))
    return World(pop, static, Models.fit(logged_rows(logs, static)))
