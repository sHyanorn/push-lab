"""Policy-side state and features for the gate and response models (METHODS Section 12.3).

Features for day d are built at the start of the day from data before d, so the same code
serves the exploration logs the models are fitted on and the policies that use them.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from push_lab.sim.env import Outcome
from push_lab.sim.world import SLOTS, slot_index, slot_time

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

NAMES = (
    "slot",
    "share",
    "is_best",
    "visit_rate",
    "next_score",
    "day",
    "visit_share",
    "visited_yesterday",
    "opens",
    "days_since_open",
    "days_since_delivery",
    "volume",
)
# The policy does not know the simulator's fatigue half-life
VOLUME_HALF_LIFE = 7.0
DAYS_CAP = 28.0


@dataclass(frozen=True)
class Static:
    """Per-user inputs a policy may read: perturbed profile, policy-period visit rate and
    the scores of its candidate list in list order."""

    profile: FloatArray
    visit_rate: FloatArray
    score: FloatArray

    def subset(self, users: IntArray) -> Static:
        return Static(self.profile[users], self.visit_rate[users], self.score[users])


class History:
    """Own deliveries, opens and visits of the days processed so far."""

    def __init__(self, users: int) -> None:
        self.days = 0
        self.deliveries = np.zeros(users, dtype=np.int64)
        self.last_delivery = np.full(users, -np.inf)
        self.opens = np.zeros(users)
        self.last_open = np.full(users, -np.inf)
        self.visits = np.zeros(users)
        self.yesterday = np.full(users, np.nan)
        self.volume = np.zeros(users)
        self.volume_time = 0.0

    def end_day(self, delivered: BoolArray, opened: BoolArray, visited: BoolArray) -> None:
        """Add one day: ``delivered`` and ``opened`` are [slot, user], ``visited`` [user]."""
        d = self.days
        for s in range(SLOTS):
            t = slot_time(d, s)
            self.volume *= 0.5 ** ((t - self.volume_time) / VOLUME_HALF_LIFE)
            self.volume_time = t
            self.volume += delivered[s]
        any_delivery, any_open = delivered.any(axis=0), opened.any(axis=0)
        self.deliveries += delivered.sum(axis=0)
        self.last_delivery[any_delivery] = d
        self.opens += opened.sum(axis=0)
        self.last_open[any_open] = d
        self.visits += visited
        self.yesterday = visited.astype(np.float64)
        self.days += 1

    def features(self, static: Static) -> FloatArray:
        """Features [user, slot, feature] for the next day, in the order of ``NAMES``."""
        d = self.days
        n = len(self.deliveries)
        out = np.empty((n, SLOTS, len(NAMES)))
        best = static.profile.argmax(axis=1)
        position = np.minimum(self.deliveries, static.score.shape[1] - 1)
        common = {
            "visit_rate": static.visit_rate,
            "next_score": static.score[np.arange(n), position],
            "day": np.full(n, float(d)),
            "visit_share": self.visits / d if d > 0 else np.full(n, np.nan),
            "visited_yesterday": self.yesterday,
            "opens": self.opens,
            "days_since_open": np.minimum(DAYS_CAP, d - self.last_open),
            "days_since_delivery": np.minimum(DAYS_CAP, d - self.last_delivery),
        }
        for s in range(SLOTS):
            t = slot_time(d, s)
            column = common | {
                "slot": np.full(n, float(s)),
                "share": static.profile[:, s],
                "is_best": (best == s).astype(np.float64),
                "volume": self.volume * 0.5 ** ((t - self.volume_time) / VOLUME_HALF_LIFE),
            }
            for j, name in enumerate(NAMES):
                out[:, s, j] = column[name]
        return out


@dataclass(frozen=True)
class Rows:
    """One row per (user, day, slot) at which the user had not opted out."""

    x: FloatArray
    treated: BoolArray
    visited: BoolArray
    opened: BoolArray
    user: IntArray
    day: IntArray
    slot: IntArray


def eligible(out: Outcome, day: int, slot: int) -> BoolArray:
    """Users not opted out before this slot (a user who opts out at it was delivered to)."""
    at = out.opted_out_at
    mask: BoolArray = (at < 0) | (at >= slot_index(day, slot))
    return mask


def logged_rows(out: Outcome, static: Static) -> Rows:
    """Rows of a logged run with features as of the start of each day."""
    n = len(out.opted_out_at)
    history = History(n)
    parts: list[tuple[FloatArray, BoolArray, BoolArray, BoolArray, IntArray, IntArray, IntArray]]
    parts = []
    days = out.visited.shape[0]
    for d in range(days):
        f = history.features(static)
        for s in range(SLOTS):
            users = np.flatnonzero(eligible(out, d, s))
            parts.append(
                (
                    f[users, s],
                    out.delivered[d, s, users],
                    out.visited[d, users],
                    out.opened[d, s, users],
                    users,
                    np.full(len(users), d, dtype=np.int64),
                    np.full(len(users), s, dtype=np.int64),
                )
            )
        history.end_day(out.delivered[d], out.opened[d], out.visited[d])
    x, treated, visited, opened, user, day, slot = (
        np.concatenate([p[i] for p in parts]) for i in range(7)
    )
    return Rows(x, treated, visited, opened, user, day, slot)
