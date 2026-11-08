"""Template policies on a fixed delivery rule (METHODS 13.4).

U picks uniformly among the eligible templates; RD is the recovering difference softmax of
Yancey and Settles (2020), D0 the same without the recency penalty, RAW the same on raw
means; O is the myopic oracle that knows the drawn copy effects. Every policy sends one push a
day at the user's given slot, so all of them make the same sends.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from push_lab.sim.env import Action, Observation
from push_lab.sim.world import CHOICE_STREAM, SLOT_HOURS, stream

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

Kind = Literal["U", "RD", "D0", "RAW", "O"]
KINDS: tuple[Kind, ...] = ("U", "RD", "D0", "RAW", "O")
LEARNED: tuple[Kind, ...] = ("RD", "D0", "RAW")

# Fixed before any run (D037)
PENALTY = 0.017
PENALTY_HALF_LIFE = 15.0
TEMPERATURE = 0.005
THETA = 0.005

# Family ids with eligibility rules (METHODS 13.1)
CREATOR, POPULAR, BACK = 0, 1, 4
BACK_DAYS = 2


def choice_uniforms(run_seed: int, day: int, slot: int, users: int) -> FloatArray:
    """One uniform per user at (day, slot), shared by every template policy."""
    out: FloatArray = stream(CHOICE_STREAM, run_seed, day, slot).random(users)
    return out


def inverse_cdf(probs: FloatArray, u: FloatArray) -> IntArray:
    """Per row, the first column whose cumulative probability exceeds ``u``."""
    cum = np.cumsum(probs, axis=1)
    idx = (cum <= (u * cum[:, -1])[:, None]).sum(axis=1)
    # Guard against rounding at the top: never pick a zero-probability column
    last = probs.shape[1] - 1 - np.argmax(probs[:, ::-1] > 0, axis=1)
    out: IntArray = np.minimum(idx, last).astype(np.int64)
    return out


@dataclass
class ArmStats:
    """Additive sums over rounds for each arm (Yancey and Settles 2020, Section 2)."""

    arms: int
    plus: FloatArray = field(init=False)  # rows: sum w, sum w r, sum w^2
    minus: FloatArray = field(init=False)
    eligible: FloatArray = field(init=False)  # rows: rounds, rewards
    rounds: float = 0.0
    rewards: float = 0.0

    def __post_init__(self) -> None:
        self.plus = np.zeros((3, self.arms))
        self.minus = np.zeros((3, self.arms))
        self.eligible = np.zeros((2, self.arms))

    def add(self, probs: FloatArray, chosen: IntArray, reward: FloatArray) -> None:
        """``probs`` [round, arm] is 0 for ineligible arms; ``reward`` is 0 or 1 per round."""
        valid = (probs >= THETA) & (probs <= 1 - THETA)
        picked = np.zeros(probs.shape, dtype=bool)
        picked[np.arange(len(chosen)), chosen] = True
        r = reward[:, None]
        with np.errstate(divide="ignore", over="ignore"):
            w_plus = np.where(valid & picked, 1 / probs, 0.0)
            w_minus = np.where(valid & ~picked, 1 / (1 - probs), 0.0)
        for sums, w in ((self.plus, w_plus), (self.minus, w_minus)):
            sums[0] += w.sum(axis=0)
            sums[1] += (w * r).sum(axis=0)
            sums[2] += (w**2).sum(axis=0)
        self.eligible[0] += valid.sum(axis=0)
        self.eligible[1] += (valid * r).sum(axis=0)
        self.rounds += len(chosen)
        self.rewards += float(reward.sum())

    @staticmethod
    def _mean_and_size(sums: FloatArray) -> tuple[FloatArray, FloatArray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(sums[0] > 0, sums[1] / sums[0], 0.0)
            size = np.where(sums[2] > 0, sums[0] ** 2 / sums[2], 0.0)
        return mean, size

    @staticmethod
    def prior_strength(
        mean: FloatArray, size: FloatArray, center: FloatArray, base: float
    ) -> float:
        """Method-of-moments prior strength m; infinite when arms vary no more than noise."""
        seen = size > 0
        if not seen.any() or not 0 < base < 1:
            return np.inf
        c = center[seen]
        excess = np.mean((mean[seen] - c) ** 2) - np.mean(c * (1 - c) / size[seen])
        if excess <= 0:
            return np.inf
        return max(0.0, base * (1 - base) / excess - 1)

    @staticmethod
    def _shrink(mean: FloatArray, size: FloatArray, center: FloatArray, m: float) -> FloatArray:
        """Posterior mean toward ``center`` with prior strength ``m`` (the center if no data)."""
        total = size + m
        with np.errstate(divide="ignore", invalid="ignore"):
            out: FloatArray = np.where(total > 0, (mean * size + center * m) / total, center)
        return out

    def scores(self, raw: bool = False) -> FloatArray:
        """Relative difference scores (Eq. 2); with ``raw``, relative to the overall mean."""
        out = np.zeros(self.arms)
        if self.rounds == 0:
            return out
        overall = self.rewards / self.rounds
        mu_plus, n_plus = self._mean_and_size(self.plus)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_arm = np.where(self.eligible[0] > 0, self.eligible[1] / self.eligible[0], 0.0)
        center = np.full(self.arms, overall) if raw else mu_arm
        m = self.prior_strength(mu_plus, n_plus, center, overall)
        if not np.isfinite(m):
            return out
        plus = self._shrink(mu_plus, n_plus, center, m)
        if raw:
            baseline = np.full(self.arms, overall)
        else:
            baseline = self._shrink(*self._mean_and_size(self.minus), center, m)
        ok = (baseline > 0) & (self.eligible[0] > 0)
        out[ok] = (plus[ok] - baseline[ok]) / baseline[ok]
        return out


def softmax(scores: FloatArray, eligible: BoolArray, temperature: float) -> FloatArray:
    z = np.where(eligible, scores / temperature, -np.inf)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: FloatArray = e / e.sum(axis=1, keepdims=True)
    return out


@dataclass
class Templated:
    """One push a day at ``slot[u]``, with the template chosen by ``kind``.

    ``family`` gives each template's family; ``creator_clicks`` and ``trend_rank`` are
    [user, list position] eligibility inputs. ``copy``, ``novelty`` and ``novelty_half_life``
    are the setting's truth and are read only by the oracle O."""

    kind: Kind
    slot: IntArray
    family: IntArray
    creator_clicks: IntArray
    trend_rank: IntArray
    run_seed: int
    copy: FloatArray | None = None
    novelty: float = 0.0
    novelty_half_life: float = 1.0
    scores: FloatArray = field(init=False)
    _stats: ArmStats = field(init=False)
    _delivered: IntArray = field(init=False)
    _last_seen: FloatArray = field(init=False)
    _away: BoolArray = field(init=False)
    _pending: list[tuple[int, IntArray, FloatArray, IntArray]] = field(init=False)
    _day: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        n, k = len(self.slot), len(self.family)
        if self.kind == "O" and self.copy is None:
            raise ValueError("the oracle needs the copy effects")
        self.scores = np.zeros(k)
        self._stats = ArmStats(k)
        self._delivered = np.zeros(n, dtype=np.int64)
        self._last_seen = np.full((n, k), -np.inf)
        self._away = np.zeros(n, dtype=bool)
        self._pending = []

    @property
    def penalty(self) -> float:
        return 0.0 if self.kind == "D0" else PENALTY

    def eligible(self, users: IntArray) -> BoolArray:
        position = np.minimum(self._delivered[users], self.creator_clicks.shape[1] - 1)
        rules = {
            CREATOR: self.creator_clicks[users, position] >= 1,
            POPULAR: self.trend_rank[users, position] > 0,
            BACK: self._away[users],
        }
        out = np.ones((len(users), len(self.family)), dtype=bool)
        for fam, ok in rules.items():
            out[:, self.family == fam] = ok[:, None]
        return out

    def _start_day(self, obs: Observation) -> None:
        if obs.day != self._day + 1:
            raise ValueError("a template policy must see every day from day 0")
        self._day = obs.day
        opened, visited = obs.history()
        for slot, users, probs, chosen in self._pending:
            reward = opened[obs.day - 1, slot, users].astype(np.float64)
            self._stats.add(probs, chosen, reward)
        self._pending = []
        if self.kind in LEARNED:
            self.scores = self._stats.scores(raw=self.kind == "RAW")
        d = obs.day
        if d >= BACK_DAYS:
            self._away = ~visited[d - BACK_DAYS : d].any(axis=0)

    def probabilities(self, users: IntArray, now: float) -> tuple[BoolArray, FloatArray]:
        eligible = self.eligible(users)
        since = now - self._last_seen[users]
        if self.kind == "U":
            probs = eligible / eligible.sum(axis=1, keepdims=True)
        elif self.kind == "O":
            assert self.copy is not None
            value = self.copy[None, :] * (
                1 - self.novelty * 0.5 ** (since / self.novelty_half_life)
            )
            value = np.where(eligible, value, -np.inf)
            probs = np.zeros(eligible.shape)
            probs[np.arange(len(users)), value.argmax(axis=1)] = 1.0
        else:
            s = self.scores[None, :] - self.penalty * 0.5 ** (since / PENALTY_HALF_LIFE)
            probs = softmax(s, eligible, TEMPERATURE)
        return eligible, probs

    def act(self, obs: Observation) -> Action:
        if obs.slot == 0:
            self._start_day(obs)
        n = len(self.slot)
        send = self.slot == obs.slot
        template = np.zeros(n, dtype=np.int64)
        propensity = np.ones(n)
        users = np.flatnonzero(send & ~obs.opted_out)
        if len(users):
            now = obs.day + SLOT_HOURS[obs.slot] / 24
            _, probs = self.probabilities(users, now)
            u = choice_uniforms(self.run_seed, obs.day, obs.slot, n)[users]
            chosen = inverse_cdf(probs, u)
            template[users] = chosen
            propensity[users] = probs[np.arange(len(users)), chosen]
            self._pending.append((obs.slot, users, probs, chosen))
            self._last_seen[users, chosen] = now
            self._delivered[users] += 1
        return Action(send, template, propensity)
