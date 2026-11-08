"""Simulator parameters and the constants derived from them (METHODS Section 11)."""

from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DAYS = 28
SLOT_HOURS = (8, 12, 17, 21)
# A push sent at a slot is opened, if at all, within the following three hours
WINDOWS = ((8, 11), (12, 15), (17, 20), (21, 24))
SLOTS = len(SLOT_HOURS)
# Until the committed copy sets them (M6)
TEMPLATES = 60
FAMILIES = 6


@dataclass(frozen=True)
class Params:
    """One simulator setting; defaults are the reference setting of METHODS Section 11.5."""

    base_rate: float = 0.06
    timing_lift: float = 0.10
    copy_spread: float = 0.05
    fatigue_elasticity: float = 0.4
    fatigue_half_life: float = 7.0
    novelty: float = 0.017
    novelty_half_life: float = 15.0
    opt_out: float = 0.0003
    return_boost: float = 0.03
    return_half_life: float = 1.0
    annoyance: float = 0.05


RANGES: dict[str, tuple[float, float]] = {
    "base_rate": (0.03, 0.2),
    "timing_lift": (0.0, 0.25),
    "copy_spread": (0.0, 0.2),
    "fatigue_elasticity": (0.0, 0.8),
    "fatigue_half_life": (3.0, 14.0),
    "novelty": (0.0, 0.05),
    "novelty_half_life": (7.0, 30.0),
    "opt_out": (0.0, 0.001),
    "return_boost": (0.0, 0.1),
    "return_half_life": (0.5, 3.0),
    "annoyance": (0.0, 0.1),
}
OPT_OUT_STRESS = 0.005

assert set(RANGES) == {f.name for f in fields(Params)}


def reference_volume(half_life: float) -> float:
    """Decayed volume just before a push under one push a day: sum of 0.5^(k/h), k >= 1."""
    return float(1.0 / (2.0 ** (1.0 / half_life) - 1.0))


def fatigue_rate(p: Params) -> float:
    """phi such that the open rate has elasticity -epsilon_f at one push a day."""
    return p.fatigue_elasticity / reference_volume(p.fatigue_half_life)


def annoyance_rate(p: Params) -> float:
    """delta such that organic visits have elasticity -epsilon_v at one push a day."""
    return p.annoyance / reference_volume(p.fatigue_half_life)


def timing_multipliers(profile: FloatArray, eta: float) -> FloatArray:
    """m[u, s] = a^eta / mean over slots of a^eta, so a uniformly random slot averages 1."""
    # In logs, shifted by the row maximum, so large exponents do not underflow
    z = eta * np.log(profile)
    powered = np.exp(z - z.max(axis=1, keepdims=True))
    out: FloatArray = powered / powered.mean(axis=1, keepdims=True)
    return out


def timing_lift(profile: FloatArray, eta: float) -> float:
    """Mean gain from sending at each user's true best slot over a random slot."""
    return float(timing_multipliers(profile, eta).max(axis=1).mean() - 1.0)


def solve_eta(profile: FloatArray, lift: float, tol: float = 1e-10) -> float:
    """The exponent that gives the target timing lift, by bisection (the lift grows with eta)."""
    if lift <= 0:
        return 0.0
    low, high = 0.0, 1.0
    while timing_lift(profile, high) < lift:
        high *= 2
        if high > 1e6:
            raise ValueError("the profiles cannot reach this timing lift")
    while high - low > tol:
        mid = (low + high) / 2
        low, high = (mid, high) if timing_lift(profile, mid) < lift else (low, mid)
    return (low + high) / 2


def copy_effects(
    spread: float, seed: int, templates: int = TEMPLATES, families: int = FAMILIES
) -> FloatArray:
    """c_k = exp(sigma z_k - sigma^2 / 2), z_k half family effect, half template effect."""
    rng = np.random.default_rng(seed)
    family = rng.standard_normal(families)[np.arange(templates) % families]
    z = np.sqrt(0.5) * family + np.sqrt(0.5) * rng.standard_normal(templates)
    out: FloatArray = np.exp(spread * z - spread**2 / 2)
    return out


# Purposes of keyed random streams (D032)
EXPLORE_STREAM, TEMPLATE_STREAM, COIN_STREAM, HALF_STREAM, PERTURB_STREAM = 1, 2, 3, 4, 5
# Uniforms shared by the template policies of M6 (METHODS 13.4)
CHOICE_STREAM = 6


def stream(purpose: int, *key: int) -> np.random.Generator:
    """Generator for one purpose and key. NumPy's seed sequence ignores trailing zeros, so
    ``default_rng([s, 0, 0])`` equals ``default_rng(s)``; a leading purpose and a closing 1
    keep each stream distinct from every other key and from every integer seed."""
    return np.random.default_rng([purpose, *key, 1])


QUICK_SAMPLE = 5_000


def quick_sample(users: int, size: int = QUICK_SAMPLE, seed: int = 0) -> npt.NDArray[np.int64]:
    """Sorted positions of the users kept for quick runs."""
    chosen = np.random.default_rng(seed).choice(users, min(size, users), replace=False)
    out: npt.NDArray[np.int64] = np.sort(chosen).astype(np.int64)
    return out
