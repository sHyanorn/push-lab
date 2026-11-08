"""Policy-side activity profile with a swept shift and noise (METHODS Section 12.5)."""

import numpy as np
import numpy.typing as npt

from push_lab.sim import inputs

FloatArray = npt.NDArray[np.float64]


def shift_hours(hourly: FloatArray, shift: float) -> FloatArray:
    """Move each user's hourly counts ``shift`` hours later, interpolating linearly."""
    whole = int(np.floor(shift))
    f = shift - whole
    out: FloatArray = (1 - f) * np.roll(hourly, whole, axis=1) + f * np.roll(
        hourly, whole + 1, axis=1
    )
    return out


def perturbed(hourly: FloatArray, shift: float, noise: float, seed: int) -> FloatArray:
    """Smoothed slot shares from shifted hourly counts, times exp(noise * z), renormalized."""
    profile = inputs.smoothed_profile(inputs.windows_from_hourly(shift_hours(hourly, shift)))
    if noise == 0:
        return profile
    z = np.random.default_rng(seed).standard_normal(profile.shape)
    noisy = profile * np.exp(noise * z)
    out: FloatArray = noisy / noisy.sum(axis=1, keepdims=True)
    return out
