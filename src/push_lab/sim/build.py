"""Fit the simulator's inputs and check them (METHODS Section 11).

Writes ``results/simulator.json``. It reports checks of the fitted inputs and the cost of a
run, never policy outcomes: those are Tier C and come only with the sweep (M5).
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

from push_lab.data import paths
from push_lab.rank import metrics
from push_lab.sim import env, inputs, world
from push_lab.sim.explore import EpsilonGreedy
from push_lab.sim.world import SLOTS, Params

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

LIFTS = (0.02, 0.10, 0.25)
CALIBRATION_BINS = 10
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class Settings:
    inputs: inputs.Settings = field(default_factory=inputs.Settings)
    bootstrap_samples: int = metrics.BOOTSTRAP_SAMPLES


def _round(x: float) -> float:
    return round(float(x), 6)


def _est(e: metrics.Estimate, tier: str = "A") -> dict[str, Any]:
    return {k: _round(v) if isinstance(v, float) else v for k, v in e.as_dict().items()} | {
        "tier": tier
    }


def _quantiles(values: FloatArray) -> dict[str, float]:
    return {f"p{round(q * 100)}": _round(np.quantile(values, q)) for q in QUANTILES}


def _per_user(users: IntArray, values: FloatArray) -> tuple[IntArray, FloatArray, FloatArray]:
    """Unique users, the mean of ``values`` per user and the rows per user."""
    order, inverse = np.unique(users, return_inverse=True)
    rows = np.bincount(inverse).astype(np.float64)
    return order, np.bincount(inverse, weights=values) / rows, rows


def response_checks(v: inputs.Validation, samples: int) -> dict[str, Any]:
    """Held-out log loss, AUC, GAUC and reliability, with intervals over held-out users."""
    p = np.clip(v.predicted, 1e-12, 1 - 1e-12)
    loss = -(v.labels * np.log(p) + (1 - v.labels) * np.log(1 - p))
    order, mean_loss, rows = _per_user(v.users, loss)
    idx = metrics.resample_indices(len(order), samples)
    inverse = np.searchsorted(order, v.users)
    aucs = []
    for draw in idx:
        counts = np.bincount(draw, minlength=len(order))[inverse]
        aucs.append(roc_auc_score(v.labels, v.predicted, sample_weight=counts))
    low, high = np.percentile(aucs, [2.5, 97.5])
    gauc = metrics.user_auc(v.users, v.predicted, v.labels, order)
    edges = np.quantile(v.predicted, np.linspace(0, 1, CALIBRATION_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, v.predicted, side="right")
    reliability = []
    for b in range(CALIBRATION_BINS):
        inside = bins == b
        _, observed, n = _per_user(v.users[inside], v.labels[inside])
        rate = np.full(len(order), np.nan)
        weight = np.zeros(len(order))
        pos = np.searchsorted(order, np.unique(v.users[inside]))
        rate[pos], weight[pos] = observed, n
        reliability.append(
            {
                "mean_predicted": _round(v.predicted[inside].mean()),
                "observed": _est(metrics.estimate(rate, idx, weights=weight)),
            }
        )
    return {
        "training_rows": v.rows_train,
        "validation_rows": len(v.labels),
        "validation_users": len(order),
        "best_rounds": v.best_rounds,
        "log_loss": _est(metrics.estimate(mean_loss, idx, weights=rows)),
        "auc": {
            "value": _round(roc_auc_score(v.labels, v.predicted)),
            "ci_low": _round(low),
            "ci_high": _round(high),
            "tier": "A",
        },
        "gauc": _est(metrics.estimate(gauc, idx)),
        "click_rate": _est(metrics.estimate(_per_user(v.users, v.labels)[1], idx, weights=rows)),
        "mean_predicted": _est(
            metrics.estimate(_per_user(v.users, v.predicted)[1], idx, weights=rows)
        ),
        "note": "validation users are held out from fitting but drive early stopping",
        "feature_gain_share": {k: _round(v) for k, v in v.gain_share.items()},
        "reliability_by_decile": reliability,
    }


def input_checks(fitted: inputs.Fitted, samples: int) -> dict[str, Any]:
    inp = fitted.inputs
    n = len(inp.user_ids)
    idx = metrics.resample_indices(n, samples)
    best_policy = inp.profile_policy.argmax(axis=1)
    best_sim = inp.profile_sim.argmax(axis=1)
    agree = (best_policy == best_sim).astype(np.float64)
    # Agreement expected if the two best slots were independent with these marginals
    chance = (
        float(np.bincount(best_policy, minlength=SLOTS) @ np.bincount(best_sim, minlength=SLOTS))
        / n**2
    )
    # Users with no rows in the policy period all get the same smoothed profile, the prior
    _, shared = np.unique(inp.profile_policy, axis=0, return_counts=True)
    no_early_rows = int(shared.max()) if shared.max() > 1 else 0
    reference = Params()
    hourly = fitted.hourly_sim
    return {
        "users": n,
        "visit_rate": _quantiles(inp.visit_rate) | {"mean": _round(inp.visit_rate.mean())},
        "slot_shares": {
            "simulator_profile": [_round(x) for x in inp.profile_sim.mean(axis=0)],
            "policy_profile": [_round(x) for x in inp.profile_policy.mean(axis=0)],
        },
        "best_slot_agreement_policy_vs_simulator": _est(
            metrics.estimate(agree, idx), tier="descriptive"
        )
        | {
            "chance_if_independent": _round(chance),
            "users_without_policy_period_rows": no_early_rows,
        },
        "hourly_activity_utc8": [_round(x) for x in hourly],
        "lowest_activity_hour_utc8": int(np.argmin(hourly)),
        "date_field_matches_local_date": {
            f"utc+{h}": _round(share) for h, share in fitted.date_agreement.items()
        },
        "timing_eta": {str(lift): _round(world.solve_eta(inp.profile_sim, lift)) for lift in LIFTS},
        "reference": asdict(reference)
        | {
            "reference_volume": _round(world.reference_volume(reference.fatigue_half_life)),
            "fatigue_rate": _round(world.fatigue_rate(reference)),
            "annoyance_rate": _round(world.annoyance_rate(reference)),
        },
        "ranges": {k: list(v) for k, v in world.RANGES.items()}
        | {"opt_out_stress": world.OPT_OUT_STRESS},
    }


def run(root: Path, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or Settings()
    start = time.perf_counter()
    fitted = inputs.build(root, s.inputs)
    timings = fitted.timings | {"inputs_s": round(time.perf_counter() - start, 1)}
    fitted.inputs.save(paths.processed_dir(root, "sim") / inputs.CACHE)

    inp = fitted.inputs
    explorer = EpsilonGreedy(inp.profile_policy.argmax(axis=1))
    pop = env.Population(inp.visit_rate, inp.profile_sim, inp.lists["cascade"].p0)
    start = time.perf_counter()
    env.simulate(pop, Params(), explorer, env.draws(pop.users, seed=0))
    timings["one_run_all_users_s"] = round(time.perf_counter() - start, 1)

    return {
        "response_model": response_checks(fitted.validation, s.bootstrap_samples),
        "inputs": input_checks(fitted, s.bootstrap_samples),
        "exploration": {
            "epsilon": explorer.epsilon,
            "q": explorer.q,
            "templates": explorer.templates,
            "send_probability_best_slot": _round(
                1 - explorer.epsilon + explorer.epsilon * explorer.q
            ),
            "send_probability_other_slots": _round(explorer.epsilon * explorer.q),
            "min_propensity": _round(explorer.min_propensity),
        },
        "quick_sample_users": len(world.quick_sample(pop.users)),
        "timings_s": timings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and check the simulator's inputs.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    result = run(args.root)
    out = args.root / paths.RESULTS / "simulator.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
