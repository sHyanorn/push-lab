"""Evidence matrix: every headline claim with its tier, source and interval (METHODS 14.5).

``python -m push_lab.report.evidence`` reads the committed results and writes
``results/evidence.json``. Numbers are copied from the results files, never typed in.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from push_lab.data import paths
from push_lab.runtime import write_json

TIERS = {
    "A": {
        "evidence": "Randomized real data: KuaiRand random exposures, Hillstrom, Criteo, "
        "Open Bandit Dataset",
        "supports": "A component ranks or targets better in that domain; on the Open Bandit "
        "Dataset, how close an off-policy estimator comes to a known policy value",
        "cannot": "Anything about push, fatigue or long-term effects",
    },
    "B": {
        "evidence": "Off-policy estimate on real logs from an estimator first validated "
        "against a known answer",
        "supports": "The value of a new policy within the support of the logging policy",
        "cannot": "Policies outside that support; long horizons",
    },
    "C": {
        "evidence": "Simulator runs with a sensitivity sweep",
        "supports": "Policy A beats B across a stated share of assumptions, or flips at a "
        "stated threshold",
        "cannot": "Real-world effect sizes",
    },
    "history": {
        "evidence": "A registered verdict later overturned by a larger check",
        "supports": "Nothing beyond the record of the verdict",
        "cannot": "Any claim",
    },
    "descriptive": {
        "evidence": "Counts of a fixed set, no inference",
        "supports": "What the set contains",
        "cannot": "Any effect",
    },
}

Status = Literal["pending", "reported", "verdict"]


class Source(TypedDict):
    file: str
    path: list[str]


class Interval(TypedDict):
    value: float
    ci_low: float
    ci_high: float


class ClaimEntry(TypedDict, total=False):
    """A row of the matrix. Every row has the first five fields; the others depend on the
    claim, and a pending or verdict row has none of them."""

    id: str
    tier: str
    statement: str
    source: Source
    status: Status
    value: float
    ci_low: float
    ci_high: float
    # Tested comparisons; None for a sweep that has not finished
    holm_rejected: bool | None
    # Sweep comparisons
    label: str
    wins: int
    losses: int
    neither: int
    spread_over_bootstrap_se: float
    spread_source: str
    # Generated copy
    returned: int
    passed: int
    kept: int
    # Simulator off-policy checks
    sweep_validated: str
    truth: Interval
    reversed_settings: str
    # OBD, IPS only
    post_hoc_empirical_target: Interval
    post_hoc_difference: Interval


def _interval(found: Any) -> ClaimEntry:
    return {"value": found["value"], "ci_low": found["ci_low"], "ci_high": found["ci_high"]}


def _tested(found: Any) -> ClaimEntry:
    return {**_interval(found), "holm_rejected": found["holm_rejected"]}


def _sweep(found: Any) -> ClaimEntry:
    """The sweep average of a comparison with its label and per-setting verdict counts."""
    return {
        **_interval(found["sweep_average"]),
        "label": found["label"],
        "holm_rejected": found["holm_rejected"],
        "wins": found["wins"],
        "losses": found["losses"],
        "neither": found["neither"],
    }


def _crossing(found: Any) -> ClaimEntry:
    """Where a comparison changes sign along a grid."""
    return {"value": found["at"], "ci_low": found["ci_low"], "ci_high": found["ci_high"]}


def _copy_counts(found: Any) -> ClaimEntry:
    """Totals over the copy families."""
    return {
        "returned": sum(c["returned"] for c in found.values()),
        "passed": sum(c["passed"] for c in found.values()),
        "kept": sum(c["kept"] for c in found.values()),
    }


@dataclass(frozen=True)
class Claim:
    """A claim whose numbers sit at one path of one results file; ``read`` names their layout."""

    id: str
    tier: str
    statement: str
    file: str
    path: tuple[str, ...]
    read: Callable[[Any], ClaimEntry]


CLAIMS = (
    Claim(
        "C1",
        "A",
        "Two-tower retrieval recalls more of a user's clicked random-exposure items in the top "
        "200 than popularity (difference in Recall@200)",
        "ranking.json",
        ("primary", "C1_two_tower_vs_popularity_recall200"),
        _tested,
    ),
    Claim(
        "C2",
        "A",
        "The fused fine ranker ranks random exposures worse than the two-tower score alone "
        "(difference in click GAUC)",
        "ranking.json",
        ("primary", "C2_fused_vs_two_tower_gauc_click"),
        _tested,
    ),
    Claim(
        "R1",
        "A",
        "The simulator's response model ranks random exposures of users held out from fitting "
        "(AUC; these users also set the stopping round, D058)",
        "simulator.json",
        ("response_model", "auc"),
        _interval,
    ),
    Claim(
        "U1",
        "A",
        "On Hillstrom, the T-learner targets visits better than random (Qini coefficient)",
        "uplift.json",
        ("primary", "U1_hillstrom_t_learner_qini"),
        _tested,
    ),
    Claim(
        "U2",
        "A",
        "On Hillstrom, the T-learner against the response model (difference in Qini)",
        "uplift.json",
        ("primary", "U2_hillstrom_t_learner_vs_response_qini"),
        _tested,
    ),
    Claim(
        "U3",
        "A",
        "On Criteo, the T-learner targets visits better than random (Qini coefficient)",
        "uplift.json",
        ("primary", "U3_criteo_t_learner_qini"),
        _tested,
    ),
    Claim(
        "U4",
        "A",
        "On Criteo, the T-learner against the response model (difference in Qini)",
        "uplift.json",
        ("primary", "U4_criteo_t_learner_vs_response_qini"),
        _tested,
    ),
    *(
        Claim(
            f"{h}{'' if sweep == 'main' else '-af'}",
            "C",
            f"{text} ({'main' if sweep == 'main' else 'annoyance-free'} sweep, difference in "
            "active days per user over 28 days)",
            "policy.json",
            ("primary", sweep, h),
            _sweep,
        )
        for sweep in ("main", "annoyance_free")
        for h, text in (
            ("H1", "Uplift gate G against response threshold S at equal volume"),
            ("H2", "Decomposed gate G2 against S at equal volume"),
            ("H3", "Gate G at half volume against one push a day at the best slot"),
            ("H4", "Each user's best slot against one fixed slot"),
        )
    ),
    # One row per grid along which F-N flips; check_readings ties the rows to the flips
    Claim(
        "F-N",
        "C",
        "With the other parameters at the reference, one push a day gains active days over no "
        "pushes only below this annoyance elasticity (one-at-a-time grid)",
        "policy.json",
        ("flips", "annoyance", "F-N", "crossings", "0"),
        _crossing,
    ),
    Claim(
        "F-N-base",
        "C",
        "With the other parameters at the reference, annoyance included, one push a day gains "
        "active days over no pushes only above this base-rate factor (one-at-a-time grid)",
        "policy.json",
        ("flips", "base_rate", "F-N", "crossings", "0"),
        _crossing,
    ),
    *(
        Claim(
            f"OPE-flip-{target}-DR",
            "C",
            f"In the simulator, over annoyance elasticities 0 to 0.1 with the other parameters at "
            f"the reference, per-day DR overestimates {text} below this value and underestimates "
            "it above; fatigue, opt-out, return and novelty change no direction over their "
            "ranges",
            "ope_carryover.json",
            ("flips", "annoyance", f"{target}.DR", "crossings", "0"),
            _crossing,
        )
        for target, text in (("T_half", "half the push volume"), ("N", "no pushes"))
    ),
    Claim(
        "B1",
        "C",
        "Recovering difference softmax against uniform templates (difference in opens per "
        "user over 28 days)",
        "templates.json",
        ("primary", "B1"),
        _sweep,
    ),
    Claim(
        "B2",
        "C",
        "Recency penalty against none, same learned scores (difference in opens per user over "
        "28 days)",
        "templates.json",
        ("primary", "B2"),
        _sweep,
    ),
    Claim(
        "copy",
        "descriptive",
        "Generated variants that passed every filter, of those returned",
        "copy.json",
        ("counts",),
        _copy_counts,
    ),
)


def _load(root: Path, file: str) -> Any:
    """A results file, or None before it is written."""
    path = root / paths.RESULTS / file
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def _get(data: Any, path: Sequence[str]) -> Any:
    for key in path:
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def _entry(
    claim_id: str, tier: str, statement: str, file: str, path: Sequence[str], status: Status
) -> ClaimEntry:
    return {
        "id": claim_id,
        "tier": tier,
        "statement": statement,
        "source": {"file": f"results/{file}", "path": list(path)},
        "status": status,
    }


def claim_entry(root: Path, claim: Claim) -> ClaimEntry:
    header = (claim.id, claim.tier, claim.statement, claim.file, claim.path)
    data = _load(root, claim.file)
    if data is None:
        return _entry(*header, "pending")
    numbers = claim.read(_get(data, claim.path))
    return _entry(*header, "reported") | numbers | _spread(data, claim)


def _comparison(claim_id: str) -> str:
    """The registered comparison of a claim: H1 for both H1 and H1-af, whose two sweeps share
    the refits at the reference."""
    return claim_id.split("-")[0]


def _spread(data: Any, claim: Claim) -> ClaimEntry:
    """How much wider the spread over refits or seed triples is than a per-setting interval,
    wherever the results measured it."""
    key = _comparison(claim.id)
    if claim.file == "templates.json":
        seeds = data["seed_variability"][key]
        return {
            "spread_over_bootstrap_se": seeds["ratio"],
            "spread_source": f"{seeds['triples']} seed triples at the reference setting",
        }
    fits = data.get("fit_variability", {}).get(key) if claim.file == "policy.json" else None
    if fits and fits["sd_over_refits"] > 0:
        ratio = fits["sd_over_refits"] / fits["mean_bootstrap_se"]
        return {
            "spread_over_bootstrap_se": round(ratio, 2),
            "spread_source": f"{fits['refits']} model refits at the reference setting",
        }
    return {}


def ope_entries(root: Path) -> list[ClaimEntry]:
    """The OBD estimator checks, the simulator verdicts and the post hoc orderings."""
    out: list[ClaimEntry] = []
    carryover = _load(root, "ope_carryover.json")
    obd = _load(root, "ope_obd.json")
    if obd is not None:
        out += _obd_entries(obd)
    if (root / paths.RESULTS / "ope_simulator.json").exists():
        out.append(
            _entry(
                "OPE-first-run",
                "C",
                "First simulator check (D043): registered verdict 'implementation "
                "failure', since IPS and DR failed the no-carryover control for T-half and N; "
                "post hoc, against exact truths computed afterwards, the biases contain 0, "
                "consistent with a chance false alarm among 12 unadjusted tests",
                "ope_simulator.json",
                ("control",),
                "verdict",
            )
        )
    amended = _load(root, "ope_simulator_amended.json")
    if amended is not None:
        for setting in ("control", "reference"):
            for target, rewards in amended[setting].items():
                for name in ("DR", "IPS"):
                    check = rewards["visit"][name]
                    out.append(
                        _entry(
                            f"OPE-{setting}-{target}-{name}",
                            "C",
                            f"{name} bias against simulator truth for target "
                            f"{target}, {setting} setting (active-day share): "
                            f"{check['verdict']}",
                            "ope_simulator_amended.json",
                            (setting, target, "visit", name),
                            "reported",
                        )
                        | _interval(check["bias"])
                        | _sweep_share(carryover, setting, target, name)
                    )
    out += _ordering_entries(root, carryover)
    return out


def _sweep_share(data: Any, setting: str, target: str, name: str) -> ClaimEntry:
    """For the reference setting, the share of the sweep settings where the same estimator and
    target are validated."""
    if setting != "reference" or data is None:
        return {}
    verdicts = [s["targets"][target][name]["verdict"] for s in data["settings"]]
    return {"sweep_validated": f"{verdicts.count('validated')} of {len(verdicts)} settings"}


def _ordering_entries(root: Path, data: Any) -> list[ClaimEntry]:
    """Post hoc: at the reference annoyance of the policy design, the per-day DR estimate of a
    difference between targets against its truth, and how many sweep settings reverse the
    ordering."""
    if data is None or "post_hoc" not in data:
        return []
    out: list[ClaimEntry] = []
    orderings = data["post_hoc"]["orderings"]
    design = json.loads((root / paths.RESULTS / "policy_design.json").read_text("utf-8"))
    reference = orderings["annoyance"]["values"].index(design["reference"]["annoyance"])
    for pair, text in (("N-T", "no pushes minus one a day"), ("T2-T", "two a day minus one")):
        found = orderings["annoyance"][pair]
        reversals = [
            any(orderings[grid][pair]["reversed"]["DR"][point] for grid, point in setting["grid"])
            for setting in data["settings"]
        ]
        out.append(
            {
                **_entry(
                    f"OPE-order-{pair}-DR",
                    "C",
                    f"Post hoc: per-day DR estimate of {text} (active-day share) at the "
                    "reference, against the simulator truth; an ordering is reversed where both "
                    "intervals exclude 0 with opposite signs",
                    "ope_carryover.json",
                    ("post_hoc", "orderings", "annoyance", pair, "DR", str(reference)),
                    "reported",
                ),
                "truth": found["truth"][reference],
                "reversed_settings": f"{sum(reversals)} of {len(reversals)}",
                **_interval(found["DR"][reference]),
            }
        )
    return out


def _obd_entries(data: Any) -> list[ClaimEntry]:
    """The real-data result first: on the full campaign, how far IPS and DR fall from the
    on-policy value; then each registered small-sample estimate with its outcome on both."""
    out: list[ClaimEntry] = []
    small, full = data.get("small"), data.get("full")
    if isinstance(full, dict):
        for name in ("IPS", "DR"):
            ratio = full["validation"][name].get("ratio_to_on_policy")
            if ratio is None:
                continue
            post_hoc = _empirical_target(full, name)
            note = (
                " (post_hoc_empirical_target re-weights to the BTS logs' own frequencies; "
                "post_hoc_difference is its paired change from this ratio, with an interval "
                "that includes 0)"
            )
            out.append(
                _entry(
                    f"OBD-ratio-{name}",
                    # Randomized real data measuring the estimator itself, not a policy value
                    "A",
                    f"The estimator {name}, on real randomized logs (full Men's "
                    "campaign of OBD), estimates the BTS click rate at this fraction of its "
                    "on-policy value. The target is BTS "
                    "with its fixed prior, which the deployed BTS was not, so the shortfall may "
                    "come from the estimator or from that mismatch, and these data do not "
                    f"separate the two{note if post_hoc else ''}. Post hoc reading: the small "
                    "sample has too little power to show the shortfall",
                    "ope_obd.json",
                    ("full", "validation", name),
                    "reported",
                )
                | _interval(ratio)
                | post_hoc
            )
    if isinstance(small, dict):
        for name, check in small["validation"].items():
            on_full = full["validation"][name]["validated"] if isinstance(full, dict) else None
            outcome = (
                ("passes" if check["validated"] else "fails")
                + " on the small sample"
                + ("" if on_full is None else f", {'passes' if on_full else 'fails'} on the full")
            )
            out.append(
                _entry(
                    f"OBD-small-{name}",
                    # Registered verdict history only: no estimator passed on the full campaign
                    "history",
                    f"Registered verdict on the BTS click rate estimated by {name} "
                    f"from the small sample's random logs ({outcome}); not a Tier B result, and "
                    "its numbers are not reported here",
                    "ope_obd.json",
                    ("small", "estimates", name),
                    "verdict",
                )
            )
    return out


def _empirical_target(full: dict[str, Any], name: str) -> ClaimEntry:
    """Post hoc IPS ratio with the BTS logs' own frequencies as the target and its paired
    difference from the registered ratio, where computed."""
    post_hoc = full.get("post_hoc")
    if name != "IPS" or not isinstance(post_hoc, dict):
        return {}
    return {
        "post_hoc_empirical_target": post_hoc["ratio_to_on_policy"],
        "post_hoc_difference": post_hoc["difference_from_fixed_prior"],
    }


NO_TIER_B = (
    "The project has no Tier B result: no estimator passed the check on the full Men's "
    "campaign of OBD, the only real logs with two policies and known propensities here."
)


def build(root: Path) -> dict[str, Any]:
    return {
        "tiers": TIERS,
        "note": NO_TIER_B,
        "claims": [claim_entry(root, c) for c in CLAIMS] + ope_entries(root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the evidence matrix.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    target = args.root / paths.RESULTS / "evidence.json"
    write_json(target, build(args.root))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
