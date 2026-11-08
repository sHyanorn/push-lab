"""Evidence matrix: every headline claim with its tier, source and interval (METHODS 14.5).

``python -m push_lab.report.evidence`` reads the committed results and writes
``results/evidence.json``. Numbers are copied from the results files, never typed in.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from push_lab.data import paths

TIERS = {
    "A": {
        "evidence": "Randomized real data: KuaiRand random exposures, Hillstrom, Criteo",
        "supports": "A component ranks or targets better in that domain",
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


@dataclass(frozen=True)
class Claim:
    id: str
    tier: str
    statement: str
    file: str
    path: tuple[str, ...]


CLAIMS = (
    Claim(
        "C1",
        "A",
        "Two-tower retrieval recalls more of a user's clicked random-exposure items in the top "
        "200 than popularity (difference in Recall@200)",
        "ranking.json",
        ("primary", "C1_two_tower_vs_popularity_recall200"),
    ),
    Claim(
        "C2",
        "A",
        "The fused fine ranker ranks random exposures worse than the two-tower score alone "
        "(difference in click GAUC)",
        "ranking.json",
        ("primary", "C2_fused_vs_two_tower_gauc_click"),
    ),
    Claim(
        "R1",
        "A",
        "The simulator's response model ranks held-out random exposures (AUC)",
        "simulator.json",
        ("response_model", "auc"),
    ),
    Claim(
        "U1",
        "A",
        "On Hillstrom, the T-learner targets visits better than random (Qini coefficient)",
        "uplift.json",
        ("primary", "U1_hillstrom_t_learner_qini"),
    ),
    Claim(
        "U2",
        "A",
        "On Hillstrom, the T-learner against the response model (difference in Qini)",
        "uplift.json",
        ("primary", "U2_hillstrom_t_learner_vs_response_qini"),
    ),
    Claim(
        "U3",
        "A",
        "On Criteo, the T-learner targets visits better than random (Qini coefficient)",
        "uplift.json",
        ("primary", "U3_criteo_t_learner_qini"),
    ),
    Claim(
        "U4",
        "A",
        "On Criteo, the T-learner against the response model (difference in Qini)",
        "uplift.json",
        ("primary", "U4_criteo_t_learner_vs_response_qini"),
    ),
    *(
        Claim(
            f"{h}{'' if sweep == 'main' else '-af'}",
            "C",
            f"{text} ({'main' if sweep == 'main' else 'annoyance-free'} sweep, difference in "
            "active days per user over 28 days)",
            "policy.json",
            ("primary", sweep, h),
        )
        for sweep in ("main", "annoyance_free")
        for h, text in (
            ("H1", "Uplift gate G against response threshold S at equal volume"),
            ("H2", "Decomposed gate G2 against S at equal volume"),
            ("H3", "Gate G at half volume against one push a day at the best slot"),
            ("H4", "Each user's best slot against one fixed slot"),
        )
    ),
    Claim(
        "F-N",
        "C",
        "One push a day gains active days over no pushes only below this annoyance elasticity "
        "(one-at-a-time grid)",
        "policy.json",
        ("flips", "annoyance", "F-N", "crossings", "0"),
    ),
    *(
        Claim(
            f"OPE-flip-{target}-DR",
            "C",
            f"Per-day DR overestimates {text} below this annoyance elasticity and underestimates "
            "it above (one-at-a-time grid; fatigue, opt-out, return and novelty change no "
            "direction)",
            "ope_carryover.json",
            ("flips", "annoyance", f"{target}.DR", "crossings", "0"),
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
    ),
    Claim(
        "B2",
        "C",
        "Recency penalty against none, same learned scores (difference in opens per user over "
        "28 days)",
        "templates.json",
        ("primary", "B2"),
    ),
    Claim(
        "copy",
        "descriptive",
        "Generated variants that passed every filter, of those returned",
        "copy.json",
        ("counts",),
    ),
)


def _get(data: Any, path: Sequence[str]) -> Any:
    for key in path:
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def _interval(found: Any) -> dict[str, Any]:
    """The estimate and interval of a results entry, whichever layout it uses."""
    entry = found.get("sweep_average", found) if isinstance(found, dict) else found
    keys = ("value", "ci_low", "ci_high") if "value" in entry else ("at", "ci_low", "ci_high")
    out = {"value": entry[keys[0]], "ci_low": entry["ci_low"], "ci_high": entry["ci_high"]}
    for extra in ("label", "holm_rejected", "wins", "losses", "neither"):
        if extra in found:
            out[extra] = found[extra]
    return out


def _copy_counts(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "returned": sum(c["returned"] for c in counts.values()),
        "passed": sum(c["passed"] for c in counts.values()),
        "kept": sum(c["kept"] for c in counts.values()),
    }


def claim_entry(root: Path, claim: Claim) -> dict[str, Any]:
    source = root / paths.RESULTS / claim.file
    entry: dict[str, Any] = {
        "id": claim.id,
        "tier": claim.tier,
        "statement": claim.statement,
        "source": {"file": f"results/{claim.file}", "path": list(claim.path)},
    }
    if not source.exists():
        return entry | {"status": "pending"}
    data = json.loads(source.read_text("utf-8"))
    found = _get(data, claim.path)
    result = _copy_counts(found) if claim.id == "copy" else _interval(found)
    return entry | {"status": "reported"} | result | _randomness(data, claim)


def _randomness(data: dict[str, Any], claim: Claim) -> dict[str, Any]:
    """How much wider the spread over refits or seed triples is than a per-setting interval,
    wherever the results measured it; win and loss counts are read with it."""
    key = claim.id.split("-")[0]
    if claim.file == "templates.json":
        v = data["seed_variability"][key]
        source = "10 seed triples at the reference setting"
        return {"spread_over_bootstrap_se": v["ratio"], "spread_source": source}
    variability = data.get("fit_variability", {}).get(key) if claim.file == "policy.json" else None
    if variability and variability["sd_over_refits"] > 0:
        ratio = variability["sd_over_refits"] / variability["mean_bootstrap_se"]
        source = "10 model refits at the reference setting"
        return {"spread_over_bootstrap_se": round(ratio, 2), "spread_source": source}
    return {}


def ope_entries(root: Path) -> list[dict[str, Any]]:
    """The OBD estimator checks, the simulator verdicts and the post hoc orderings."""
    out: list[dict[str, Any]] = []
    obd_file = root / paths.RESULTS / "ope_obd.json"
    if obd_file.exists():
        out += _obd_entries(json.loads(obd_file.read_text("utf-8")))
    first = root / paths.RESULTS / "ope_simulator.json"
    if first.exists():
        out.append(
            {
                "id": "OPE-first-run",
                "tier": "C",
                "statement": "First simulator check (D043): registered verdict 'implementation "
                "failure', since IPS and DR failed the no-carryover control for T-half and N; "
                "against exact truths computed afterwards the biases contain 0, consistent with a "
                "chance false alarm among 12 unadjusted tests",
                "source": {"file": "results/ope_simulator.json", "path": ["control"]},
                "status": "verdict",
            }
        )
    sim_file = root / paths.RESULTS / "ope_simulator_amended.json"
    if sim_file.exists():
        data = json.loads(sim_file.read_text("utf-8"))
        for setting in ("control", "reference"):
            for target, rewards in data[setting].items():
                for name in ("DR", "IPS"):
                    check = rewards["visit"][name]
                    out.append(
                        {
                            "id": f"OPE-{setting}-{target}-{name}",
                            "tier": "C",
                            "statement": f"{name} bias against simulator truth for target "
                            f"{target}, {setting} setting (active-day share): "
                            f"{check['verdict']}",
                            "source": {
                                "file": "results/ope_simulator_amended.json",
                                "path": [setting, target, "visit", name],
                            },
                            "status": "reported",
                        }
                        | check["bias"]
                        | _sweep_share(root, setting, target, name)
                    )
    out += _ordering_entries(root)
    return out


def _carryover(root: Path) -> dict[str, Any] | None:
    path = root / paths.RESULTS / "ope_carryover.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def _sweep_share(root: Path, setting: str, target: str, name: str) -> dict[str, Any]:
    """For the reference setting, the share of the 24 sweep settings where the same estimator
    and target are validated."""
    data = _carryover(root)
    if setting != "reference" or data is None:
        return {}
    verdicts = [s["targets"][target][name]["verdict"] for s in data["settings"]]
    return {"sweep_validated": f"{verdicts.count('validated')} of {len(verdicts)} settings"}


def _ordering_entries(root: Path) -> list[dict[str, Any]]:
    """Post hoc: at the reference, the per-day DR estimate of a difference between targets
    against its truth, and how many sweep settings reverse the ordering."""
    data = _carryover(root)
    if data is None or "post_hoc" not in data:
        return []
    out: list[dict[str, Any]] = []
    grid = data["post_hoc"]["orderings"]["annoyance"]
    reference = grid["values"].index(0.05)
    for pair, text in (("N-T", "no pushes minus one a day"), ("T2-T", "two a day minus one")):
        found = grid[pair]
        reversed_at = {
            sid
            for name, entry in data["post_hoc"]["orderings"].items()
            for sid, s in enumerate(data["settings"])
            for g in s["grid"]
            if g[0] == name and entry[pair]["reversed"]["DR"][g[1]]
        }
        out.append(
            {
                "id": f"OPE-order-{pair}-DR",
                "tier": "C",
                "statement": f"Post hoc: per-day DR estimate of {text} (active-day share) at the "
                "reference, against the simulator truth; an ordering is reversed where both "
                "intervals exclude 0 with opposite signs",
                "source": {
                    "file": "results/ope_carryover.json",
                    "path": ["post_hoc", "orderings", "annoyance", pair, "DR", str(reference)],
                },
                "status": "reported",
                "truth": found["truth"][reference],
                "reversed_settings": f"{len(reversed_at)} of {len(data['settings'])}",
            }
            | found["DR"][reference]
        )
    return out


def _obd_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The real-data result first: on the full campaign, how far IPS and DR fall from the
    on-policy value; then each registered small-sample estimate with its outcome on both."""
    out: list[dict[str, Any]] = []
    small, full = data.get("small"), data.get("full")
    source = "results/ope_obd.json"
    if isinstance(full, dict):
        for name in ("IPS", "DR"):
            ratio = full["validation"][name].get("ratio_to_on_policy")
            if ratio is None:
                continue
            out.append(
                {
                    "id": f"OBD-ratio-{name}",
                    # Randomized real data measuring the estimator itself, not a policy value
                    "tier": "A",
                    "statement": f"The estimator {name}, on real randomized logs (full Men's "
                    "campaign of OBD), estimates the BTS click rate at this fraction of its "
                    "on-policy value. The target is BTS "
                    "with its fixed prior, which the deployed BTS was not, so the shortfall mixes "
                    "estimator error with that mismatch. Post hoc reading: the small sample has "
                    "too little power to show it",
                    "source": {"file": source, "path": ["full", "validation", name]},
                    "status": "reported",
                }
                | ratio
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
                {
                    "id": f"OBD-small-{name}",
                    # Registered verdict history only: no estimator passed on the full campaign
                    "tier": "history",
                    "statement": f"BTS click rate estimated by {name} from the small sample's "
                    f"random logs ({outcome}); not a Tier B result; on-policy value "
                    f"{small['on_policy']['value']:.5f}",
                    "source": {"file": source, "path": ["small", "estimates", name]},
                    "status": "reported",
                }
                | small["estimates"][name]
            )
    return out


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
    target.write_text(json.dumps(build(args.root), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
