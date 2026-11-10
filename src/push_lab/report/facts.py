"""What the report states, read from the committed results (D049).

The values the prose quotes, the readings it puts in words, and the rows and series of the
tables and charts. Every number on the page or in the README comes from here; the prose lives
in ``text.py``.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from push_lab.data import paths
from push_lab.report import svg, text
from push_lab.report.evidence import ClaimEntry
from push_lab.report.svg import MINUS, num

FILES = (
    *("evidence", "policy", "policy_design", "templates", "ope_carryover", "copy"),
    "ope_obd_blocks",
)
OBD_ESTIMATORS = ("DM", "IPS", "SNIPS", "DR")


# Loading and formatting ---------------------------------------------------------------


def load(root: Path) -> dict[str, Any]:
    return {
        name: json.loads((root / paths.RESULTS / f"{name}.json").read_text("utf-8"))
        for name in FILES
    }


def claims(results: dict[str, Any]) -> dict[str, ClaimEntry]:
    return {c["id"]: c for c in results["evidence"]["claims"]}


def interval(entry: Mapping[str, Any], decimals: int, signed: bool = False) -> str:
    v = num(entry["value"], decimals, signed)
    lo, hi = num(entry["ci_low"], decimals), num(entry["ci_high"], decimals)
    return f"{v} ({lo} to {hi})"


def ends(entry: Mapping[str, Any], decimals: int) -> str:
    """The two ends of an interval, without its estimate."""
    return f"{num(entry['ci_low'], decimals)} to {num(entry['ci_high'], decimals)}"


def pct_interval(entry: Mapping[str, Any], decimals: int) -> str:
    scaled = {k: entry[k] * 100 for k in ("value", "ci_low", "ci_high")}
    v, lo, hi = (num(scaled[k], decimals) + "%" for k in ("value", "ci_low", "ci_high"))
    return f"{v} ({lo} to {hi})"


def sig_decimals(x: float, digits: int = 3) -> int:
    if x == 0:
        return digits
    return max(0, digits - 1 - int(f"{abs(x):e}".split("e")[1]))


def auto_interval(entry: Mapping[str, Any]) -> str:
    return interval(entry, sig_decimals(entry["value"]))


def short(x: float) -> str:
    """A number without trailing zeros, as grid values and bounds are written."""
    return f"{x:g}".replace("-", MINUS)


def same(values: Iterable[str]) -> str:
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(f"expected one value, got {sorted(distinct)}")
    return distinct.pop()


def crossing(flip: dict[str, Any], decimals: int) -> str:
    """The single resolved crossing of a flip entry."""
    (only,) = flip["crossings"]
    if not only["resolved"]:
        raise ValueError("crossing not resolved")
    return interval(only | {"value": only["at"]}, decimals)


# Registered labels that a post hoc sensitivity reads more weakly once the refit or seed
# spread at the reference is added to each setting's interval (D057): id -> (label, reading)
SENSITIVITY = {
    "B1": ("beats in most sampled settings", "better on average"),
    "H1-af": ("loses in most sampled settings", "worse on average"),
    "H2-af": ("beats in most sampled settings", "better on average"),
}


def shown_label(claim: ClaimEntry) -> str:
    """The registered label, followed by the post hoc reading of D057 where one applies."""
    label = str(claim.get("label", ""))
    if claim["id"] in SENSITIVITY:
        reading = SENSITIVITY[claim["id"]][1]
        label += f"; read as {reading} with refit or seed spread (D057, post hoc)"
    return label


def check_readings(results: dict[str, Any]) -> None:
    """Readings the prose states in words; a rerun that changes one fails the build."""
    c = claims(results)
    policy = results["policy"]
    labels = {
        "H1": "worse on average",
        "H2": "beats in most sampled settings",
        "H4": "no detectable difference",
        "H4-af": "worse on average",
        "B2": "depends on the setting",
        # D057 readings were written for these registered labels
        **{key: label for key, (label, _) in SENSITIVITY.items()},
    }
    for key, label in labels.items():
        if c[key]["label"] != label:
            raise ValueError(f"{key} is now labeled {c[key]['label']!r}")
    if c["B2"]["holm_rejected"] or not c["B2"]["ci_low"] < 0 < c["B2"]["ci_high"]:
        raise ValueError("B2 is no longer undetectable")
    if c["C2"]["ci_high"] >= 0 or c["U4"]["ci_high"] >= 0:
        raise ValueError("C2 or U4 no longer goes against expectation")
    if any(c[f"OBD-ratio-{e}"]["ci_high"] >= 1 for e in ("IPS", "DR")):
        raise ValueError("an OBD interval now reaches the on-policy value")
    change = c["OBD-ratio-IPS"]["post_hoc_difference"]
    if not change["ci_low"] < 0 < change["ci_high"]:
        raise ValueError("the OBD post hoc change no longer includes 0")
    blocks = results["ope_obd_blocks"]["full"]["validation"]
    for check in ("interval_contains_on_policy", "validated"):
        holds = {e for e in OBD_ESTIMATORS if blocks[e][check]}
        if holds != {"IPS", "SNIPS", "DR"}:
            raise ValueError(f"with blocks, {check} now holds for {sorted(holds)} on OBD")
    reference_b1 = results["templates"]["seed_variability"]["B1"]["mean_over_triples"]
    if not reference_b1["ci_low"] < 0 < reference_b1["ci_high"]:
        raise ValueError("B1 at the reference is no longer undetectable")
    fn_flips = {n for n, f in policy["flips"].items() if f["F-N"]["result"] == "flip"}
    if fn_flips != {"annoyance", "base_rate"}:
        raise ValueError(f"F-N now flips along {sorted(fn_flips)}")
    resolved = {
        (n, h)
        for n, f in policy["flips"].items()
        for h in ("H1", "H2", "H3", "H4")
        if any(x["resolved"] for x in f[h]["crossings"])
    }
    if resolved != {("annoyance", "H3")}:
        raise ValueError(f"resolved crossings of H1 to H4 are now {sorted(resolved)}")
    band = policy["band_F_over_N"]["main_sweep_counts"]
    if band["below"] * 2 <= sum(band.values()):
        raise ValueError("most settings are no longer below the band")
    carry = results["ope_carryover"]["flips"]
    for name in ("fatigue_elasticity", "opt_out", "return_boost", "novelty"):
        if any(carry[name][k]["result"] == "flip" for k in carry[name] if k != "values"):
            raise ValueError(f"the OPE bias now flips along {name}")


# Facts: every number the prose uses -------------------------------------------------------


def reference_setting(results: dict[str, Any]) -> dict[str, Any]:
    """The registered reference setting: the grid point at the design's reference values.
    The refits share these values but use other exploration seeds (METHODS 12.6)."""
    reference = results["policy_design"]["reference"]
    for s in results["policy"]["settings"]:
        if s["sweep"] == "grid" and s["values"] == reference:
            return dict(s)
    raise ValueError("no grid setting at the reference values")


def facts(results: dict[str, Any]) -> dict[str, str]:
    """The value of every placeholder in ``text.py``."""
    c = claims(results)
    policy, carry, copy = results["policy"], results["ope_carryover"], results["copy"]
    grid = carry["flips"]["annoyance"]["values"]
    orderings = carry["post_hoc"]["orderings"]["annoyance"]["N-T"]
    dr_values = [p["value"] for p in orderings["DR"]]
    truth = orderings["truth"]
    band = policy["band_F_over_N"]
    rating = copy["rating"]
    blocks = results["ope_obd_blocks"]["full"]
    ratios = {e: blocks["validation"][e]["ratio_to_on_policy"] for e in OBD_ESTIMATORS}
    streams = blocks["other_streams"]
    seeds = results["templates"]["seed_variability"]
    n_main = sum(s["sweep"] == "main" for s in policy["settings"])
    reference = reference_setting(results)
    ref = reference["policies"]
    return {
        "no_tier_b": results["evidence"]["note"],
        "crossing_note": text.CROSSING_NOTE,
        "h2_note": text.H2_NOTE,
        "ope_grids": str(len(carry["post_hoc"]["orderings"])),
        "opens_w1_t": interval(ref["T"]["opens_week1"], 3),
        "opens_w1_g": interval(ref["G"]["opens_week1"], 3),
        "active_w4_f": interval(ref["F"]["active_week4"], 2),
        "active_w4_n": interval(ref["N"]["active_week4"], 2),
        "h3_losses": str(c["H3"]["losses"]),
        "h4_half": num(policy["primary"]["main"]["H4"]["median_half_width"], 3),
        "ope_range": f"{short(grid[0])} to {short(grid[-1])}",
        "ope_settings": str(len(carry["settings"])),
        "ref_annoyance": short(results["policy_design"]["reference"]["annoyance"]),
        "ope_nt_truth": interval(c["OPE-order-N-T-DR"]["truth"], 4, signed=True),
        "ope_nt_dr": interval(c["OPE-order-N-T-DR"], 4, signed=True),
        "bias_n": interval(c["OPE-reference-N-DR"], 4, signed=True),
        "bias_thalf": interval(c["OPE-reference-T_half-DR"], 4, signed=True),
        "bias_t2": interval(c["OPE-reference-T2-DR"], 4, signed=True),
        "flip_thalf": interval(c["OPE-flip-T_half-DR"], 4),
        "flip_n": interval(c["OPE-flip-N-DR"], 4),
        "ope_daily_validated": same(c[f"OPE-reference-{t}-DR"]["sweep_validated"] for t in "TRF"),
        "ope_nt_reversed": c["OPE-order-N-T-DR"]["reversed_settings"],
        "ope_t2t_reversed": c["OPE-order-T2-T-DR"]["reversed_settings"],
        "ope_nt_dr_span": f"{num(min(dr_values), 4)} and {num(max(dr_values), 4)}",
        "ope_nt_truth_span": (
            f"{num(truth[0]['value'], 4, True)} to {num(truth[-1]['value'], 4, True)}"
        ),
        "main_settings": str(n_main),
        "ref_lift_fn": pct_interval(reference["daily_lift_F_over_N"], 1),
        "fn_flip": interval(c["F-N"], 3),
        "fn_flip_base": crossing(policy["flips"]["base_rate"]["F-N"], 2),
        "band_low": short(band["bounds"][0] * 100) + "%",
        "band_high": short(band["bounds"][1] * 100) + "%",
        "band_below": str(band["main_sweep_counts"]["below"]),
        "band_inside": str(band["main_sweep_counts"]["inside"]),
        "band_above": str(band["main_sweep_counts"]["above"]),
        "band_straddling": str(band["main_sweep_counts"]["straddling"]),
        "h3_wins": str(c["H3"]["wins"]),
        "h3_af_wins": "none" if c["H3-af"]["wins"] == 0 else str(c["H3-af"]["wins"]),
        "h3_flip": crossing(policy["flips"]["annoyance"]["H3"], 3),
        "obd_ips": interval(c["OBD-ratio-IPS"], 3),
        "obd_dr": interval(c["OBD-ratio-DR"], 3),
        "obd_ips_post_hoc": interval(c["OBD-ratio-IPS"]["post_hoc_empirical_target"], 3),
        "obd_ips_change": interval(c["OBD-ratio-IPS"]["post_hoc_difference"], 3, signed=True),
        # One figure for all four estimators, or the prose no longer holds
        "obd_shortfall": same(f"{round(1 - r['value'], 1):.0%}" for r in ratios.values()),
        "obd_ips_blocks": ends(ratios["IPS"], 3),
        "obd_dr_blocks": ends(ratios["DR"], 3),
        "obd_streams": str(streams["streams"]),
        "obd_ips_streams": str(streams["IPS"]["passes"]),
        "obd_dr_streams": str(streams["DR"]["passes"]),
        "b1_spread": num(c["B1"]["spread_over_bootstrap_se"], 1),
        "b2_spread": num(c["B2"]["spread_over_bootstrap_se"], 1),
        "b1_reference": interval(seeds["B1"]["mean_over_triples"], 4, signed=True),
        "b1_triples": str(seeds["B1"]["triples"]),
        "b1_negative": str(seeds["B1"]["negative_triples"]),
        "rater": rating["rater"],
        "kept_items": str(rating["kept"]["items"]),
        "kept_clear": str(rating["kept"]["clear_yes"]),
        "kept_fits": str(rating["kept"]["fits_yes"]),
        "kept_misleading": str(rating["kept"]["misleading_yes"]),
        "other_items": str(rating["passed_not_kept"]["items"]),
        "other_clear": str(rating["passed_not_kept"]["clear_yes"]),
        "other_fits": str(rating["passed_not_kept"]["fits_yes"]),
        "other_misleading": str(rating["passed_not_kept"]["misleading_yes"]),
    }


# Tables ---------------------------------------------------------------------------------

MAIN_SWEEP_ROWS = ("H1", "H2", "H3", "H4")
POLICY_ROWS = (*MAIN_SWEEP_ROWS, "H1-af", "H2-af", "H3-af", "H4-af")
TIER_A_ROWS = ("C1", "C2", "R1", "U1", "U2", "U3", "U4", "OBD-ratio-IPS", "OBD-ratio-DR")
TEMPLATE_ROWS = ("B1", "B2")


def sweep_average(claim: ClaimEntry) -> str:
    return interval(claim, 3, signed=True)


def verdict_counts(claim: ClaimEntry) -> str:
    if "wins" not in claim:
        return ""
    return f"{claim['wins']} / {claim['losses']} / {claim['neither']}"


def holm(claim: ClaimEntry) -> str:
    if "holm_rejected" not in claim:
        return ""
    return "rejected" if claim["holm_rejected"] else "not rejected"


def spread(claim: ClaimEntry) -> str:
    return num(claim["spread_over_bootstrap_se"], 1) if "spread_over_bootstrap_se" in claim else ""


def sweep_rows(results: dict[str, Any], ids: Sequence[str]) -> list[list[str]]:
    c = claims(results)
    return [
        [
            i,
            c[i]["statement"],
            shown_label(c[i]),
            sweep_average(c[i]),
            verdict_counts(c[i]),
            spread(c[i]),
            holm(c[i]),
        ]
        for i in ids
    ]


SWEEP_HEAD = (
    "Id",
    "Comparison",
    "Label",
    "Sweep average",
    "Wins / losses / neither",
    "Refit or seed SD over SE",
    "Holm",
)


def tier_a_rows(results: dict[str, Any]) -> list[list[str]]:
    c = claims(results)
    return [[i, c[i]["statement"], auto_interval(c[i]), holm(c[i])] for i in TIER_A_ROWS]


TIER_A_HEAD = ("Id", "Claim", "Value", "Holm")


def matrix_value(claim: ClaimEntry) -> str:
    if "value" in claim and "ci_low" in claim:
        return auto_interval(claim)
    if claim["id"] == "copy":
        return f"{claim['passed']} of {claim['returned']} passed; {claim['kept']} kept"
    return "verdict only"


def matrix_detail(claim: ClaimEntry) -> str:
    parts = []
    if "label" in claim:
        parts.append(shown_label(claim))
    if verdict_counts(claim):
        parts.append(f"wins / losses / neither {verdict_counts(claim)}")
    if holm(claim):
        parts.append(f"Holm {holm(claim)}")
    if "sweep_validated" in claim:
        parts.append(f"validated at {claim['sweep_validated']}")
    if "truth" in claim:
        parts.append(f"truth {auto_interval(claim['truth'])}")
    if "reversed_settings" in claim:
        parts.append(f"reversed at {claim['reversed_settings']} settings")
    if "post_hoc_empirical_target" in claim:
        parts.append(f"post hoc {auto_interval(claim['post_hoc_empirical_target'])}")
    if claim["id"] in ("H2", "H2-af"):
        parts.append(text.H2_NOTE)
    if claim["id"] in ("B1", "B2"):
        parts.append(text.TPL_LIMITS)
    return "; ".join(parts)


def matrix_rows(results: dict[str, Any]) -> list[list[str]]:
    """Id, tier, claim, source, value and detail of every claim."""
    return [
        [
            claim["id"],
            claim["tier"],
            claim["statement"],
            f"{claim['source']['file']}: {'.'.join(claim['source']['path'])}",
            matrix_value(claim),
            matrix_detail(claim),
        ]
        for claim in results["evidence"]["claims"]
    ]


def story_rows(values: dict[str, str]) -> list[list[str]]:
    return [[item, outcome.format(**values)] for item, outcome in text.STORY]


def tier_rows(results: dict[str, Any]) -> list[list[str]]:
    counts: dict[str, int] = {}
    for claim in results["evidence"]["claims"]:
        counts[claim["tier"]] = counts.get(claim["tier"], 0) + 1
    tiers = results["evidence"]["tiers"]
    return [
        [name, t["evidence"], t["supports"], t["cannot"], str(counts.get(name, 0))]
        for name, t in tiers.items()
    ]


TIER_HEAD = ("Tier", "Evidence", "Can support", "Cannot support", "Claims")


# Chart series -----------------------------------------------------------------------------

BIAS_CAPTION = "Carryover sweep (D048): DR bias against simulator truth, active-day share"
ORDERING_CAPTION = "No pushes minus one a day, active-day share"


def ope_bias(results: dict[str, Any]) -> tuple[list[float], list[svg.Series]]:
    """DR bias of the volume targets along the annoyance grid."""
    flips = results["ope_carryover"]["flips"]["annoyance"]
    names = (
        ("T½: half volume", "T_half.DR"),
        ("N: no pushes", "N.DR"),
        ("T2: two a day", "T2.DR"),
    )
    return flips["values"], [svg.Series(label, flips[path]["bias"]) for label, path in names]


def ope_ordering(results: dict[str, Any]) -> tuple[list[float], list[svg.Series]]:
    """Post hoc: no pushes minus one a day, truth against the per-day DR estimate."""
    nt = results["ope_carryover"]["post_hoc"]["orderings"]["annoyance"]
    pair = nt["N-T"]
    return nt["values"], [
        svg.Series("Truth", pair["truth"]),
        svg.Series("Per-day DR estimate", pair["DR"]),
    ]
