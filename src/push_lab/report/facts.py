"""What the report states, read from the committed results (D049).

The values the prose quotes, the readings it puts in words, and the rows and series of the
tables and charts. Every number on the page or in the README comes from here; the prose lives
in ``text.py``.
"""

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from push_lab.data import paths
from push_lab.report import svg, text
from push_lab.report.evidence import HORIZON_DESIGNS, ClaimEntry
from push_lab.report.svg import MINUS, num

FILES = (
    *("evidence", "policy", "policy_design", "templates", "ope_carryover", "copy"),
    "ope_obd_blocks",
    "cited",
    "duolingo",
    "ope_horizon",
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


def above(results: dict[str, Any], sweep: str, metric: str, high: str, low: str) -> int:
    """Settings of a sweep where policy ``high`` has the higher point estimate of ``metric``."""
    return sum(
        s["policies"][high][metric]["value"] > s["policies"][low][metric]["value"]
        for s in results["policy"]["settings"]
        if s["sweep"] == sweep
    )


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
    # The README states that the gate beats a daily push on average, yet not in most settings
    if c["H3"]["label"] != "depends on the setting" or c["H3"]["ci_low"] <= 0:
        raise ValueError("H3 is no longer positive on average and setting-dependent")
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
    # D065: the prose cites both campaigns as labeled "no shortfall shown"
    for name, entry in results["cited"]["ope_checks_obd"]["campaigns"].items():
        if name != "pooled" and (
            entry["label"] != "no shortfall shown" or not entry["ci_low"] <= 1 <= entry["ci_high"]
        ):
            raise ValueError(f"the cited {name} campaign no longer shows no shortfall")
    reference_b1 = results["templates"]["seed_variability"]["B1"]["mean_over_triples"]
    if not reference_b1["ci_low"] < 0 < reference_b1["ci_high"]:
        raise ValueError("B1 at the reference is no longer undetectable")
    # Every grid along which F-N flips has an evidence row that names the side of the crossing
    # where a daily push gains, and no other grid has one
    rows = {x["source"]["path"][1]: x["statement"] for k, x in c.items() if k.startswith("F-N")}
    fn_flips = {n: f["F-N"] for n, f in policy["flips"].items() if f["F-N"]["result"] == "flip"}
    if set(rows) != set(fn_flips):
        raise ValueError(f"F-N flips along {sorted(fn_flips)}; its rows name {sorted(rows)}")
    for name, flip in fn_flips.items():
        side = "below" if flip["estimates"][0]["value"] > 0 else "above"
        if f"only {side} this" not in rows[name]:
            raise ValueError(f"the F-N row of {name} does not say it gains {side} the crossing")
    resolved = {
        (n, h)
        for n, f in policy["flips"].items()
        for h in ("H1", "H2", "H3", "H4")
        if any(x["resolved"] for x in f[h]["crossings"])
    }
    if resolved != {("annoyance", "H3")}:
        raise ValueError(f"resolved crossings of H1 to H4 are now {sorted(resolved)}")
    main = sum(s["sweep"] == "main" for s in policy["settings"])
    if above(results, "main", "opens_week1", "T", "G") != main:
        raise ValueError("a daily push no longer has more week-one opens in every main setting")
    band = policy["band_F_over_N"]["main_sweep_counts"]
    if band["below"] * 2 <= sum(band.values()):
        raise ValueError("most settings are no longer below the band")
    carry = results["ope_carryover"]["flips"]
    for name in ("fatigue_elasticity", "opt_out", "return_boost", "novelty"):
        if any(carry[name][k]["result"] == "flip" for k in carry[name] if k != "values"):
            raise ValueError(f"the OPE bias now flips along {name}")
    check_duolingo(results["duolingo"])
    check_horizon(results["ope_horizon"])


# The readings of the Duolingo study that the prose states in words (D080)
DUOLINGO_READINGS = {
    ("logging_check", "uniform"): True,
    ("interpreted",): True,
    ("DL1", "answer"): "replicated",
    ("DL2", "paper_form", "converged"): True,
    ("DL2", "gamma_reading"): "consistent with these data",
    ("DL2", "half_life", "reading"): "h not determined",
    ("DL3", "DL3a", "IPS", "label"): "RD better",
    ("DL3", "DL3b", "IPS", "label"): "no detectable difference",
}


def check_duolingo(data: dict[str, Any]) -> None:
    for path, expected in DUOLINGO_READINGS.items():
        found: Any = data
        for key in path:
            found = found[key]
        if found != expected:
            raise ValueError(f"Duolingo {'.'.join(path)} is now {found!r}")
    if recent_change(data)["conservative"]["ci_high"] >= 0:
        raise ValueError("Duolingo: a recent showing no longer lowers completion")


# The primary labels of the long-horizon study as the prose describes them (D079)
HORIZON_LABELS = {
    ("E1", "SMIS", "N"): "reverses",
    ("E1", "FQE", "T2"): "reverses",
    **{
        ("E2", name, target): "validated, does not separate"
        for name in ("PDIS_week", "SMIS_week")
        for target in ("N", "T2")
    },
    ("E2", "FQE_week", "N"): "not validated",
    ("E2", "FQE_week", "T2"): "not validated",
    ("E3", "arm_means", "N"): "validated, does not separate",
    ("E3", "arm_means", "T2"): "validated, does not separate",
}


def check_horizon(data: dict[str, Any]) -> None:
    if not data["implementation_check"]["passed"]:
        raise ValueError("the long-horizon implementation check no longer passes")
    for (design, name, target), label in HORIZON_LABELS.items():
        if data["primary"][design]["labels"][name][target] != label:
            raise ValueError(f"long-horizon {design} {name} {target} is no longer {label!r}")
    reference = data["settings"]["reference"]["designs"]
    for target in ("N", "T2"):
        if "bias" not in reference["E2"]["FQE_week|S0"][target]["reasons"]:
            raise ValueError(f"FQE-week is no longer biased for {target}")
    if not arms_recover(data):
        raise ValueError("arm means no longer recover both orderings at any setting")
    if smis_reversals(data) != len(annoyed(data)):
        raise ValueError("SMIS no longer reverses at every setting with annoyance")
    methods = {name: s["truth"]["method"] for name, s in data["settings"].items()}
    runs = f"{data['truth_runs']} paired online runs"
    if methods.pop("control") != "exact" or set(methods.values()) != {runs}:
        raise ValueError(f"the truth is now computed otherwise: {methods}")
    if any(data["settings"][s]["truth"]["D"]["N"]["value"] <= 0 for s in annoyed(data)):
        raise ValueError("no pushes no longer beat one a day wherever pushes annoy users")


def annoyed(data: dict[str, Any]) -> list[str]:
    """Settings of the long-horizon study where pushes annoy users."""
    return [name for name, s in data["settings"].items() if s["params"]["annoyance"] > 0]


def smis_reversals(data: dict[str, Any]) -> int:
    """Settings where pushes annoy users at which SMIS on the current logs reverses no pushes
    minus one a day."""
    settings = data["settings"]
    return sum(
        settings[s]["designs"]["E1"]["SMIS|S0"]["N"]["label"] == "reverses" for s in annoyed(data)
    )


def arms_recover(data: dict[str, Any]) -> list[str]:
    """Settings where arm means recover the ordering of both primary differences."""
    return [
        name
        for name, s in data["settings"].items()
        if all(
            s["designs"]["E3"]["arm_means"][x]["label"] == "recovers the ordering"
            for x in ("N", "T2")
        )
    ]


def setting_name(data: dict[str, Any], name: str, article: bool = True) -> str:
    """A setting of the long-horizon study by the parameter it changes from the reference."""
    if name in ("reference", "control"):
        return f"the {name}" if article else name
    reference = data["settings"]["reference"]["params"]
    (key,) = (k for k, v in data["settings"][name]["params"].items() if v != reference[k])
    value = short(data["settings"][name]["params"][key])
    if key == "annoyance":
        return f"an annoyance of {value}" if article else f"annoyance {value}"
    words = key.replace("_", " ").replace("half life", "half-life")
    return f"a {words} of {value} days" if article else f"{words} {value}"


def recent_change(data: dict[str, Any]) -> dict[str, Any]:
    """The relative change in completion of the most recent recency bin of the DL2 curve."""
    bins = [k for k in data["DL2"]["curve"]["parameters"] if k.startswith("beta")]
    return dict(data["DL2"]["curve"]["parameters"][bins[0]]["relative change"])


def conservative(found: Mapping[str, Any]) -> dict[str, float]:
    """An estimate with the conservative interval of METHODS 15."""
    return {"value": found["value"], **found["conservative"]}


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
    cited = results["cited"]["ope_checks_obd"]["campaigns"]
    seeds = results["templates"]["seed_variability"]
    n_main = sum(s["sweep"] == "main" for s in policy["settings"])
    n_above_af = above(results, "annoyance_free", "active_week4", "N", "F")
    reference = reference_setting(results)
    ref = reference["policies"]
    return study_facts(results) | {
        "no_tier_b": results["evidence"]["note"],
        "crossing_note": text.CROSSING_NOTE,
        "h2_note": text.H2_NOTE,
        "ope_grids": str(len(carry["post_hoc"]["orderings"])),
        "opens_w1_t": interval(ref["T"]["opens_week1"], 3),
        "opens_w1_g": interval(ref["G"]["opens_week1"], 3),
        "active_w4_f": interval(ref["F"]["active_week4"], 2),
        "active_w4_n": interval(ref["N"]["active_week4"], 2),
        "w1_t_above_g": str(above(results, "main", "opens_week1", "T", "G")),
        "w4_n_above_f": str(above(results, "main", "active_week4", "N", "F")),
        "w4_n_above_f_af": "none" if n_above_af == 0 else str(n_above_af),
        "af_settings": str(sum(s["sweep"] == "annoyance_free" for s in policy["settings"])),
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
        "h3_average": sweep_average(c["H3"]),
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
        "obd_cited_all": interval(cited["all"], 3),
        "obd_cited_women": interval(cited["women"], 3),
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


def study_facts(results: dict[str, Any]) -> dict[str, str]:
    """The values of the Duolingo and long-horizon prose."""
    duo, hz = results["duolingo"], results["ope_horizon"]
    rows = duo["DL1"]["rows"]
    replicated = sum(r["label"] == "replicated" for r in rows.values())
    recent = recent_change(duo)
    lowered = {
        "value": -recent["value"],
        "ci_low": -recent["conservative"]["ci_high"],
        "ci_high": -recent["conservative"]["ci_low"],
    }
    bins = [k for k in duo["DL2"]["curve"]["parameters"] if k.startswith("beta")]
    truth = hz["settings"]["reference"]["truth"]["D"]
    logs = {s: len(v["E1"]) for s, v in hz["log_seeds"].items()}
    projection = hz["arm_mean_projection"]
    return {
        # The gain follows "by", which gives its sign
        "dl3a": interval(conservative(duo["DL3"]["DL3a"]["IPS"]), 5),
        "dl3b": interval(conservative(duo["DL3"]["DL3b"]["IPS"]), 5, signed=True),
        "dl1_replicated": f"{replicated} of {len(rows)}",
        "dl_rows": f"{sum(duo['rows'].values()):,}",
        "dl_days": str(len(duo["j_mean_by_day"])),
        "dl_test_rows": f"{duo['rows']['test']:,}",
        "dl_test_days": str(duo["DL3"]["DL3a"]["IPS"]["days"]),
        "dl_mean_reward": num(duo["DL1"]["mean_test_reward"], 5),
        "dl_gamma": interval(conservative(duo["DL2"]["paper_form"]["parameters"]["gamma"]), 4),
        "dl_h": interval(conservative(duo["DL2"]["half_life"]), 1),
        "dl_recent_days": bins[0].split(", ")[1].rstrip("])"),
        "dl_recent": pct_interval(lowered, 1),
        "hz_settings": str(len(hz["settings"])),
        "hz_logs_ref": str(logs.pop("reference")),
        "hz_logs": same(str(n) for n in logs.values()),
        "hz_users": f"{hz['users']:,}",
        "hz_all_users": f"{projection['all_users']:,}",
        "hz_truth_runs": str(hz["truth_runs"]),
        "hz_truth_n": interval(truth["N"], 4, signed=True),
        "hz_truth_t2": interval(truth["T2"], 4, signed=True),
        "hz_e1_reversed": f"{smis_reversals(hz)} of the {len(hz['settings'])}",
        "hz_arm_recovers": " and ".join(setting_name(hz, s) for s in arms_recover(hz)),
        "hz_projected": same(num(projection[x]["projected"], 3) for x in ("N", "T_half", "T2")),
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
    if "setting_labels" in claim:
        counts = Counter(claim["setting_labels"].values())
        shown = ", ".join(f"{label} {n}" for label, n in counts.most_common())
        parts.append(f"labels at the {len(claim['setting_labels'])} settings: {shown}")
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


DL1_HEAD = ("Policy", "IPS lift", "Paper's lift", "Label")
DL3_HEAD = ("Comparison", "IPS", "SNIPS", "DR", "Label")
DL_CURVE_HEAD = ("Last sent, days before", "Change in completion")


def dl1_rows(results: dict[str, Any]) -> list[list[str]]:
    """Each DL1 policy's IPS lift over the mean test reward against the paper's."""
    return [
        [
            name,
            pct_interval(conservative(row["lift"]), 2),
            num(row["paper_lift"] * 100, 2) + "%",
            row["label"],
        ]
        for name, row in results["duolingo"]["DL1"]["rows"].items()
    ]


def dl3_rows(results: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key in ("DL3a", "DL3b"):
        found = results["duolingo"]["DL3"][key]
        name = f"{key}: {found['comparison'].replace(' - ', ' minus ')}"
        cells = [interval(conservative(found[e]), 5, signed=True) for e in ("IPS", "SNIPS", "DR")]
        rows.append([name, *cells, found["IPS"]["label"]])
    return rows


def dl_curve_rows(results: dict[str, Any]) -> list[list[str]]:
    """Relative change in completion per recency bin, against a template not in the history."""
    parameters = results["duolingo"]["DL2"]["curve"]["parameters"]
    return [
        [key.removeprefix("beta "), pct_interval(conservative(entry["relative change"]), 1)]
        for key, entry in parameters.items()
        if key.startswith("beta")
    ]


HZ_HEAD = ("Design", "Estimator", "No pushes minus one a day", "Two a day minus one")
HZ_CODE = {
    "reverses": "R",
    "recovers the ordering": "O",
    "validated, does not separate": "V",
    "not validated": "X",
}


def horizon_claims(results: dict[str, Any]) -> list[ClaimEntry]:
    return [c for c in results["evidence"]["claims"] if c["id"].startswith("HZ-")]


def horizon_rows(results: dict[str, Any]) -> list[list[str]]:
    """The primary labels at the reference, each with its mean estimate over logs."""
    rows: dict[tuple[str, str], list[str]] = {}
    for claim in horizon_claims(results):
        _, design, name, _ = claim["id"].split("-")
        cell = f"{claim['label']}; {interval(claim, 4, signed=True)}"
        rows.setdefault((design, name), []).append(cell)
    return [
        [f"{d}, {HORIZON_DESIGNS[d][0]}", n.replace("_week", "-week").replace("_", " "), *v]
        for (d, n), v in rows.items()
    ]


def horizon_setting_head(results: dict[str, Any]) -> tuple[str, ...]:
    hz = results["ope_horizon"]
    return ("Id", *(setting_name(hz, s, article=False) for s in hz["settings"]))


def horizon_setting_rows(results: dict[str, Any]) -> list[list[str]]:
    """The label code of each primary estimator and difference at every setting."""
    return [
        [claim["id"], *(HZ_CODE[label] for label in claim["setting_labels"].values())]
        for claim in horizon_claims(results)
    ]


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
