"""Static HTML report and README numbers, built from the committed results (D049).

``python -m push_lab.report.build`` writes ``site/index.html`` and refreshes the generated
blocks of ``README.md``; ``--check`` only verifies that the README blocks are current.
Every number is read from ``results/*.json``; the prose lives in ``text.py``.
"""

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any

from push_lab.data import paths
from push_lab.report import svg, text
from push_lab.report.svg import MINUS, num

REPO = "https://github.com/shyanorn/push-lab/blob/main"
SITE = Path("site")
FILES = (
    "evidence",
    "policy",
    "policy_design",
    "templates",
    "ope_carryover",
    "copy",
)


# Loading and formatting ---------------------------------------------------------------


def load(root: Path) -> dict[str, Any]:
    return {
        name: json.loads((root / paths.RESULTS / f"{name}.json").read_text("utf-8"))
        for name in FILES
    }


def claims(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in results["evidence"]["claims"]}


def interval(entry: dict[str, Any], decimals: int, signed: bool = False) -> str:
    v = num(entry["value"], decimals, signed)
    lo, hi = num(entry["ci_low"], decimals), num(entry["ci_high"], decimals)
    return f"{v} ({lo} to {hi})"


def pct_interval(entry: dict[str, Any], decimals: int) -> str:
    scaled = {k: entry[k] * 100 for k in ("value", "ci_low", "ci_high")}
    v, lo, hi = (num(scaled[k], decimals) + "%" for k in ("value", "ci_low", "ci_high"))
    return f"{v} ({lo} to {hi})"


def sig_decimals(x: float, digits: int = 3) -> int:
    if x == 0:
        return digits
    return max(0, digits - 1 - int(f"{abs(x):e}".split("e")[1]))


def auto_interval(entry: dict[str, Any]) -> str:
    return interval(entry, sig_decimals(entry["value"]))


def g(x: float) -> str:
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


def shown_label(claim: dict[str, Any]) -> str:
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
    c = claims(results)
    policy, carry, copy = results["policy"], results["ope_carryover"], results["copy"]
    grid = carry["flips"]["annoyance"]["values"]
    orderings = carry["post_hoc"]["orderings"]["annoyance"]["N-T"]
    dr_values = [p["value"] for p in orderings["DR"]]
    truth = orderings["truth"]
    band = policy["band_F_over_N"]
    rating = copy["rating"]
    seeds = results["templates"]["seed_variability"]
    n_main = sum(s["sweep"] == "main" for s in policy["settings"])
    check_readings(results)
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
        "ope_range": f"{g(grid[0])} to {g(grid[-1])}",
        "ope_settings": str(len(carry["settings"])),
        "ref_annoyance": g(results["policy_design"]["reference"]["annoyance"]),
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
        "band_low": g(band["bounds"][0] * 100) + "%",
        "band_high": g(band["bounds"][1] * 100) + "%",
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

POLICY_ROWS = ("H1", "H2", "H3", "H4", "H1-af", "H2-af", "H3-af", "H4-af")
TIER_A_ROWS = ("C1", "C2", "R1", "U1", "U2", "U3", "U4", "OBD-ratio-IPS", "OBD-ratio-DR")
TEMPLATE_ROWS = ("B1", "B2")


def wln(claim: dict[str, Any]) -> str:
    if "wins" not in claim:
        return ""
    return f"{claim['wins']} / {claim['losses']} / {claim['neither']}"


def holm(claim: dict[str, Any]) -> str:
    if "holm_rejected" not in claim:
        return ""
    return "rejected" if claim["holm_rejected"] else "not rejected"


def spread(claim: dict[str, Any]) -> str:
    return num(claim["spread_over_bootstrap_se"], 1) if "spread_over_bootstrap_se" in claim else ""


def sweep_rows(results: dict[str, Any], ids: Sequence[str]) -> list[list[str]]:
    c = claims(results)
    return [
        [
            i,
            c[i]["statement"],
            shown_label(c[i]),
            interval(c[i], 3, signed=True),
            wln(c[i]),
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


def matrix_value(claim: dict[str, Any]) -> str:
    if "value" in claim and "ci_low" in claim:
        return auto_interval(claim)
    if claim["id"] == "copy":
        return f"{claim['passed']} of {claim['returned']} passed; {claim['kept']} kept"
    return "verdict only"


def matrix_detail(claim: dict[str, Any]) -> str:
    parts = []
    if "label" in claim:
        parts.append(shown_label(claim))
    if wln(claim):
        parts.append(f"wins / losses / neither {wln(claim)}")
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
    rows = []
    for claim in results["evidence"]["claims"]:
        source = claim["source"]
        rows.append(
            [
                claim["id"],
                claim["tier"],
                claim["statement"],
                f"{source['file']}: {'.'.join(source['path'])}",
                matrix_value(claim),
                matrix_detail(claim),
            ]
        )
    return rows


def matrix_table(rows: Sequence[Sequence[str]]) -> str:
    """Id, tier, claim with its source beneath, value with its detail beneath."""
    th = "".join(f"<th>{h}</th>" for h in ("Id", "Tier", "Claim", "Value"))
    body = "".join(
        f"<tr><td>{escape(i)}</td><td>{escape(t)}</td>"
        f'<td>{escape(statement)}<span class="sub">{escape(source)}</span></td>'
        f'<td class="value">{escape(value)}<span class="sub">{escape(detail)}</span></td></tr>'
        for i, t, statement, source, value, detail in rows
    )
    return (
        '<div class="scroll"><table class="matrix">'
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
    )


# HTML -----------------------------------------------------------------------------------


def fill(template: str, f: dict[str, str]) -> str:
    return template.format(**f)


def p(template: str, f: dict[str, str], cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<p{attr}>{escape(fill(template, f))}</p>"


def table(
    head: Sequence[str],
    rows: Sequence[Sequence[str]],
    cls: str = "",
    nowrap: Sequence[int] = (),
    stack: bool = False,
) -> str:
    """A table in a horizontal scroller. With ``stack``, narrow screens show each row as a
    labeled block; the ARIA roles keep the header-cell pairing that block display can drop."""
    classes = " ".join(c for c in (cls, "stack" if stack else "") if c)
    attr = f' class="{classes}"' if classes else ""

    def role(name: str) -> str:
        return f' role="{name}"' if stack else ""

    th = "".join(f"<th{role('columnheader')}>{escape(h)}</th>" for h in head)

    def cell(k: int, value: str) -> str:
        nw = ' class="nw"' if k in nowrap else ""
        label = f' data-label="{escape(head[k])}"' if stack else ""
        return f"<td{nw}{role('cell')}{label}>{escape(value)}</td>"

    row = f"<tr{role('row')}>"
    body = "".join(row + "".join(cell(k, v) for k, v in enumerate(r)) + "</tr>" for r in rows)
    group = role("rowgroup")
    inner = f"<thead{group}>{row}{th}</tr></thead><tbody{group}>{body}</tbody>"
    return f'<div class="scroll"><table{attr}{role("table")}>{inner}</table></div>'


def legend(names: Sequence[str]) -> str:
    items = "".join(
        f'<span class="key s{i}"><i></i>{escape(n)}</span>' for i, n in enumerate(names, 1)
    )
    return f'<div class="legend">{items}</div>'


def shows_limits(shows: str, limits: str, f: dict[str, str]) -> str:
    return (
        '<dl class="reading">'
        f"<div><dt>{text.SHOWS}</dt><dd>{escape(fill(shows, f))}</dd></div>"
        f"<div><dt>{text.LIMITS}</dt><dd>{escape(fill(limits, f))}</dd></div>"
        "</dl>"
    )


def chart_figure(
    caption: str,
    xs: Sequence[float],
    series: Sequence[svg.Series],
    x_title: str,
    y_title: str,
    tag: str = "",
) -> str:
    fmt: Callable[[dict[str, Any]], str] = lambda e: interval(e, 4, signed=True)  # noqa: E731
    chart = svg.line_chart(xs, series, x_title, y_title, caption, fmt)
    rows = [[g(x), *(fmt(s.points[k]) for s in series)] for k, x in enumerate(xs)]
    data = table((x_title, *(s.name for s in series)), rows, "data", nowrap=(0,))
    badge = f' <span class="tag">{escape(tag)}</span>' if tag else ""
    return (
        f"<figure><figcaption>{escape(caption)}{badge}</figcaption>"
        f'{legend([s.name for s in series])}<div class="scroll">{chart}</div>'
        f"<details><summary>Data</summary>{data}</details></figure>"
    )


def ope_bias(results: dict[str, Any]) -> tuple[list[float], list[svg.Series]]:
    """DR bias of the volume targets along the annoyance grid."""
    flips = results["ope_carryover"]["flips"]["annoyance"]
    names = (
        ("T\u00bd: half volume", "T_half.DR"),
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


BIAS_CAPTION = "Carryover sweep (D048): DR bias against simulator truth, active-day share"
ORDERING_CAPTION = "No pushes minus one a day, active-day share"


def finding_one(results: dict[str, Any], f: dict[str, str]) -> str:
    xs, bias = ope_bias(results)
    _, ordering = ope_ordering(results)
    post_hoc = escape(fill(text.F1_POST_HOC, f))
    return (
        f'<section id="ope"><h2>{escape(text.F1_TITLE)}</h2>'
        + p(text.F1_SCOPE, f, "scope")
        + p(text.F1_BODY, f)
        + chart_figure(BIAS_CAPTION, xs, bias, "Annoyance", "Bias")
        + f'<p><span class="tag">{text.POST_HOC}</span> {post_hoc}</p>'
        + chart_figure(ORDERING_CAPTION, xs, ordering, "Annoyance", "Difference", text.POST_HOC)
        + shows_limits(text.F1_SHOWS, text.F1_LIMITS, f)
        + "</section>"
    )


def finding_two(results: dict[str, Any], f: dict[str, str]) -> str:
    return (
        f'<section id="policies"><h2>{escape(text.F2_TITLE)}</h2>'
        + p(text.F2_SCOPE, f, "scope")
        + p(text.F2_BODY, f)
        + p(text.F2_GATES, f)
        + table(SWEEP_HEAD, sweep_rows(results, POLICY_ROWS), nowrap=(0, 4))
        + f'<p class="scope">{escape(text.SPREAD_NOTE)}</p>'
        + losing(results["policy"]["primary"]["main"], ("H1", "H2", "H3", "H4"))
        + shows_limits(text.F2_SHOWS, text.F2_LIMITS, f)
        + "</section>"
    )


def finding_three(results: dict[str, Any], f: dict[str, str]) -> str:
    return (
        f'<section id="real-data"><h2>{escape(text.F3_TITLE)}</h2>'
        + p(text.F3_SCOPE, f, "scope")
        + p(text.F3_BODY, f)
        + p(text.F3_OBD_MISMATCH, f)
        + table(TIER_A_HEAD, tier_a_rows(results), nowrap=(0, 2))
        + shows_limits(text.F3_SHOWS, text.F3_LIMITS, f)
        + "</section>"
    )


def losing(primary: dict[str, Any], ids: Sequence[str]) -> str:
    """Ids of the main-sweep settings where each comparison loses (METHODS 12.7)."""
    rows = [
        [i, ", ".join(str(s["id"]) for s in primary[i]["losing_settings"]) or "none"] for i in ids
    ]
    settings = table(("Id", "Settings"), rows, nowrap=(0,))
    return f"<details><summary>{text.LOSING}</summary>{settings}</details>"


def templates_section(results: dict[str, Any], f: dict[str, str]) -> str:
    return (
        f'<section id="templates"><h2>{escape(text.TPL_TITLE)}</h2>'
        + p(text.TPL_SCOPE, f, "scope")
        + p(text.TPL_BODY, f)
        + table(SWEEP_HEAD, sweep_rows(results, TEMPLATE_ROWS), nowrap=(0, 4))
        + losing(results["templates"]["primary"], ("B1", "B2"))
        + p(text.TPL_LIMITS, f)
        + f'<p><span class="tag">{text.RATING_TAG}</span> {escape(fill(text.TPL_RATING, f))}</p>'
        + "</section>"
    )


def story_rows(f: dict[str, str]) -> list[list[str]]:
    return [[item, fill(outcome, f)] for item, outcome in text.STORY]


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

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;
--s2:#eb6834;--s3:#1baf7a;--accent:#256abf}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--accent:#86b6ef}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent:#86b6ef}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:28px;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:20px;margin:56px 0 8px;line-height:1.3}
p{margin:0 0 12px;max-width:72ch}
a{color:var(--accent)}
.lead{font-size:18px;color:var(--ink-2)}
.scope{color:var(--muted);font-size:14px}
.note{border-left:3px solid var(--accent);padding:8px 14px;background:var(--surface);
border-radius:0 6px 6px 0;max-width:72ch}
.reading-note{color:var(--muted);font-size:14px}
nav{display:flex;flex-wrap:wrap;gap:4px 16px;margin:20px 0 0;font-size:14px}
figure{margin:20px 0;padding:16px;background:var(--surface);border:1px solid var(--ring);
border-radius:8px}
figcaption{font-weight:600;font-size:15px}
.tag{font-weight:500;font-size:12px;color:var(--ink-2);border:1px solid var(--axis);
border-radius:999px;padding:1px 8px;margin:0 4px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink-2);margin:6px 0}
.key i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;
vertical-align:middle;background:currentColor}
.key.s1 i{background:var(--s1)}.key.s2 i{background:var(--s2)}.key.s3 i{background:var(--s3)}
details{margin-top:8px;font-size:14px}summary{cursor:pointer;color:var(--ink-2)}
.scroll{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;font-size:14px;width:100%}
th,td{text-align:left;vertical-align:top;padding:6px 10px;border-bottom:1px solid var(--grid)}
th{color:var(--ink-2);font-weight:600;white-space:nowrap}
td{font-variant-numeric:tabular-nums}
td.nw{white-space:nowrap}
.reading{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0 0}
.reading div{background:var(--surface);border:1px solid var(--ring);border-radius:8px;
padding:12px 14px}
.reading dt{font-weight:600;font-size:14px;margin-bottom:4px}.reading dd{margin:0}
.sub{display:block;color:var(--muted);font-size:12px;margin-top:2px;overflow-wrap:anywhere}
.matrix td.value{min-width:190px}
.matrix td:first-child{overflow-wrap:anywhere;min-width:7rem}
footer{margin-top:64px;color:var(--muted);font-size:14px}
@media (max-width:979px){th{white-space:normal}}
@media (max-width:640px){.reading{grid-template-columns:1fr}h1{font-size:24px}
.chart{min-width:600px}figure{padding:12px}
.stack thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
.stack th{white-space:nowrap}
.stack,.stack tbody,.stack tr{display:block}
.stack tr{padding:8px 0;border-bottom:1px solid var(--grid)}
.stack td:first-child{font-weight:600}
.stack td{display:grid;grid-template-columns:8rem 1fr;gap:8px;padding:2px 0;border:0}
.stack td::before{content:attr(data-label);content:attr(data-label) / "";color:var(--ink-2);
font-weight:600}}
"""


def html(results: dict[str, Any]) -> str:
    f = facts(results)
    nav = "".join(
        f'<a href="#{a}">{escape(t)}</a>'
        for a, t in (
            ("ope", "Off-policy estimates"),
            ("policies", "Delivery policies"),
            ("real-data", "Real-data checks"),
            ("templates", "Templates"),
            ("story", "Core story"),
            ("matrix", "Evidence matrix"),
        )
    )
    docs = " · ".join(
        f'<a href="{REPO}/{path}">{name}</a>'
        for name, path in (
            ("Specification", "docs/SPEC.md"),
            ("Methods", "docs/METHODS.md"),
            ("Decisions", "docs/DECISIONS.md"),
            ("Results", "results"),
        )
    )
    body = (
        "<main><header><h1>push-lab</h1>"
        f'<p class="lead">{escape(text.SUMMARY)}</p>'
        f'<p class="note">{escape(fill(text.NO_TIER_B, f))}</p>'
        f'<p class="reading-note">{escape(text.READING)}</p>'
        f"<nav>{nav}</nav></header>"
        + finding_one(results, f)
        + finding_two(results, f)
        + finding_three(results, f)
        + templates_section(results, f)
        + f'<section id="story"><h2>{escape(text.STORY_TITLE)}</h2>'
        + table(text.STORY_HEAD, story_rows(f))
        + "</section>"
        + f'<section id="matrix"><h2>{escape(text.MATRIX_TITLE)}</h2>'
        + f"<p>{escape(text.MATRIX_NOTE)}</p>"
        + matrix_table(matrix_rows(results))
        + f"<h2>{escape(text.TIERS_HEAD)}</h2>"
        + table(TIER_HEAD, tier_rows(results), nowrap=(0,), stack=True)
        + "</section>"
        + f"<footer>{docs}</footer></main>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>push-lab report</title>"
        f'<meta name="description" content="{escape(text.SUMMARY)}">'
        f"<style>{CSS}{svg.CHART_CSS}</style></head><body>{body}</body></html>\n"
    )


# README blocks ----------------------------------------------------------------------------


def md_escape(cell: str) -> str:
    return cell.replace("|", "\\|")


def md_table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(md_escape(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def readme_blocks(results: dict[str, Any]) -> dict[str, str]:
    f = facts(results)
    tiers = md_table(TIER_HEAD, tier_rows(results)[:3])
    evidence = f"{tiers}\n\n{fill(text.NO_TIER_B, f)}"
    findings = []
    for k, (title, scope, body, shows, limits, extra) in enumerate(
        (
            (
                text.F1_TITLE,
                text.F1_SCOPE,
                text.F1_BODY,
                text.F1_SHOWS,
                text.F1_LIMITS,
                f"*{text.POST_HOC.capitalize()}:* " + fill(text.F1_POST_HOC, f),
            ),
            (
                text.F2_TITLE,
                text.F2_SCOPE,
                text.F2_BODY,
                text.F2_SHOWS,
                text.F2_LIMITS,
                fill(text.F2_GATES, f),
            ),
            (
                text.F3_TITLE,
                text.F3_SCOPE,
                text.F3_BODY,
                text.F3_SHOWS,
                text.F3_LIMITS,
                fill(text.F3_OBD_MISMATCH, f),
            ),
        ),
        start=1,
    ):
        table_md = ""
        if k == 2:
            compact = [[r[0], r[2], r[3], r[4]] for r in sweep_rows(results, POLICY_ROWS[:4])]
            head = ("Id (main sweep)", "Label", "Sweep average", "Wins / losses / neither")
            table_md = md_table(head, compact) + "\n\n"
        findings.append(
            f"### {k}. {title}\n\n*{fill(scope, f)}*\n\n{fill(body, f)} {extra}\n\n{table_md}"
            f"- **{text.SHOWS}:** {fill(shows, f)}\n- **{text.LIMITS}:** {fill(limits, f)}"
        )
    return {
        "evidence": evidence,
        "findings": "\n\n".join(findings),
        "story": md_table(text.STORY_HEAD, story_rows(f)),
    }


def render_readme(readme: str, blocks: dict[str, str]) -> str:
    for name, content in blocks.items():
        start, end = f"<!-- results:{name} -->\n", f"<!-- /results:{name} -->"
        head, found, rest = readme.partition(start)
        _, closed, tail = rest.partition(end)
        if not (found and closed):
            raise ValueError(f"README has no results:{name} block")
        readme = head + start + content + "\n" + end + tail
    return readme


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the report and the README numbers.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--check", action="store_true", help="fail if the README is stale")
    args = parser.parse_args(argv)
    results = load(args.root)
    readme_path = args.root / "README.md"
    readme = readme_path.read_text("utf-8")
    updated = render_readme(readme, readme_blocks(results))
    if args.check:
        if updated != readme:
            print("README.md is out of date with results/; run python -m push_lab.report.build")
            return 1
        print("README.md matches results/")
        return 0
    readme_path.write_text(updated, encoding="utf-8", newline="\n")
    site = args.root / SITE
    site.mkdir(exist_ok=True)
    (site / "index.html").write_text(html(results), encoding="utf-8", newline="\n")
    print(f"wrote {site / 'index.html'} and README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
