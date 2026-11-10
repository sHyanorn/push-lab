"""The report page: one static HTML file with inline CSS and SVG and no scripts (D049)."""

from collections.abc import Sequence
from html import escape
from typing import Any

from push_lab.report import facts, svg, text

REPO = "https://github.com/shyanorn/push-lab/blob/main"


def paragraph(template: str, values: dict[str, str], cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<p{attr}>{escape(template.format(**values))}</p>"


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


def legend(names: Sequence[str]) -> str:
    items = "".join(
        f'<span class="key s{i}"><i></i>{escape(n)}</span>' for i, n in enumerate(names, 1)
    )
    return f'<div class="legend">{items}</div>'


def shows_limits(shows: str, limits: str, values: dict[str, str]) -> str:
    return (
        '<dl class="reading">'
        f"<div><dt>{text.SHOWS}</dt><dd>{escape(shows.format(**values))}</dd></div>"
        f"<div><dt>{text.LIMITS}</dt><dd>{escape(limits.format(**values))}</dd></div>"
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
    def shown(point: dict[str, Any]) -> str:
        return facts.interval(point, 4, signed=True)

    chart = svg.line_chart(xs, series, x_title, y_title, caption, shown)
    rows = [[facts.short(x), *(shown(s.points[k]) for s in series)] for k, x in enumerate(xs)]
    data = table((x_title, *(s.name for s in series)), rows, "data", nowrap=(0,))
    badge = f' <span class="tag">{escape(tag)}</span>' if tag else ""
    return (
        f"<figure><figcaption>{escape(caption)}{badge}</figcaption>"
        f'{legend([s.name for s in series])}<div class="scroll">{chart}</div>'
        f"<details><summary>Data</summary>{data}</details></figure>"
    )


def finding_one(results: dict[str, Any], values: dict[str, str]) -> str:
    xs, bias = facts.ope_bias(results)
    _, ordering = facts.ope_ordering(results)
    post_hoc = escape(text.F1_POST_HOC.format(**values))
    return (
        f'<section id="ope"><h2>{escape(text.F1_TITLE)}</h2>'
        + paragraph(text.F1_SCOPE, values, "scope")
        + paragraph(text.F1_BODY, values)
        + chart_figure(facts.BIAS_CAPTION, xs, bias, "Annoyance", "Bias")
        + f'<p><span class="tag">{text.POST_HOC}</span> {post_hoc}</p>'
        + chart_figure(
            facts.ORDERING_CAPTION, xs, ordering, "Annoyance", "Difference", text.POST_HOC
        )
        + shows_limits(text.F1_SHOWS, text.F1_LIMITS, values)
        + "</section>"
    )


def finding_two(results: dict[str, Any], values: dict[str, str]) -> str:
    rows = facts.sweep_rows(results, facts.POLICY_ROWS)
    return (
        f'<section id="policies"><h2>{escape(text.F2_TITLE)}</h2>'
        + paragraph(text.F2_SCOPE, values, "scope")
        + paragraph(text.F2_BODY, values)
        + paragraph(text.F2_GATES, values)
        + table(facts.SWEEP_HEAD, rows, nowrap=(0, 4))
        + f'<p class="scope">{escape(text.SPREAD_NOTE)}</p>'
        + losing(results["policy"]["primary"]["main"], facts.MAIN_SWEEP_ROWS)
        + shows_limits(text.F2_SHOWS, text.F2_LIMITS, values)
        + "</section>"
    )


def finding_three(results: dict[str, Any], values: dict[str, str]) -> str:
    return (
        f'<section id="real-data"><h2>{escape(text.F3_TITLE)}</h2>'
        + paragraph(text.F3_SCOPE, values, "scope")
        + paragraph(text.F3_BODY, values)
        + paragraph(text.F3_OBD_MISMATCH, values)
        + table(facts.TIER_A_HEAD, facts.tier_a_rows(results), nowrap=(0, 2))
        + shows_limits(text.F3_SHOWS, text.F3_LIMITS, values)
        + "</section>"
    )


def losing(primary: dict[str, Any], ids: Sequence[str]) -> str:
    """Ids of the main-sweep settings where each comparison loses (METHODS 12.7)."""
    rows = [
        [i, ", ".join(str(s["id"]) for s in primary[i]["losing_settings"]) or "none"] for i in ids
    ]
    settings = table(("Id", "Settings"), rows, nowrap=(0,))
    return f"<details><summary>{text.LOSING}</summary>{settings}</details>"


def templates_section(results: dict[str, Any], values: dict[str, str]) -> str:
    rows = facts.sweep_rows(results, facts.TEMPLATE_ROWS)
    rating = escape(text.TPL_RATING.format(**values))
    return (
        f'<section id="templates"><h2>{escape(text.TPL_TITLE)}</h2>'
        + paragraph(text.TPL_SCOPE, values, "scope")
        + paragraph(text.TPL_BODY, values)
        + table(facts.SWEEP_HEAD, rows, nowrap=(0, 4))
        + losing(results["templates"]["primary"], facts.TEMPLATE_ROWS)
        + paragraph(text.TPL_LIMITS, values)
        + f'<p><span class="tag">{text.RATING_TAG}</span> {rating}</p>'
        + "</section>"
    )


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


def page(results: dict[str, Any], values: dict[str, str]) -> str:
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
        + paragraph(text.NO_TIER_B, values, "note")
        + f'<p class="reading-note">{escape(text.READING)}</p>'
        + f"<nav>{nav}</nav></header>"
        + finding_one(results, values)
        + finding_two(results, values)
        + finding_three(results, values)
        + templates_section(results, values)
        + f'<section id="story"><h2>{escape(text.STORY_TITLE)}</h2>'
        + table(text.STORY_HEAD, facts.story_rows(values))
        + "</section>"
        + f'<section id="matrix"><h2>{escape(text.MATRIX_TITLE)}</h2>'
        + f"<p>{escape(text.MATRIX_NOTE)}</p>"
        + matrix_table(facts.matrix_rows(results))
        + f"<h2>{escape(text.TIERS_HEAD)}</h2>"
        + table(facts.TIER_HEAD, facts.tier_rows(results), nowrap=(0,), stack=True)
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
