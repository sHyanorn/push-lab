"""The generated blocks of ``README.md``, each between a pair of ``results`` markers (D049)."""

from collections.abc import Sequence
from typing import Any

from push_lab.report import facts, text


def md_table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main_sweep_table(results: dict[str, Any]) -> str:
    """The delivery comparisons on the main sweep, without the columns the page explains in
    notes."""
    c = facts.claims(results)
    rows = [
        [i, facts.shown_label(c[i]), facts.sweep_average(c[i]), facts.verdict_counts(c[i])]
        for i in facts.MAIN_SWEEP_ROWS
    ]
    return md_table(("Id (main sweep)", "Label", "Sweep average", "Wins / losses / neither"), rows)


def finding(
    title: str,
    scope: str,
    body: str,
    shows: str,
    limits: str,
    values: dict[str, str],
    table: str = "",
) -> str:
    """A finding under its title; ``scope``, ``body``, ``shows`` and ``limits`` are templates."""
    scope, body, shows, limits = (t.format(**values) for t in (scope, body, shows, limits))
    reading = f"- **{text.SHOWS}:** {shows}\n- **{text.LIMITS}:** {limits}"
    return "\n\n".join(part for part in (title, f"*{scope}*", body, table, reading) if part)


def blocks(results: dict[str, Any], values: dict[str, str]) -> dict[str, str]:
    """The content of each generated block, by marker name."""
    tiers = md_table(facts.TIER_HEAD, facts.tier_rows(results)[:3])
    post_hoc = f"*{text.POST_HOC.capitalize()}:* {text.F1_POST_HOC}"
    findings = (
        finding(
            text.F1_TITLE,
            text.F1_SCOPE,
            f"{text.F1_BODY} {post_hoc}",
            text.F1_SHOWS,
            text.F1_LIMITS,
            values,
        ),
        finding(
            text.F2_TITLE,
            text.F2_SCOPE,
            f"{text.F2_BODY} {text.F2_GATES}",
            text.F2_SHOWS,
            text.F2_LIMITS,
            values,
            main_sweep_table(results),
        ),
        finding(
            text.F3_TITLE,
            text.F3_SCOPE,
            f"{text.F3_BODY} {text.F3_OBD_MISMATCH}",
            text.F3_SHOWS,
            text.F3_LIMITS,
            values,
        ),
    )
    numbered = (f"### {number}. {section}" for number, section in enumerate(findings, start=1))
    return {
        "evidence": f"{tiers}\n\n{text.NO_TIER_B.format(**values)}",
        "findings": "\n\n".join(numbered),
        "story": md_table(text.STORY_HEAD, facts.story_rows(values)),
    }


def render(readme: str, generated: dict[str, str]) -> str:
    """``readme`` with the content of every generated block replaced."""
    for name, content in generated.items():
        start, end = f"<!-- results:{name} -->\n", f"<!-- /results:{name} -->"
        head, found, rest = readme.partition(start)
        _, closed, tail = rest.partition(end)
        if not (found and closed):
            raise ValueError(f"README has no results:{name} block")
        readme = head + start + content + "\n" + end + tail
    return readme
