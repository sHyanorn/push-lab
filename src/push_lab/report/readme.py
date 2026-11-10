"""The generated blocks of ``README.md``, each between a pair of ``results`` markers (D049)."""

from collections.abc import Sequence
from typing import Any

from push_lab.report import facts, text


def md_table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def blocks(results: dict[str, Any], values: dict[str, str]) -> dict[str, str]:
    """The content of each generated block, by marker name."""
    tiers = md_table(facts.TIER_HEAD, facts.tier_rows(results)[:3])
    findings = []
    for k, (title, scope, body, shows, limits, extra) in enumerate(
        (
            (
                text.F1_TITLE,
                text.F1_SCOPE,
                text.F1_BODY,
                text.F1_SHOWS,
                text.F1_LIMITS,
                f"*{text.POST_HOC.capitalize()}:* " + text.F1_POST_HOC.format(**values),
            ),
            (
                text.F2_TITLE,
                text.F2_SCOPE,
                text.F2_BODY,
                text.F2_SHOWS,
                text.F2_LIMITS,
                text.F2_GATES.format(**values),
            ),
            (
                text.F3_TITLE,
                text.F3_SCOPE,
                text.F3_BODY,
                text.F3_SHOWS,
                text.F3_LIMITS,
                text.F3_OBD_MISMATCH.format(**values),
            ),
        ),
        start=1,
    ):
        table_md = ""
        if k == 2:
            rows = facts.sweep_rows(results, facts.POLICY_ROWS[:4])
            compact = [[r[0], r[2], r[3], r[4]] for r in rows]
            head = ("Id (main sweep)", "Label", "Sweep average", "Wins / losses / neither")
            table_md = md_table(head, compact) + "\n\n"
        findings.append(
            f"### {k}. {title}\n\n*{scope.format(**values)}*\n\n{body.format(**values)} {extra}"
            f"\n\n{table_md}- **{text.SHOWS}:** {shows.format(**values)}\n"
            f"- **{text.LIMITS}:** {limits.format(**values)}"
        )
    return {
        "evidence": f"{tiers}\n\n{text.NO_TIER_B.format(**values)}",
        "findings": "\n\n".join(findings),
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
