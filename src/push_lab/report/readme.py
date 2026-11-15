"""The generated blocks of ``README.md``, each between a pair of ``results`` markers (D049).

The README gives the summary and one line per finding; the report page keeps the full text
and the outcome of the registered story.
"""

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
    findings = md_table(
        text.README_FINDINGS_HEAD,
        [
            (str(number), finding.format(**values), evidence.format(**values))
            for number, (finding, evidence) in enumerate(text.README_FINDINGS, start=1)
        ],
    )
    return {
        "summary": text.SUMMARY.format(**values),
        "evidence": f"{tiers}\n\n{text.NO_TIER_B.format(**values)}",
        "findings": f"{findings}\n\n{text.README_FINDINGS_NOTE.format(**values)}",
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
