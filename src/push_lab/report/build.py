"""Static HTML report and README numbers, built from the committed results (D049).

``python -m push_lab.report.build`` writes ``site/index.html`` and refreshes the generated
blocks of ``README.md``; ``--check`` only verifies that the README blocks are current.
Every number is read from ``results/*.json`` by ``facts.py``; the prose lives in ``text.py``,
the page layout in ``html.py`` and the README blocks in ``readme.py``.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from push_lab.report import facts, html, readme

SITE = Path("site")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the report and the README numbers.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--check", action="store_true", help="fail if the README is stale")
    args = parser.parse_args(argv)
    results = facts.load(args.root)
    readme_path = args.root / "README.md"
    current = readme_path.read_text("utf-8")
    updated = readme.render(current, readme.blocks(results))
    if args.check:
        if updated != current:
            print("README.md is out of date with results/; run python -m push_lab.report.build")
            return 1
        print("README.md matches results/")
        return 0
    readme_path.write_text(updated, encoding="utf-8", newline="\n")
    site = args.root / SITE
    site.mkdir(exist_ok=True)
    (site / "index.html").write_text(html.page(results), encoding="utf-8", newline="\n")
    print(f"wrote {site / 'index.html'} and README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
