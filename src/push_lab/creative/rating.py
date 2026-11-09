"""Blind rating of the generated copy by one rater (METHODS 13.3).

``python -m push_lab.creative.rating --sheet <csv>`` writes the rating sheet: the 60 kept
templates and 20 other variants that passed the filters, shuffled, placeholders filled with
synthetic values, with no sign of which were kept. ``--ratings <csv>`` reads the filled sheet
and adds only counts to ``results/copy.json``; the sheet itself is not committed.
"""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from push_lab.creative import generate
from push_lab.creative.families import FAMILIES, render
from push_lab.data import paths
from push_lab.runtime import write_json

QUESTIONS = {
    "clear": "Easy to understand at a glance?",
    "fits": "Reads like a push from a short-video app?",
    "misleading": "Misleads or pressures the reader (false urgency, invented facts, rewards, "
    "guilt)?",
}
EXTRA = 20
SEED = 0
FILL_SEED = 1
ANSWERS = {"y": True, "n": False}


def items(root: Path) -> list[dict[str, Any]]:
    """The items in sheet order, each with its family, index, kept flag and rendered text."""
    folder = root / generate.CREATIVE
    exchanges = json.loads((folder / "exchanges.json").read_text("utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    raw = {e["family"]: generate.variants(e["text"]) for e in exchanges}
    passed = [v for v in manifest["variants"] if v["failed"] is None]
    kept = [v for v in passed if v["template"] is not None]
    others = [v for v in passed if v["template"] is None]
    rng = np.random.default_rng(SEED)
    extra = [others[i] for i in sorted(rng.choice(len(others), min(EXTRA, len(others)), False))]
    chosen = kept + extra
    order = rng.permutation(len(chosen))
    fill = np.random.default_rng(FILL_SEED)
    out = []
    for n, i in enumerate(order, 1):
        v = chosen[i]
        text = raw[v["family"]][v["index"]]
        p = FAMILIES[v["family"]].placeholder
        values = {} if p is None else {p.name: str(fill.choice(p.values))}
        out.append(
            {
                "item": n,
                "family": v["family"],
                "index": v["index"],
                "kept": v["template"] is not None,
                "title": render(text["title"], values),
                "body": render(text["body"], values),
            }
        )
    return out


def write_sheet(root: Path, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "title", "body", *QUESTIONS])
        for it in items(root):
            w.writerow([it["item"], it["title"], it["body"], "", "", ""])


def tally(root: Path, path: Path) -> dict[str, Any]:
    """Counts of yes per question among kept and other variants."""
    by_item = {it["item"]: it for it in items(root)}
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if sorted(int(r["item"]) for r in rows) != sorted(by_item):
        raise ValueError("the sheet must rate every item once")
    out: dict[str, Any] = {"rater": "one rater", "questions": QUESTIONS}
    for group, kept in (("kept", True), ("passed_not_kept", False)):
        mine = [r for r in rows if by_item[int(r["item"])]["kept"] is kept]
        counts: dict[str, int] = {"items": len(mine)}
        for q in QUESTIONS:
            answers = [r[q].strip().lower() for r in mine]
            if not set(answers) <= set(ANSWERS):
                raise ValueError(f"answers to {q} must be y or n")
            counts[f"{q}_yes"] = sum(ANSWERS[a] for a in answers)
        out[group] = counts
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the rating sheet or tally it.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sheet", type=Path, help="write a blank sheet here")
    group.add_argument("--ratings", type=Path, help="tally this filled sheet")
    args = parser.parse_args(argv)
    if args.sheet:
        write_sheet(args.root, args.sheet)
        print(f"wrote {args.sheet}")
        return 0
    results = args.root / paths.RESULTS / generate.RESULTS_FILE
    data = json.loads(results.read_text("utf-8"))
    data["rating"] = tally(args.root, args.ratings)
    write_json(results, data)
    print(f"added the rating counts to {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
