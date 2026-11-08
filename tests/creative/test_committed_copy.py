"""Checks on the committed copy in data/creative (no API, no data)."""

import json
from pathlib import Path

import pytest

from push_lab.creative import filters, generate
from push_lab.creative.families import BODY_LENGTH, FAMILIES, TEMPLATES, TITLE_LENGTH, render

ROOT = Path(__file__).parents[2]
FOLDER = ROOT / generate.CREATIVE
pytestmark = pytest.mark.skipif(not (FOLDER / "catalog.json").exists(), reason="no copy yet")


def test_catalog_is_what_the_filters_select() -> None:
    exchanges = json.loads((FOLDER / "exchanges.json").read_text("utf-8"))
    selection = filters.select([generate.variants(e["text"]) for e in exchanges])
    catalog = generate.load_catalog(ROOT)
    assert [(t["family"], t["title"], t["body"]) for t in catalog["templates"]] == [
        (f, v.title, v.body) for _, (f, v) in sorted(selection.catalog.items())
    ]
    assert [t["id"] for t in catalog["templates"]] == list(range(TEMPLATES))
    assert all(t["id"] % len(FAMILIES) == t["family"] for t in catalog["templates"])


def test_every_fill_is_within_the_limits() -> None:
    for t in generate.load_catalog(ROOT)["templates"]:
        p = FAMILIES[t["family"]].placeholder
        for value in [""] if p is None else p.values:
            values = {} if p is None else {p.name: value}
            title, body = render(t["title"], values), render(t["body"], values)
            assert len(title) <= TITLE_LENGTH[1] and len(body) <= BODY_LENGTH[1]
            assert "{" not in title + body
