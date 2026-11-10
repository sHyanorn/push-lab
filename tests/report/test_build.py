import ast
import copy
import html
import itertools
import re
import shutil
from pathlib import Path

import pytest

from push_lab.report import build, svg, text

ROOT = Path(__file__).parents[2]
# Identifiers such as H1, T2, C2, U4 or D046, section references and the interval level
ALLOWED = re.compile(r"\b[A-Z]{1,3}\d{1,3}\b|95%|METHODS \d+(\.\d+)?")


def _strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text("utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_prose_holds_no_numbers() -> None:
    for value in _strings(Path(text.__file__)):
        stripped = ALLOWED.sub("", re.sub(r"\{\w+\}", "", value))
        assert not re.search(r"\d", stripped), value


def test_every_placeholder_is_filled_from_the_results() -> None:
    facts = build.facts(build.load(ROOT))
    names = {
        name for value in _strings(Path(text.__file__)) for name in re.findall(r"\{(\w+)\}", value)
    }
    assert names <= set(facts)
    assert all(facts.values())


def test_facts_quote_the_results() -> None:
    results = build.load(ROOT)
    facts = build.facts(results)
    claims = build.claims(results)
    assert facts["no_tier_b"] == results["evidence"]["note"]
    assert facts["obd_ips"] == build.interval(claims["OBD-ratio-IPS"], 3)
    assert facts["ope_nt_truth"].startswith("+0.0129 ")
    assert facts["ref_lift_fn"] == "-2.3% (-2.4% to -2.2%)".replace("-", svg.MINUS)
    assert facts["b1_spread"] == "17.7"
    assert facts["b2_spread"] == "19.2"


def test_reference_facts_come_from_the_registered_grid_point() -> None:
    results = build.load(ROOT)
    ref = build.reference_setting(results)
    assert ref["sweep"] == "grid"
    assert ref["values"] == results["policy_design"]["reference"]
    gate = ref["policies"]["G"]["opens_week1"]
    assert build.facts(results)["opens_w1_g"] == build.interval(gate, 3)


def test_committed_readme_matches_the_results() -> None:
    assert build.main(["--root", str(ROOT), "--check"]) == 0


def test_build_writes_the_report_and_refreshes_a_stale_readme(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "results", tmp_path / "results", ignore=shutil.ignore_patterns("runs"))
    readme = (ROOT / "README.md").read_text("utf-8")
    stale = readme.replace("<!-- /results:story -->", "stale\n<!-- /results:story -->")
    (tmp_path / "README.md").write_text(stale, encoding="utf-8")
    assert build.main(["--root", str(tmp_path), "--check"]) == 1
    assert build.main(["--root", str(tmp_path)]) == 0
    assert (tmp_path / "README.md").read_text("utf-8") == readme
    page = (tmp_path / "site" / "index.html").read_text("utf-8")
    facts = build.facts(build.load(ROOT))
    for name in ("ope_nt_truth", "flip_n", "ref_lift_fn", "obd_dr", "b2_spread"):
        assert facts[name] in page
    assert "<script" not in page


def test_render_readme_needs_every_block() -> None:
    with pytest.raises(ValueError, match="results:story"):
        build.render_readme(
            "<!-- results:evidence -->\n<!-- /results:evidence -->",
            {
                "evidence": "x",
                "story": "y",
            },
        )


def test_only_short_cells_are_kept_on_one_line() -> None:
    page = build.html(build.load(ROOT))
    cells = [html.unescape(c) for c in re.findall(r'<td class="nw">(.*?)</td>', page)]
    assert cells
    assert max(len(c) for c in cells) <= 32


def test_number_formatting() -> None:
    assert svg.num(-0.00004, 4) == "0.0000"
    assert svg.num(0.0129, 4, signed=True) == "+0.0129"
    assert svg.num(-0.0044, 4) == svg.MINUS + "0.0044"
    assert build.sig_decimals(0.004722) == 5
    assert build.sig_decimals(0.891) == 3


def test_nice_ticks_cover_the_range() -> None:
    ticks = svg.nice_ticks(-0.0361, 0.0354)
    assert ticks[0] <= -0.0361 and ticks[-1] >= 0.0354
    assert 0.0 in ticks
    steps = {round(b - a, 12) for a, b in itertools.pairwise(ticks)}
    assert len(steps) == 1


def test_tick_labels_keep_every_digit() -> None:
    chart = svg.line_chart(
        [0.0, 0.025, 0.05],
        [
            svg.Series(
                "a", [{"value": v, "ci_low": v - 0.01, "ci_high": v + 0.01} for v in (0, 1, 2)]
            )
        ],
        "Annoyance",
        "Bias",
        "test",
        str,
    )
    assert ">0.025<" in chart and ">0.000<" in chart


def test_stacked_tables_label_every_cell() -> None:
    plain = build.table(("A", "B"), [["x", "y"]], nowrap=(0,))
    assert "role=" not in plain and "data-label" not in plain
    stacked = build.table(("A", "B & C"), [["x", "y"]], nowrap=(0,), stack=True)
    assert '<table class="stack" role="table">' in stacked
    assert stacked.count('role="columnheader"') == 2
    assert '<td class="nw" role="cell" data-label="A">x</td>' in stacked
    assert 'data-label="B &amp; C">y</td>' in stacked


def test_post_hoc_readings_sit_beside_the_labels_they_were_written_for() -> None:
    results = build.load(ROOT)
    c = build.claims(results)
    for key, (label, reading) in build.SENSITIVITY.items():
        shown = build.shown_label(c[key])
        assert shown.startswith(label) and reading in shown and "D057" in shown
    assert build.shown_label(c["H2"]) == c["H2"]["label"]
    changed = copy.deepcopy(results)
    next(x for x in changed["evidence"]["claims"] if x["id"] == "H2-af")["label"] = "other"
    with pytest.raises(ValueError, match="H2-af"):
        build.check_readings(changed)


def test_prose_on_the_obd_post_hoc_change_fails_once_it_excludes_zero() -> None:
    results = build.load(ROOT)
    ips = next(x for x in results["evidence"]["claims"] if x["id"] == "OBD-ratio-IPS")
    assert build.facts(results)["obd_ips_change"] == build.interval(
        ips["post_hoc_difference"], 3, signed=True
    )
    changed = copy.deepcopy(results)
    claim = next(x for x in changed["evidence"]["claims"] if x["id"] == "OBD-ratio-IPS")
    claim["post_hoc_difference"]["ci_low"] = 0.001
    with pytest.raises(ValueError, match="post hoc"):
        build.check_readings(changed)
