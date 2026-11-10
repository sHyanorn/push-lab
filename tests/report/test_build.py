import ast
import copy
import itertools
import re
import shutil
from html import unescape
from pathlib import Path

import pytest

from push_lab.report import build, facts, html, readme, svg, text

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
    values = facts.facts(facts.load(ROOT))
    names = {
        name for value in _strings(Path(text.__file__)) for name in re.findall(r"\{(\w+)\}", value)
    }
    assert names <= set(values)
    assert all(values.values())


def test_facts_quote_the_results() -> None:
    results = facts.load(ROOT)
    values = facts.facts(results)
    ips = facts.claims(results)["OBD-ratio-IPS"]
    assert values["no_tier_b"] == results["evidence"]["note"]
    assert values["obd_ips"] == facts.interval(ips, 3)
    assert values["ope_nt_truth"].startswith("+0.0129 ")
    assert values["ref_lift_fn"] == "-2.3% (-2.4% to -2.2%)".replace("-", svg.MINUS)
    assert values["b1_spread"] == "17.7"
    assert values["b2_spread"] == "19.2"


def test_reference_facts_come_from_the_registered_grid_point() -> None:
    results = facts.load(ROOT)
    ref = facts.reference_setting(results)
    assert ref["sweep"] == "grid"
    assert ref["values"] == results["policy_design"]["reference"]
    gate = ref["policies"]["G"]["opens_week1"]
    assert facts.facts(results)["opens_w1_g"] == facts.interval(gate, 3)


def test_committed_readme_matches_the_results() -> None:
    assert build.main(["--root", str(ROOT), "--check"]) == 0


def test_build_writes_the_report_and_refreshes_a_stale_readme(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "results", tmp_path / "results", ignore=shutil.ignore_patterns("runs"))
    committed = (ROOT / "README.md").read_text("utf-8")
    stale = committed.replace("<!-- /results:story -->", "stale\n<!-- /results:story -->")
    (tmp_path / "README.md").write_text(stale, encoding="utf-8")
    assert build.main(["--root", str(tmp_path), "--check"]) == 1
    assert build.main(["--root", str(tmp_path)]) == 0
    assert (tmp_path / "README.md").read_text("utf-8") == committed
    page = (tmp_path / "site" / "index.html").read_text("utf-8")
    values = facts.facts(facts.load(ROOT))
    for name in ("ope_nt_truth", "flip_n", "ref_lift_fn", "obd_dr", "b2_spread"):
        assert values[name] in page
    assert "<script" not in page


def test_render_readme_needs_every_block() -> None:
    with pytest.raises(ValueError, match="results:story"):
        readme.render(
            "<!-- results:evidence -->\n<!-- /results:evidence -->",
            {
                "evidence": "x",
                "story": "y",
            },
        )


def test_only_short_cells_are_kept_on_one_line() -> None:
    page = html.page(facts.load(ROOT))
    cells = [unescape(c) for c in re.findall(r'<td class="nw">(.*?)</td>', page)]
    assert cells
    assert max(len(c) for c in cells) <= 32


def test_number_formatting() -> None:
    assert svg.num(-0.00004, 4) == "0.0000"
    assert svg.num(0.0129, 4, signed=True) == "+0.0129"
    assert svg.num(-0.0044, 4) == svg.MINUS + "0.0044"
    assert facts.sig_decimals(0.004722) == 5
    assert facts.sig_decimals(0.891) == 3


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
    plain = html.table(("A", "B"), [["x", "y"]], nowrap=(0,))
    assert "role=" not in plain and "data-label" not in plain
    stacked = html.table(("A", "B & C"), [["x", "y"]], nowrap=(0,), stack=True)
    assert '<table class="stack" role="table">' in stacked
    assert stacked.count('role="columnheader"') == 2
    assert '<td class="nw" role="cell" data-label="A">x</td>' in stacked
    assert 'data-label="B &amp; C">y</td>' in stacked


def test_post_hoc_readings_sit_beside_the_labels_they_were_written_for() -> None:
    results = facts.load(ROOT)
    c = facts.claims(results)
    for key, (label, reading) in facts.SENSITIVITY.items():
        shown = facts.shown_label(c[key])
        assert shown.startswith(label) and reading in shown and "D057" in shown
    assert facts.shown_label(c["H2"]) == c["H2"]["label"]
    changed = copy.deepcopy(results)
    next(x for x in changed["evidence"]["claims"] if x["id"] == "H2-af")["label"] = "other"
    with pytest.raises(ValueError, match="H2-af"):
        facts.check_readings(changed)


def test_prose_on_the_obd_post_hoc_change_fails_once_it_excludes_zero() -> None:
    results = facts.load(ROOT)
    ips = next(x for x in results["evidence"]["claims"] if x["id"] == "OBD-ratio-IPS")
    assert facts.facts(results)["obd_ips_change"] == facts.interval(
        ips["post_hoc_difference"], 3, signed=True
    )
    changed = copy.deepcopy(results)
    claim = next(x for x in changed["evidence"]["claims"] if x["id"] == "OBD-ratio-IPS")
    claim["post_hoc_difference"]["ci_low"] = 0.001
    with pytest.raises(ValueError, match="post hoc"):
        facts.check_readings(changed)
