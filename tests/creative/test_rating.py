import csv
import json
from pathlib import Path

import pytest

from push_lab.creative import generate, rating

from .test_generate import FakeApi


@pytest.fixture
def root(tmp_path: Path) -> Path:
    exchanges, records = generate.generate(FakeApi(), "some-model")
    generate.write(tmp_path, exchanges, records, "sdk-version")
    return tmp_path


def test_sheet_is_blind_and_complete(root: Path, tmp_path: Path) -> None:
    items = rating.items(root)
    assert len(items) == 60 + rating.EXTRA
    assert sum(it["kept"] for it in items) == 60
    assert [it["item"] for it in items] == list(range(1, 81))
    assert all("{" not in it["title"] + it["body"] for it in items)
    assert items == rating.items(root)  # deterministic
    sheet = tmp_path / "sheet.csv"
    rating.write_sheet(root, sheet)
    with sheet.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["item", "title", "body", *rating.QUESTIONS]


def test_tally_counts_and_survives_a_rebuild(root: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    rating.write_sheet(root, sheet)
    with sheet.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(clear="y", fits="n" if int(r["item"]) % 2 else "y", misleading="n")
    with sheet.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert rating.main(["--root", str(root), "--ratings", str(sheet)]) == 0
    assert generate.main(["--root", str(root), "--rebuild"]) == 0
    counts = json.loads((root / "results" / generate.RESULTS_FILE).read_text("utf-8"))["rating"]
    assert counts["rater"] == "one rater"
    assert counts["kept"]["items"] == 60 and counts["passed_not_kept"]["items"] == 20
    assert counts["kept"]["clear_yes"] == 60 and counts["kept"]["misleading_yes"] == 0
    assert counts["kept"]["fits_yes"] + counts["passed_not_kept"]["fits_yes"] == 40


def test_tally_rejects_bad_sheets(root: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    rating.write_sheet(root, sheet)
    with pytest.raises(ValueError):
        rating.tally(root, sheet)  # blank answers
