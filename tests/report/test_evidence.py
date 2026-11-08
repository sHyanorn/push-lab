import json
from pathlib import Path

from push_lab.report import evidence

ROOT = Path(__file__).parents[2]


def test_every_claim_has_a_tier_and_an_interval() -> None:
    out = evidence.build(ROOT)
    assert set(out["tiers"]) == {"A", "B", "C", "history", "descriptive"}
    assert not [c for c in out["claims"] if c["tier"] == "B"]
    ids = [c["id"] for c in out["claims"]]
    assert len(ids) == len(set(ids))
    for c in out["claims"]:
        assert c["tier"] in (*out["tiers"], "not validated")
        if c["status"] == "reported" and c["tier"] in ("A", "B", "C"):
            assert c["ci_low"] <= c["value"] <= c["ci_high"], c["id"]


def test_numbers_are_copied_from_the_results() -> None:
    out = {c["id"]: c for c in evidence.build(ROOT)["claims"]}
    ranking = json.loads((ROOT / "results" / "ranking.json").read_text("utf-8"))
    assert out["C1"]["value"] == ranking["primary"]["C1_two_tower_vs_popularity_recall200"]["value"]
    templates = json.loads((ROOT / "results" / "templates.json").read_text("utf-8"))
    assert out["B1"]["value"] == templates["primary"]["B1"]["sweep_average"]["value"]
    assert out["B1"]["label"] == templates["primary"]["B1"]["label"]
    policy = json.loads((ROOT / "results" / "policy.json").read_text("utf-8"))
    assert out["F-N"]["value"] == policy["flips"]["annoyance"]["F-N"]["crossings"][0]["at"]


def test_missing_results_are_pending(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    out = evidence.build(tmp_path)
    assert all(c["status"] == "pending" for c in out["claims"])
