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
        assert c["tier"] in out["tiers"]
        if c["status"] == "reported" and c["tier"] in ("A", "B", "C"):
            assert c["ci_low"] <= c["value"] <= c["ci_high"], c["id"]


def test_numbers_are_copied_from_the_results() -> None:
    out = {c["id"]: c for c in evidence.build(ROOT)["claims"]}
    ranking = json.loads((ROOT / "results" / "ranking.json").read_text("utf-8"))
    assert out["C1"]["value"] == ranking["primary"]["C1_two_tower_vs_popularity_recall200"]["value"]
    templates = json.loads((ROOT / "results" / "templates.json").read_text("utf-8"))
    assert out["B1"]["value"] == templates["primary"]["B1"]["sweep_average"]["value"]
    assert out["B1"]["label"] == templates["primary"]["B1"]["label"]
    triples = templates["seed_variability"]["B1"]["triples"]
    assert out["B1"]["spread_source"].startswith(f"{triples} seed triples")
    policy = json.loads((ROOT / "results" / "policy.json").read_text("utf-8"))
    assert out["F-N"]["value"] == policy["flips"]["annoyance"]["F-N"]["crossings"][0]["at"]
    assert out["F-N-base"]["value"] == policy["flips"]["base_rate"]["F-N"]["crossings"][0]["at"]
    refits = policy["fit_variability"]["H1"]["refits"]
    assert out["H1-af"]["spread_source"].startswith(f"{refits} model refits")


def test_missing_results_are_pending(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    out = evidence.build(tmp_path)
    assert all(c["status"] == "pending" for c in out["claims"])


def test_simulator_ope_entries_carry_the_sweep() -> None:
    out = {c["id"]: c for c in evidence.build(ROOT)["claims"]}
    sweep = json.loads((ROOT / "results" / "ope_carryover.json").read_text("utf-8"))
    flip = sweep["flips"]["annoyance"]["N.DR"]["crossings"][0]
    assert out["OPE-flip-N-DR"]["value"] == flip["at"]
    assert out["OPE-reference-T-DR"]["sweep_validated"] == "24 of 24 settings"
    assert "sweep_validated" not in out["OPE-control-T-DR"]
    order = out["OPE-order-N-T-DR"]
    assert order["truth"]["value"] > 0 > order["ci_high"]
    assert order["reversed_settings"] == "23 of 24"
    design = json.loads((ROOT / "results" / "policy_design.json").read_text("utf-8"))
    grid = sweep["post_hoc"]["orderings"]["annoyance"]
    at = int(order["source"]["path"][-1])
    assert grid["values"][at] == design["reference"]["annoyance"]
    assert order["truth"] == grid["N-T"]["truth"][at]


def test_history_entries_carry_no_numbers() -> None:
    claims = evidence.build(ROOT)["claims"]
    history = [c for c in claims if c["tier"] == "history"]
    assert history and not [c for c in history if "value" in c]
    ips = next(c for c in claims if c["id"] == "OBD-ratio-IPS")
    target = ips["post_hoc_empirical_target"]
    assert target["ci_low"] <= target["value"] <= target["ci_high"]
    # A statement names only the fields its claim carries
    dr = next(c for c in claims if c["id"] == "OBD-ratio-DR")
    assert "post_hoc_difference" in ips and "post_hoc_difference" in ips["statement"]
    assert "post_hoc" not in dr["statement"] and not [k for k in dr if "post_hoc" in k]
