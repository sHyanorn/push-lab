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


def test_h2_rows_carry_the_note_of_their_results() -> None:
    out = {c["id"]: c for c in evidence.build(ROOT)["claims"]}
    policy = json.loads((ROOT / "results" / "policy.json").read_text("utf-8"))
    assert out["H2"]["note"] == policy["primary"]["main"]["H2"]["note"]
    assert out["H2-af"]["note"] == policy["primary"]["annoyance_free"]["H2"]["note"]
    assert "note" not in out["H1"]


def test_long_horizon_rows_carry_the_truth_and_every_setting() -> None:
    out = {c["id"]: c for c in evidence.build(ROOT)["claims"]}
    data = json.loads((ROOT / "results" / "ope_horizon.json").read_text("utf-8"))
    rows = [c for c in out.values() if c["id"].startswith("HZ-")]
    assert len(rows) == 12 and all(c["tier"] == "C" for c in rows)
    arms = out["HZ-E3-arm_means-N"]
    found = data["settings"]["reference"]["designs"]["E3"]["arm_means"]["N"]
    assert (arms["value"], arms["ci_low"]) == (found["mean"]["value"], found["mean"]["ci_low"])
    assert arms["label"] == data["primary"]["E3"]["labels"]["arm_means"]["N"]
    assert arms["truth"]["value"] == data["settings"]["reference"]["truth"]["D"]["N"]["value"]
    smis = out["HZ-E1-SMIS-N"]["setting_labels"]
    assert set(smis) == set(data["settings"])
    assert smis["reference"] == data["primary"]["E1"]["labels"]["SMIS"]["N"]


def test_duolingo_rows_use_the_conservative_intervals() -> None:
    out = {c["id"]: c for c in evidence.build(ROOT)["claims"]}
    data = json.loads((ROOT / "results" / "duolingo.json").read_text("utf-8"))
    ips = data["DL3"]["DL3a"]["IPS"]
    assert out["DL3a"]["value"] == ips["value"]
    assert out["DL3a"]["ci_low"] == ips["conservative"]["ci_low"]
    assert out["DL3a"]["label"] == ips["label"]
    assert out["DL2-gamma"]["label"] == data["DL2"]["gamma_reading"]
    assert out["DL1"]["status"] == "verdict" and "value" not in out["DL1"]
    assert {c["tier"] for k, c in out.items() if k.startswith("DL")} == {"A"}


def test_unread_studies_add_no_rows(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    for name, gate in (("ope_horizon.json", "implementation_check"), ("duolingo.json", "")):
        data = json.loads((ROOT / "results" / name).read_text("utf-8"))
        if gate:
            data[gate]["passed"] = False
        else:
            data["interpreted"] = False
        (tmp_path / "results" / name).write_text(json.dumps(data), encoding="utf-8")
    assert evidence.horizon_entries(tmp_path) == []
    assert evidence.duolingo_entries(tmp_path) == []
