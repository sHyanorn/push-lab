from pathlib import Path

import numpy as np
import pytest

from push_lab.data import fixtures, paths
from push_lab.uplift import curves, evaluate

pytestmark = pytest.mark.filterwarnings("ignore:Function stable_cumsum:FutureWarning")


SMALL = evaluate.Settings(bootstrap_samples=50, criteo_bins=20, forest_rows=300, batch_rows=700)


def test_pipeline_runs_on_synthetic_data(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=4_000)
    result = evaluate.run(tmp_path, SMALL)
    primary = result["primary"]
    assert set(primary) == {
        "U1_hillstrom_t_learner_qini",
        "U2_hillstrom_t_learner_vs_response_qini",
        "U3_criteo_t_learner_qini",
        "U4_criteo_t_learner_vs_response_qini",
    }
    for dataset in ("hillstrom", "criteo"):
        for outcome in evaluate.OUTCOMES:
            assert set(result[dataset][outcome]["methods"]) == set(evaluate.METHODS)
    assert result["hillstrom"]["setup"]["scikit_uplift_max_abs_difference"] < 1e-9
    assert result["criteo"]["setup"]["forest_training_rows"] == 300
    assert result["criteo"]["visit"]["control_rows_per_bin"]["t_learner"]["bins"] <= 20

    # The primary entries agree with the per-method table they come from
    for dataset, against_zero, paired in (
        ("hillstrom", "U1_hillstrom_t_learner_qini", "U2_hillstrom_t_learner_vs_response_qini"),
        ("criteo", "U3_criteo_t_learner_qini", "U4_criteo_t_learner_vs_response_qini"),
    ):
        methods = result[dataset]["visit"]["methods"]
        assert primary[against_zero]["value"] == methods["t_learner"]["qini"]["value"]
        difference = methods["t_learner"]["qini"]["value"] - methods["response"]["qini"]["value"]
        assert primary[paired]["value"] == pytest.approx(difference, abs=2e-6)

    # One treat-everyone interval per trial and outcome decides the budget shares of every method
    for dataset in ("hillstrom", "criteo"):
        for outcome in evaluate.OUTCOMES:
            entry = result[dataset][outcome]
            everyone = entry["treat_everyone_per_1000"]
            assert everyone["ci_low"] <= everyone["value"] <= everyone["ci_high"]
            defined = everyone["ci_low"] > 0 or everyone["ci_high"] < 0
            for name in evaluate.METHODS:
                assert "treat_everyone_per_1000" not in entry["methods"][name]
                for b in entry["budget"][name].values():
                    share = b["share_of_full_effect"]
                    assert (share is not None) == defined
                    if share is not None:
                        ratio = b["per_1000"]["value"] / everyone["value"]
                        assert share["value"] == pytest.approx(ratio, rel=1e-4, abs=1e-5)

    curve = result["criteo"]["visit"]["qini_curve"]
    assert curve["tier"] == "A"
    band = curve["methods"]["t_learner"]
    assert len(band["value"]) == len(band["ci_low"]) == len(curve["share"])
    assert band["value"][0] == 0.0
    deciles = result["criteo"]["visit"]["deciles_post_hoc"]["response"]
    assert len(deciles["uplift_per_1000"]["value"]) <= evaluate.DECILES


def test_criteo_scores_and_labels_come_from_the_same_rows(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=3_000)
    path = paths.processed_dir(tmp_path, "criteo") / "criteo.parquet"

    # Scoring functions that return a column of their input identify the row they scored
    def first_feature(x: evaluate.FloatArray) -> evaluate.FloatArray:
        return x[:, 0].copy()

    models: dict[str, dict[str, evaluate.Effect]] = {
        o: {"probe": first_feature} for o in evaluate.OUTCOMES
    }
    small = evaluate.score_criteo_test(path, models, batch_rows=128)
    large = evaluate.score_criteo_test(path, models, batch_rows=1 << 20)
    frame = fixtures.processed_tables(n=3_000)["criteo", "criteo"]
    test = frame[~frame["train"]]
    for t, outcomes, scores in (small, large):
        np.testing.assert_array_equal(t, test["treatment"].to_numpy())
        np.testing.assert_array_equal(scores["visit"]["probe"], test["f0"].to_numpy())
        np.testing.assert_array_equal(outcomes["visit"], test["visit"].to_numpy())


def test_hillstrom_only_matches_the_full_run(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=3_000)
    full = evaluate.run(tmp_path, SMALL)
    only = evaluate.run(tmp_path, SMALL, criteo=False)
    assert "criteo" not in only
    assert only["hillstrom"] == full["hillstrom"]
    for name, entry in only["primary"].items():
        assert "holm_rejected" not in entry
        expected = {
            k: v
            for k, v in full["primary"][name].items()
            if not k.startswith(("holm", "p_value_res"))
        }
        assert entry == expected


def test_treat_everyone_is_the_same_effect_every_method_sees() -> None:
    rng = np.random.default_rng(3)
    t = rng.integers(0, 2, 2_000)
    y = (rng.random(2_000) < 0.1 + 0.05 * t).astype(np.int64)
    shared = evaluate._treat_everyone(t, y, samples=200)
    full = 2 + 2 * len(evaluate.BUDGETS)
    for groups in (rng.integers(0, 7, 2_000), curves.exact_groups(rng.random(2_000))):
        point = evaluate._statistics(curves.counts(groups, t, y))[full]
        assert shared["value"] == pytest.approx(point, abs=1e-6)
    # U(n) / n is the difference in response rates between the arms
    direct = (y[t == 1].mean() - y[t == 0].mean()) * 1000
    assert shared["value"] == pytest.approx(direct, abs=1e-6)
    assert shared["ci_low"] < shared["value"] < shared["ci_high"]
