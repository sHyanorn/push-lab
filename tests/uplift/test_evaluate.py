from pathlib import Path

import pytest

from push_lab.data import fixtures
from push_lab.uplift import evaluate

pytestmark = pytest.mark.filterwarnings("ignore:Function stable_cumsum:FutureWarning")


def test_pipeline_runs_on_synthetic_data(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=4_000)
    settings = evaluate.Settings(bootstrap_samples=50, criteo_bins=20, forest_rows=300)
    result = evaluate.run(tmp_path, settings)
    assert set(result["primary"]) == {
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
    bins = result["criteo"]["visit"]["control_rows_per_bin"]["t_learner"]
    assert bins["bins"] <= 20
    assert result["hillstrom"]["visit"]["methods"]["t_learner"]["qini"]["tier"] == "A"
