from pathlib import Path

from push_lab.data import fixtures
from push_lab.rank import evaluate
from push_lab.recall import two_tower


def test_pipeline_runs_on_synthetic_data(tmp_path: Path) -> None:
    fixtures.write_processed(tmp_path, n=4_000)
    settings = evaluate.Settings(
        tower=two_tower.Config(epochs=1, batch=256, threads=1),
        fine_params={"min_data_in_leaf": 5, "num_threads": 1},
        sample_users=6,
        latency_users=3,
        bootstrap_samples=50,
    )
    result = evaluate.run(tmp_path, settings)
    assert set(result["primary"]) == {
        "C1_two_tower_vs_popularity_recall200",
        "C2_fused_vs_two_tower_gauc_click",
    }
    survival = [t["survival"]["value"] for t in result["truncation"]]
    assert survival == sorted(survival)
    assert 0.0 <= result["truncation"][0]["top1_agreement"]["value"] <= 1.0
    assert result["setup"]["evaluation_users"] == fixtures.N_USERS // 2
