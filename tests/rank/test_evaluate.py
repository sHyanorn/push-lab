from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from push_lab.data import fixtures, kuairand, paths, splits
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


def _set_clicks(root: Path, table: str, clicks: Callable[[pd.DataFrame], pd.Series]) -> None:
    path = paths.processed_dir(root, "kuairand") / f"{table}.parquet"
    log = pd.read_parquet(path)
    log["is_click"] = clicks(log).astype(np.int64)
    pq.write_table(pa.Table.from_pandas(log, schema=pq.read_schema(path)), path)


def _even(log: pd.DataFrame) -> pd.Series:
    return log["video_id"] % 2 == 0


def test_ranking_reads_aligned_labels_of_the_evaluation_half(tmp_path: Path) -> None:
    # Even items are clicked in the standard logs, so popularity ranks them first. In the random
    # log, evaluation-half users click even items and simulator-half users odd ones, so
    # popularity's GAUC is 1 on the evaluation half, 0 on the other and 0.5 with shifted labels.
    fixtures.write_processed(tmp_path, n=4_000)
    processed = paths.processed_dir(tmp_path, "kuairand")
    _set_clicks(tmp_path, "log_standard_early", _even)
    evaluation, _ = splits.user_halves(kuairand.load(processed, "log_random"))
    _set_clicks(tmp_path, "log_random", lambda log: _even(log) == log["user_id"].isin(evaluation))
    settings = evaluate.Settings(
        tower=two_tower.Config(epochs=1, batch=256, threads=1),
        fine_params={"min_data_in_leaf": 5, "num_threads": 1},
        sample_users=6,
        latency_users=3,
        bootstrap_samples=50,
    )
    result = evaluate.run(tmp_path, settings)
    assert result["ranking"]["popularity"]["gauc_click"]["value"] > 0.95
