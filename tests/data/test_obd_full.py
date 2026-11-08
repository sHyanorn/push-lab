import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from push_lab.data import fixtures, obd


def _csv(df: pd.DataFrame, names: dict[str, str]) -> bytes:
    """A frame written as the raw archive writes it: an unnamed index column first."""
    buf = io.StringIO()
    df.rename(columns=names).to_csv(buf)
    return buf.getvalue().encode("utf-8")


def test_convert_full_reads_the_mens_campaign_from_the_archive(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    affinity = {f"user_item_affinity_{k}": f"user-item_affinity_{k}" for k in range(34)}
    archive = tmp_path / "open_bandit_dataset.zip"
    logs = {}
    with zipfile.ZipFile(archive, "w") as z:
        for policy in obd.POLICIES:
            log = fixtures.obd_log(rng, 50, policy=policy)
            logs[policy] = log
            folder = f"open_bandit_dataset/{policy}/men"
            z.writestr(f"{folder}/men.csv", _csv(log, affinity))
            z.writestr(f"{folder}/item_context.csv", _csv(fixtures.obd_items(rng), {}))
            z.writestr(f"open_bandit_dataset/{policy}/women/women.csv", b"not read")
    written = obd.convert_full(archive, tmp_path / "out")
    assert len(written) == 4
    for policy, log in logs.items():
        back = obd.load(tmp_path / "out", policy)
        np.testing.assert_array_equal(back["click"], log["click"])
        np.testing.assert_array_equal(back["item_id"], log["item_id"])
