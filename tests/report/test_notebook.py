import sys
from pathlib import Path

import pytest

from push_lab.report import notebook

ROOT = Path(__file__).parents[2]


@pytest.mark.skipif(sys.platform == "win32", reason="line endings and paths differ (D051)")
def test_notebook_outputs_match_a_fresh_run() -> None:
    assert notebook.main(["--root", str(ROOT)]) == 0
