import importlib

import pytest

SUBPACKAGES = ["data", "recall", "rank", "uplift", "sim", "policy", "creative", "ope", "report"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"push_lab.{name}")
