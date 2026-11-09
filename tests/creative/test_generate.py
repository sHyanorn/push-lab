import copy
import json
from pathlib import Path
from typing import Any

import pytest

from push_lab.creative import filters, generate
from push_lab.creative.families import FAMILIES, REQUESTED

from .test_filters import good

RECORDED = json.loads((Path(__file__).parent / "message.json").read_text("utf-8"))


class FakeApi:
    """Answers each request with the recorded response, its text replaced per family."""

    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []

    def __call__(self, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        self.params.append(params)
        family = len(self.params) - 1
        message = copy.deepcopy(RECORDED)
        message["content"][1]["text"] = json.dumps({"variants": good(FAMILIES[family], REQUESTED)})
        return message, f"req_{family}"


def test_recorded_response_parses() -> None:
    raw = generate.variants(generate.response_text(RECORDED))
    assert len(raw) == 2
    assert filters.schema(raw[0]) is not None
    assert filters.schema(raw[1]) is None


@pytest.mark.parametrize("reason", ["max_tokens", "refusal"])
def test_unfinished_responses_are_errors(reason: str) -> None:
    message = copy.deepcopy(RECORDED) | {"stop_reason": reason}
    with pytest.raises(RuntimeError):
        generate.response_text(message)


def test_variants_need_a_list() -> None:
    with pytest.raises(ValueError):
        generate.variants('{"other": []}')


def test_request_has_no_sampling_parameters() -> None:
    params = generate.request("some-model", FAMILIES[3])
    assert params["model"] == "some-model"
    assert not {"temperature", "top_p", "top_k"} & set(params)
    assert params["output_config"]["effort"] == generate.EFFORT
    assert params["output_config"]["format"]["schema"] == generate.SCHEMA


@pytest.mark.parametrize("family", FAMILIES)
def test_prompt_states_the_rules(family: Any) -> None:
    text = generate.user_prompt(family)
    assert f"Write {REQUESTED} different" in text
    for word in filters.BANNED + filters.FAMILY_BANNED.get(family.id, ()):
        assert f'"{word}"' in text
    if family.placeholder is None:
        assert "no placeholder" in text
    else:
        assert f"{{{family.placeholder.name}}}" in text
        assert all(f'"{v}"' in text for v in family.placeholder.values)


def test_generate_writes_and_rebuilds(tmp_path: Path) -> None:
    api = FakeApi()
    exchanges, records = generate.generate(api, "some-model")
    assert [r["request_id"] for r in records] == [f"req_{f.id}" for f in FAMILIES]
    assert all(r["model"] == "recorded-model" for r in records)
    assert records[0]["usage"] == {
        "input_tokens": 812,
        "output_tokens": 2304,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    generate.write(tmp_path, exchanges, records, "sdk-version")
    folder = tmp_path / generate.CREATIVE
    catalog = generate.load_catalog(tmp_path)
    assert [t["id"] for t in catalog["templates"]] == list(range(60))
    assert all(t["family"] == t["id"] % 6 for t in catalog["templates"])
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert len(manifest["variants"]) == 6 * REQUESTED
    assert manifest["requests"] == records
    results = json.loads((tmp_path / "results" / generate.RESULTS_FILE).read_text("utf-8"))
    assert results["templates"] == 60
    assert results["usage"]["output_tokens"] == 6 * 2304
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert generate.main(["--root", str(tmp_path), "--rebuild"]) == 0
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_main_refuses_without_model_or_twice(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv(generate.MODEL_ENV, raising=False)
    with pytest.raises(SystemExit):
        generate.main(["--root", str(tmp_path)])
    (tmp_path / generate.CREATIVE).mkdir(parents=True)
    (tmp_path / generate.CREATIVE / "exchanges.json").write_text("[]")
    with pytest.raises(SystemExit):
        generate.main(["--root", str(tmp_path), "--model", "m"])
