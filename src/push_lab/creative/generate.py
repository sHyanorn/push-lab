"""One-off copy generation with the Claude API (METHODS 13.2).

``python -m push_lab.creative.generate --model <id>`` sends one request per family and writes
the prompts, the responses as returned, the catalog and the manifest to ``data/creative/``,
and the filter counts to ``results/copy.json``. The model comes from ``--model`` or
``PUSH_LAB_COPY_MODEL`` and the key from ``ANTHROPIC_API_KEY``; neither is stored.
``--rebuild`` redoes the catalog and counts from the committed responses without the API.
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from push_lab.creative import filters
from push_lab.creative.families import (
    BODY_LENGTH,
    FAMILIES,
    KEPT_PER_FAMILY,
    REQUESTED,
    TITLE_LENGTH,
    Family,
)
from push_lab.data import paths

CREATIVE = Path("data/creative")
RESULTS_FILE = "copy.json"
MODEL_ENV = "PUSH_LAB_COPY_MODEL"
EFFORT = "high"
MAX_TOKENS = 16_000
THINKING = {"type": "adaptive"}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["variants"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write push notification copy for a short-video app. Each push recommends one video "
    "that a recommender picked for the user. The app knows very little about the video: it has "
    "no title, no caption and no creator name, so the copy must not describe the video's "
    "content. Write plain, friendly, honest English. A push has a title and a body."
)

# Sends one request and returns the response as a dict and the request id
Send = Callable[[dict[str, Any]], tuple[dict[str, Any], str | None]]


def _quoted(words: Sequence[str]) -> str:
    return ", ".join(f'"{w}"' for w in words)


def user_prompt(family: Family) -> str:
    banned = filters.BANNED + filters.FAMILY_BANNED.get(family.id, ())
    p = family.placeholder
    if p is None:
        slot = "Use no placeholder and no curly braces."
        longest = ""
    else:
        slot = (
            f"Use the placeholder {{{p.name}}} exactly once, in the title or the body, written "
            f"exactly like that with curly braces. At send time it is replaced by one of: "
            f"{_quoted(p.values)}. The sentence must read correctly with every one of them."
        )
        longest = f' Count the placeholder as its longest value, "{p.longest}".'
    lines = [
        f"Template family: {family.name}. {family.brief}",
        "",
        f"Write {REQUESTED} different push notifications for this family.",
        "",
        "Rules:",
        f"- {slot}",
        f"- Title {TITLE_LENGTH[0]} to {TITLE_LENGTH[1]} characters, body {BODY_LENGTH[0]} "
        f"to {BODY_LENGTH[1]} characters.{longest}",
        "- Plain ASCII only: straight quotes, no emoji, no line breaks.",
        "- No digits; numbers may only come from the placeholder. No '#'.",
        "- At most one exclamation mark in title and body together. No words in capitals.",
        "- Claim nothing the app cannot know: no invented numbers or facts, no false urgency, "
        "no rewards or prizes, no guilt.",
        f"- Do not use these words or phrases: {_quoted(banned)}.",
        "- Make the variants clearly different from each other in wording and angle.",
        "",
        'Return JSON: {"variants": [{"title": ..., "body": ...}, ...]}.',
    ]
    return "\n".join(lines)


def request(model: str, family: Family) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_prompt(family)}],
        "thinking": THINKING,
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def response_text(message: dict[str, Any]) -> str:
    """The JSON text of a finished response; anything but a normal end is an error."""
    if message.get("stop_reason") != "end_turn":
        raise RuntimeError(
            f"response {message.get('id')} stopped with {message.get('stop_reason')}"
        )
    texts = [b["text"] for b in message["content"] if b.get("type") == "text"]
    if len(texts) != 1:
        raise RuntimeError(f"response {message.get('id')} has {len(texts)} text blocks")
    return str(texts[0])


def variants(text: str) -> list[Any]:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("variants"), list):
        raise ValueError("the response does not hold a list of variants")
    out: list[Any] = data["variants"]
    return out


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(message: dict[str, Any], request_id: str | None, family: Family) -> dict[str, Any]:
    usage = message.get("usage") or {}
    return {
        "family": family.id,
        "model": message.get("model"),
        "message_id": message.get("id"),
        "request_id": request_id,
        "stop_reason": message.get("stop_reason"),
        "usage": {k: v for k, v in usage.items() if isinstance(v, int)},
        "received_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def generate(send: Send, model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prompts with the returned text, and one manifest record per request."""
    exchanges, records = [], []
    for family in FAMILIES:
        params = request(model, family)
        message, request_id = send(params)
        records.append(_record(message, request_id, family))
        exchanges.append(
            {
                "family": family.id,
                "name": family.name,
                "system": params["system"],
                "user": params["messages"][0]["content"],
                "text": response_text(message),
            }
        )
    return exchanges, records


def build(
    exchanges: list[dict[str, Any]], records: list[dict[str, Any]], sdk: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Catalog, manifest and results from the exchanges."""
    selection = filters.select([variants(e["text"]) for e in exchanges])
    catalog = {
        "families": [f.name for f in FAMILIES],
        "placeholders": {
            f.placeholder.name: list(f.placeholder.values) for f in FAMILIES if f.placeholder
        },
        "templates": [
            {"id": k, "family": f, "title": v.title, "body": v.body}
            for k, (f, v) in sorted(selection.catalog.items())
        ],
    }
    manifest = {
        "sdk": sdk,
        "effort": EFFORT,
        "max_tokens": MAX_TOKENS,
        "thinking": THINKING,
        "schema_sha256": _sha256(json.dumps(SCHEMA, sort_keys=True)),
        "requests": records,
        "variants": [
            {
                "family": v.family,
                "index": v.index,
                "sha256": _sha256(f"{v.raw.get('title')}\n{v.raw.get('body')}")
                if isinstance(v.raw, dict)
                else _sha256(json.dumps(v.raw)),
                "failed": v.failed,
                "detail": v.detail,
                "template": v.template,
            }
            for v in selection.verdicts
        ],
    }
    usage: dict[str, int] = {}
    for r in records:
        for k, n in r["usage"].items():
            usage[k] = usage.get(k, 0) + n
    results = {
        "templates": len(selection.catalog),
        "requested_per_family": REQUESTED,
        "kept_per_family": KEPT_PER_FAMILY,
        "filters": list(filters.RULES),
        "counts": selection.counts(),
        "banned_matches": _banned_counts(selection),
        "usage": usage,
        "filter_self_check": filters.self_check(),
        "note": "Descriptive counts of one generation run; no simulator result reads the text. "
        "The counts are recomputed from the committed responses, and the filters reject every "
        "hand-made violation in filter_self_check. The prompt listed the limits and banned "
        "phrasings, so a run where every variant passes is possible; then the agreement between "
        "the rater and the filters cannot be estimated.",
    }
    return catalog, manifest, results


def _banned_counts(selection: filters.Selection) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in selection.verdicts:
        if v.failed == "banned" and v.detail is not None:
            out[v.detail] = out.get(v.detail, 0) + 1
    return dict(sorted(out.items()))


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def write(
    root: Path, exchanges: list[dict[str, Any]], records: list[dict[str, Any]], sdk: str
) -> None:
    catalog, manifest, results = build(exchanges, records, sdk)
    folder = root / CREATIVE
    _write(folder / "exchanges.json", exchanges)
    _write(folder / "catalog.json", catalog)
    _write(folder / "manifest.json", manifest)
    out = root / paths.RESULTS / RESULTS_FILE
    if out.exists():
        # Rating counts come from a person, not from the responses; keep them
        previous = json.loads(out.read_text("utf-8"))
        if "rating" in previous:
            results["rating"] = previous["rating"]
    _write(out, results)


def load_catalog(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((root / CREATIVE / "catalog.json").read_text("utf-8"))
    return out


def _api_sender() -> tuple[Send, str]:
    import anthropic

    client = anthropic.Anthropic()

    def send(params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        message = client.messages.create(**params)
        return message.to_dict(), message._request_id

    return send, anthropic.__version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the copy once, or rebuild its catalog.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--model", default=os.environ.get(MODEL_ENV), help=f"or ${MODEL_ENV}")
    parser.add_argument("--rebuild", action="store_true", help="reuse the committed responses")
    args = parser.parse_args(argv)
    folder = args.root / CREATIVE
    if args.rebuild:
        exchanges = json.loads((folder / "exchanges.json").read_text("utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
        write(args.root, exchanges, manifest["requests"], manifest["sdk"])
        print(f"rebuilt {folder}")
        return 0
    if not args.model:
        parser.error(f"pass --model or set {MODEL_ENV}")
    if (folder / "exchanges.json").exists():
        parser.error("copy was already generated; use --rebuild (D011)")
    send, sdk = _api_sender()
    exchanges, records = generate(send, args.model)
    write(args.root, exchanges, records, sdk)
    print(f"wrote {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
