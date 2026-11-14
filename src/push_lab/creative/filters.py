"""Rule-based copy filters, fixed before generation (METHODS 13.3)."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from push_lab.creative.families import (
    BODY_LENGTH,
    FAMILIES,
    KEPT_PER_FAMILY,
    PLACEHOLDER,
    TITLE_LENGTH,
    Family,
    render,
    template_id,
)

RULES = ("schema", "charset", "placeholders", "length", "banned", "near_duplicate")
DUPLICATE_JACCARD = 0.5
NGRAM = 4

URGENCY = (
    "hurry",
    "last chance",
    "don't miss",
    "do not miss",
    "before it's gone",
    "ending soon",
    "ends soon",
    "expires",
    "limited",
    "act now",
    "right now",
    "urgent",
    "only today",
    "countdown",
)
INDUCEMENT = (
    "free",
    "win",
    "winner",
    "prize",
    "reward",
    "bonus",
    "coins",
    "gift",
    "cash",
    "guaranteed",
    "unlock",
)
CLICKBAIT = (
    "you won't believe",
    "shocking",
    "secret",
    "everyone",
    "millions",
    "viral",
    "breaking",
    "exclusive",
)
GUILT = ("you missed", "miss you", "we miss", "don't leave", "come back now")
# Claims the data cannot back, per family id
FAMILY_BANNED = {
    0: ("new", "just", "latest", "posted", "uploaded"),
    1: ("today", "this hour"),
}
BANNED = URGENCY + INDUCEMENT + CLICKBAIT + GUILT


def _phrase(words: str) -> re.Pattern[str]:
    # Whole words, plurals included ("rewards", "bonuses"); -es only where English takes it,
    # so "win" does not catch "wines"
    plural = "(?:es)?" if words.endswith(("s", "sh", "ch", "x", "z")) else "s?"
    return re.compile(rf"(?<![a-z]){re.escape(words)}{plural}(?![a-z])", re.IGNORECASE)


_BANNED = {w: _phrase(w) for w in BANNED}
_FAMILY_BANNED = {f: {w: _phrase(w) for w in ws} for f, ws in FAMILY_BANNED.items()}
_CAPITALS = re.compile(r"(?<![A-Za-z])[A-Z]{3,}(?![A-Za-z])")


@dataclass(frozen=True)
class Variant:
    title: str
    body: str


def schema(raw: Any) -> Variant | None:
    """The variant if ``raw`` is an object with exactly a non-empty title and body."""
    if not isinstance(raw, dict) or set(raw) != {"title", "body"}:
        return None
    title, body = raw["title"], raw["body"]
    if not (isinstance(title, str) and isinstance(body, str) and title and body):
        return None
    return Variant(title, body)


def charset(v: Variant) -> bool:
    return all(" " <= c <= "~" for c in v.title + v.body)


def placeholders(v: Variant, family: Family) -> bool:
    """Each field on its own, so braces cannot open in the title and close in the body."""
    names = PLACEHOLDER.findall(v.title) + PLACEHOLDER.findall(v.body)
    rest = PLACEHOLDER.sub("", v.title) + PLACEHOLDER.sub("", v.body)
    if "{" in rest or "}" in rest:
        return False
    if family.placeholder is None:
        return not names
    return names == [family.placeholder.name]


def longest(v: Variant, family: Family) -> Variant:
    p = family.placeholder
    values = {} if p is None else {p.name: p.longest}
    return Variant(render(v.title, values), render(v.body, values))


def length(v: Variant, family: Family) -> bool:
    full = longest(v, family)
    return (
        TITLE_LENGTH[0] <= len(full.title) <= TITLE_LENGTH[1]
        and BODY_LENGTH[0] <= len(full.body) <= BODY_LENGTH[1]
    )


def banned(v: Variant, family: Family) -> str | None:
    """The first banned pattern found, or None."""
    bare = "\n".join(PLACEHOLDER.sub(" ", field) for field in (v.title, v.body))
    if re.search(r"\d", bare):
        return "digit"
    if "#" in bare:
        return "#"
    for word, pattern in (_BANNED | _FAMILY_BANNED.get(family.id, {})).items():
        if pattern.search(bare):
            return word
    if bare.count("!") > 1:
        return "!"
    if _CAPITALS.search(bare):
        return "capitals"
    return None


def normalized(v: Variant) -> str:
    text = PLACEHOLDER.sub("\x00", f"{v.title} | {v.body}".lower())
    return " ".join(text.split())


def ngrams(text: str, n: int = NGRAM) -> frozenset[str]:
    return frozenset(text[i : i + n] for i in range(max(1, len(text) - n + 1)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


@dataclass
class Verdict:
    family: int
    index: int
    raw: Any
    failed: str | None = None
    detail: str | None = None
    template: int | None = None


@dataclass
class Selection:
    verdicts: list[Verdict] = field(default_factory=list)
    catalog: dict[int, tuple[int, Variant]] = field(default_factory=dict)

    def counts(self) -> dict[str, dict[str, int]]:
        """Per family name: variants returned, dropped at each rule, survivors and kept."""
        out: dict[str, dict[str, int]] = {}
        for f in FAMILIES:
            mine = [v for v in self.verdicts if v.family == f.id]
            row = {"returned": len(mine)}
            row |= {rule: sum(v.failed == rule for v in mine) for rule in RULES}
            row["passed"] = sum(v.failed is None for v in mine)
            row["kept"] = sum(v.template is not None for v in mine)
            out[f.name] = row
        return out


def select(responses: Sequence[Sequence[Any]]) -> Selection:
    """Apply the rules in order to each family's variants (families in id order, variants in
    returned order). A near-duplicate is checked against every earlier variant that passed
    all rules; the first ``KEPT_PER_FAMILY`` survivors of a family form the catalog."""
    if len(responses) != len(FAMILIES):
        raise ValueError("one response per family is required")
    out = Selection()
    survivors: list[frozenset[str]] = []
    for family, raws in zip(FAMILIES, responses, strict=True):
        kept = 0
        for i, raw in enumerate(raws):
            verdict = Verdict(family.id, i, raw)
            out.verdicts.append(verdict)
            v = schema(raw)
            if v is None:
                verdict.failed = "schema"
                continue
            if not charset(v):
                verdict.failed = "charset"
            elif not placeholders(v, family):
                verdict.failed = "placeholders"
            elif not length(v, family):
                verdict.failed = "length"
            elif (word := banned(v, family)) is not None:
                verdict.failed, verdict.detail = "banned", word
            else:
                grams = ngrams(normalized(v))
                if any(jaccard(grams, g) >= DUPLICATE_JACCARD for g in survivors):
                    verdict.failed = "near_duplicate"
                else:
                    survivors.append(grams)
                    if kept < KEPT_PER_FAMILY:
                        verdict.template = template_id(family.id, kept)
                        out.catalog[verdict.template] = (family.id, v)
                        kept += 1
    return out


# One hand-made violation per rule, applied to every committed run as a check that the
# filters reject (METHODS 13.3)
VIOLATIONS: tuple[tuple[int, Any, str], ...] = (
    (2, {"title": "Missing body"}, "schema"),
    (2, {"title": "Don\u2019t stop now", "body": "A curly quote is not plain ASCII."}, "charset"),
    (
        3,
        {"title": "No placeholder here", "body": "The family needs its duration slot."},
        "placeholders",
    ),
    (3, {"title": "A clip of {dura", "body": "tion} to watch on the way."}, "placeholders"),
    (2, {"title": "A pick for you", "body": "Too short"}, "length"),
    (2, {"title": "Hurry, last chance", "body": "This video is gone in a moment."}, "banned"),
    (2, {"title": "Win a prize today", "body": "Open the app to claim your reward."}, "banned"),
    (2, {"title": "Bonuses are waiting", "body": "Open the app to see what is inside."}, "banned"),
    (2, {"title": "Watch 3 videos", "body": "Numbers may only come from placeholders."}, "banned"),
    (
        0,
        {"title": "A new upload", "body": "From a creator you watched {creator_videos} of."},
        "banned",
    ),
)


def self_check() -> dict[str, int]:
    """How many of ``VIOLATIONS`` each rule rejected, and how many slipped through."""
    out = {"cases": len(VIOLATIONS), "rejected_as_expected": 0}
    for family, raw, rule in VIOLATIONS:
        responses: list[list[Any]] = [[] for _ in FAMILIES]
        responses[family] = [raw]
        if select(responses).verdicts[0].failed == rule:
            out["rejected_as_expected"] += 1
    return out
