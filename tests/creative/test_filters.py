import random
import string

import pytest

from push_lab.creative import families as fam
from push_lab.creative import filters
from push_lab.creative.filters import Variant

CREATOR, POPULAR, PICK, LENGTH, BACK, TIME = fam.FAMILIES


def good(family: fam.Family, n: int) -> list[dict[str, str]]:
    """``n`` variants of random lowercase words that pass every rule and differ enough."""
    rng = random.Random(family.id)
    tag = "" if family.placeholder is None else f" {{{family.placeholder.name}}}"

    def words(k: int) -> str:
        return " ".join("".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(k))

    return [{"title": f"Go {words(2)}{tag}", "body": f"{words(4)}."} for _ in range(n)]


def test_placeholder_values_fill_and_longest() -> None:
    assert fam.DURATION.longest == "under 30 seconds"
    assert fam.creator_videos(1) == "1 video"
    assert fam.creator_videos(35) == "20+ videos"
    assert [fam.top_n(r) for r in (1, 10, 11, 50, 51, 100)] == [
        "top 10",
        "top 10",
        "top 50",
        "top 50",
        "top 100",
        "top 100",
    ]
    assert [fam.duration(ms) for ms in (29_999, 30_000, 59_999, 179_999, 599_999, 600_000)] == [
        "under 30 seconds",
        "under a minute",
        "under a minute",
        "1 to 3 minutes",
        "3 to 10 minutes",
        "over 10 minutes",
    ]
    week = "a week or more"
    assert [fam.days_away(d) for d in (2, 6, 7, 20)] == ["2 days", "6 days", week, week]
    assert fam.time_of_day(3) == "tonight"
    with pytest.raises(ValueError):
        fam.creator_videos(0)
    with pytest.raises(ValueError):
        fam.top_n(101)
    with pytest.raises(ValueError):
        fam.days_away(1)


def test_template_ids_keep_family_mod_six() -> None:
    ids = [fam.template_id(f, j) for j in range(10) for f in range(6)]
    assert sorted(ids) == list(range(60))
    assert all(fam.template_id(f, j) % 6 == f for f in range(6) for j in range(10))


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "text",
        {"title": "x"},
        {"title": "a", "body": ""},
        {"title": 1, "body": "b"},
        {"title": "a", "body": "b", "extra": "c"},
    ],
)
def test_schema_rejects(raw: object) -> None:
    assert filters.schema(raw) is None


def test_charset() -> None:
    assert filters.charset(Variant("Plain title", "Body text."))
    assert not filters.charset(Variant("Don\u2019t stop", "Body"))
    assert not filters.charset(Variant("Title", "line\nbreak"))
    assert not filters.charset(Variant("Title \U0001f525", "Body"))


def test_placeholders() -> None:
    assert filters.placeholders(Variant("Watch {duration}", "A clip."), LENGTH)
    assert not filters.placeholders(Variant("Watch", "A clip."), LENGTH)
    assert not filters.placeholders(Variant("{duration}", "{duration}"), LENGTH)
    assert not filters.placeholders(Variant("{top_n}", "A clip."), LENGTH)
    assert not filters.placeholders(Variant("{duration} }", "A clip."), LENGTH)
    assert not filters.placeholders(Variant("{duration", "A clip."), LENGTH)
    assert filters.placeholders(Variant("A pick", "For you."), PICK)
    assert not filters.placeholders(Variant("A pick {time_of_day}", "For you."), PICK)


def test_length_uses_the_longest_value() -> None:
    title = "A clip that takes {duration}"
    filled = len(fam.render(title, {"duration": fam.DURATION.longest}))
    pad = "x" * (fam.TITLE_LENGTH[1] - filled)
    assert filters.length(Variant(title + pad, "x" * 20), LENGTH)
    assert not filters.length(Variant(title + pad + "x", "x" * 20), LENGTH)
    assert not filters.length(Variant("Short", "x" * 20), PICK)
    assert not filters.length(Variant("Long enough", "x" * 19), PICK)
    assert not filters.length(Variant("Long enough", "x" * 101), PICK)


@pytest.mark.parametrize(
    ("title", "body", "family", "word"),
    [
        ("Watch 3 clips", "Fine body text here.", PICK, "digit"),
        ("Hurry up", "Fine body text here.", PICK, "hurry"),
        ("A pick", "Don't miss this one, it is good.", PICK, "don't miss"),
        ("A free pick", "Fine body text here.", PICK, "free"),
        ("Claim your rewards", "Fine body text here.", PICK, "reward"),
        ("Prizes inside", "Fine body text here.", PICK, "prize"),
        ("A pick", "You won't believe this video.", PICK, "you won't believe"),
        ("We miss you", "Fine body text here.", PICK, "miss you"),
        ("A new one", "From a creator you watched {creator_videos} of.", CREATOR, "new"),
        ("Popular today", "Among the {top_n} videos lately.", POPULAR, "today"),
        ("Wow!", "So good!", PICK, "!"),
        ("A GREAT pick", "Fine body text here.", PICK, "capitals"),
        ("A #pick", "Fine body text here.", PICK, "#"),
    ],
)
def test_banned(title: str, body: str, family: fam.Family, word: str) -> None:
    assert filters.banned(Variant(title, body), family) == word


@pytest.mark.parametrize(
    ("title", "body", "family"),
    [
        ("Winter is here", "A wintry clip with freestyle skiing.", PICK),
        ("A new one", "Picked for you with care.", PICK),
        ("Popular today", "One of the {time_of_day} picks.", TIME),
        ("Watch it {duration}", "A good one, TV style!", LENGTH),
        ("Back after {days_away}", "Here is a video for you.", BACK),
    ],
)
def test_not_banned(title: str, body: str, family: fam.Family) -> None:
    assert filters.banned(Variant(title, body), family) is None


def test_near_duplicates_ignore_case_and_placeholder_names() -> None:
    a = filters.ngrams(filters.normalized(Variant("Your {duration} watch", "A clip for you.")))
    b = filters.ngrams(filters.normalized(Variant("YOUR {top_n} WATCH", "A clip for you.")))
    assert filters.jaccard(a, b) == 1.0
    c = filters.ngrams(filters.normalized(Variant("Something else", "Entirely different words.")))
    assert filters.jaccard(a, c) < filters.DUPLICATE_JACCARD


def test_select_keeps_ten_per_family_in_order() -> None:
    responses = [good(f, 16) for f in fam.FAMILIES]
    # Family 2: the first variant fails the schema, the second is banned
    responses[2][0] = {"title": "only a title"}
    responses[2][1] = {"title": "A free pick for you", "body": "Something to watch, any style."}
    sel = filters.select(responses)
    assert sorted(sel.catalog) == list(range(60))
    assert all(sel.catalog[k][0] == k % 6 for k in sel.catalog)
    family2 = [v for v in sel.verdicts if v.family == 2]
    assert [v.failed for v in family2[:2]] == ["schema", "banned"]
    assert family2[1].detail == "free"
    assert [v.template for v in family2[2:12]] == [fam.template_id(2, j) for j in range(10)]
    counts = sel.counts()["Personal pick"]
    assert counts["returned"] == 16 and counts["kept"] == 10 and counts["passed"] == 14


def test_select_drops_duplicates_across_families() -> None:
    responses = [good(f, 12) for f in fam.FAMILIES]
    # A copy of family 0's first variant in family 3, with family 3's placeholder
    first = responses[0][0]
    responses[3][0] = {
        "title": first["title"].replace("{creator_videos}", "{duration}"),
        "body": first["body"],
    }
    sel = filters.select(responses)
    family3 = [v for v in sel.verdicts if v.family == 3]
    assert family3[0].failed == "near_duplicate"
    assert sel.counts()["Short or long watch"]["kept"] == 10


def test_select_needs_one_response_per_family() -> None:
    with pytest.raises(ValueError):
        filters.select([[]])


def test_self_check_rejects_every_violation() -> None:
    check = filters.self_check()
    assert check == {
        "cases": len(filters.VIOLATIONS),
        "rejected_as_expected": len(filters.VIOLATIONS),
    }
