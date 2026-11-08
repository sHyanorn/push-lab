"""Copy families, their placeholders and the values that fill them (METHODS 13.1)."""

import re
from dataclasses import dataclass

REQUESTED = 16
KEPT_PER_FAMILY = 10
TITLE_LENGTH = (10, 50)
BODY_LENGTH = (20, 100)
PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


@dataclass(frozen=True)
class Placeholder:
    name: str
    values: tuple[str, ...]

    @property
    def longest(self) -> str:
        return max(self.values, key=len)


@dataclass(frozen=True)
class Family:
    id: int
    name: str
    # What the push is about, as the prompt states it
    brief: str
    placeholder: Placeholder | None


CREATOR_VIDEOS = Placeholder(
    "creator_videos", ("1 video", *(f"{n} videos" for n in range(2, 20)), "20+ videos")
)
TOP_N = Placeholder("top_n", ("top 10", "top 50", "top 100"))
DURATION = Placeholder(
    "duration",
    ("under 30 seconds", "under a minute", "1 to 3 minutes", "3 to 10 minutes", "over 10 minutes"),
)
DAYS_AWAY = Placeholder("days_away", (*(f"{n} days" for n in range(2, 7)), "a week or more"))
TIME_OF_DAY = Placeholder("time_of_day", ("this morning", "at lunch", "this evening", "tonight"))

FAMILIES = (
    Family(
        0,
        "Creator update",
        "A video by a creator whose videos the user has watched before. The creator has no "
        "name in the data, so the copy cannot name them; it can say how many of their videos "
        "the user watched.",
        CREATOR_VIDEOS,
    ),
    Family(
        1,
        "Popular now",
        "A video that is among the most watched videos on the app recently.",
        TOP_N,
    ),
    Family(
        2,
        "Personal pick",
        "A video picked for the user by the recommender. Nothing else is known about it.",
        None,
    ),
    Family(
        3,
        "Short or long watch",
        "A video, described by how long it takes to watch.",
        DURATION,
    ),
    Family(
        4,
        "Welcome back",
        "A video for a user who has not opened the app for some days.",
        DAYS_AWAY,
    ),
    Family(
        5,
        "Time of day",
        "A video suggested for the time of day the push is sent.",
        TIME_OF_DAY,
    ),
)
TEMPLATES = len(FAMILIES) * KEPT_PER_FAMILY


def template_id(family: int, rank: int) -> int:
    """Template k = 6j + f, so family(k) = k mod 6 (D036)."""
    return rank * len(FAMILIES) + family


def render(text: str, values: dict[str, str]) -> str:
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


# Fill values from the data (METHODS 13.1)


def creator_videos(clicks: int) -> str:
    if clicks < 1:
        raise ValueError("Creator update needs at least one click on the author")
    return CREATOR_VIDEOS.values[min(clicks, 20) - 1]


def top_n(rank: int) -> str:
    """``rank`` counts from 1 in the trending list."""
    if not 1 <= rank <= 100:
        raise ValueError("Popular now needs a trending rank from 1 to 100")
    return "top 10" if rank <= 10 else "top 50" if rank <= 50 else "top 100"


def duration(ms: float) -> str:
    seconds = ms / 1000
    for limit, value in zip((30, 60, 180, 600), DURATION.values, strict=False):
        if seconds < limit:
            return value
    return DURATION.values[-1]


def days_away(days: int) -> str:
    if days < 2:
        raise ValueError("Welcome back needs at least two days without a visit")
    return DAYS_AWAY.values[min(days, 7) - 2]


def time_of_day(slot: int) -> str:
    return TIME_OF_DAY.values[slot]
