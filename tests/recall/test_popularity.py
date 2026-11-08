import pandas as pd

from push_lab.recall import popularity


def _log() -> pd.DataFrame:
    rows = [
        # (date, video_id, is_click)
        (20220410, 1, 1),
        (20220410, 1, 1),
        (20220410, 1, 1),
        (20220419, 2, 1),
        (20220420, 2, 1),
        (20220420, 3, 1),
        (20220421, 4, 1),
        (20220410, 4, 1),
        (20220421, 5, 0),
    ]
    return pd.DataFrame(rows, columns=["date", "video_id", "is_click"])


def test_popularity_orders_by_clicks_then_id() -> None:
    assert popularity.popularity(_log()).tolist() == [1, 2, 4, 3, 5]


def test_trending_uses_last_three_days_before_cutoff() -> None:
    # Cutoff 04-22: recent days are 04-19 to 04-21
    ranked = popularity.trending(_log(), 20220422)
    assert ranked.tolist() == [2, 4, 3, 1, 5]
