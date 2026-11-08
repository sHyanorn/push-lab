"""Fine-ranker features at a cutoff (METHODS Section 5).

Every statistic comes from the standard-log window before the cutoff. Pair features are
looked up with index arrays, so millions of (user, item) pairs can be built at once.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from push_lab.rank import splits
from push_lab.recall.two_tower import TwoTower

ALPHA = 20.0

USER_CATEGORICAL = [
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "register_days_range",
    *[f"onehot_feat{i}" for i in range(18)],
]
ITEM_CATEGORICAL = ["upload_type", "video_type", "first_tag"]
CATEGORICAL = [*USER_CATEGORICAL, *ITEM_CATEGORICAL]

IntArray = npt.NDArray[np.int64]


def smoothed(k: pd.Series, n: pd.Series, mu: float, alpha: float = ALPHA) -> pd.Series:
    """Rate shrunk toward ``mu``: (k + alpha * mu) / (n + alpha)."""
    return (k + alpha * mu) / (n + alpha)


def _rates(window: pd.DataFrame, keys: list[str], mu_click: float, mu_long: float) -> pd.DataFrame:
    grouped = window.groupby(keys).agg(
        n=("is_click", "size"), clicks=("is_click", "sum"), longs=("long_view", "sum")
    )
    grouped["click_rate"] = smoothed(grouped["clicks"], grouped["n"], mu_click)
    grouped["long_rate"] = smoothed(grouped["longs"], grouped["n"], mu_long)
    return grouped


@dataclass
class CutoffStats:
    """Lookup tables for one cutoff."""

    cutoff: int
    mu_click: float
    mu_long: float
    user: pd.DataFrame
    item: pd.DataFrame
    user_author: pd.Series
    user_tag: pd.Series
    popularity_rank: pd.Series


class FeatureBuilder:
    def __init__(
        self,
        window: pd.DataFrame,
        cutoff: int,
        users: pd.DataFrame,
        items: pd.DataFrame,
        tower: TwoTower,
    ) -> None:
        """``items`` holds the static item features from ``two_tower.item_side_features``."""
        if (window["date"] >= cutoff).any():
            raise ValueError("the window contains rows on or after the cutoff")
        self.tower = tower
        self.users = users.set_index("user_id")
        self.items = items.set_index("video_id")
        self.stats = self._stats(window, cutoff)
        self.categories = {
            **{c: pd.unique(users[c].dropna()) for c in USER_CATEGORICAL},
            **{c: pd.unique(items[c].dropna()) for c in ITEM_CATEGORICAL},
        }

    def _stats(self, window: pd.DataFrame, cutoff: int) -> CutoffStats:
        days = splits.window_days(cutoff)
        mu_click = float(window["is_click"].mean()) if len(window) else 0.0
        mu_long = float(window["long_view"].mean()) if len(window) else 0.0
        log = window.merge(
            self.items[["author_id", "first_tag"]], left_on="video_id", right_index=True
        )
        user = _rates(log, ["user_id"], mu_click, mu_long)
        user["per_day"] = user["n"] / days
        item = _rates(log, ["video_id"], mu_click, mu_long)
        item["per_day"] = item["n"] / days
        ua = _rates(log, ["user_id", "author_id"], mu_click, mu_long)["click_rate"]
        ut = _rates(log, ["user_id", "first_tag"], mu_click, mu_long)["click_rate"]
        clicks = log.groupby("video_id")["is_click"].sum()
        rank = clicks.rank(ascending=False, method="min")
        return CutoffStats(cutoff, mu_click, mu_long, user, item, ua, ut, rank)

    def build_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Feature rows for the ``user_id`` and ``video_id`` columns of ``rows``."""
        return self.build(
            rows["user_id"].to_numpy(dtype=np.int64), rows["video_id"].to_numpy(dtype=np.int64)
        )

    def build(self, user_ids: IntArray, item_ids: IntArray) -> pd.DataFrame:
        """Feature rows for the (user, item) pairs, in order."""
        s = self.stats
        u = self.users.reindex(user_ids)
        i = self.items.reindex(item_ids)
        out = pd.DataFrame(index=pd.RangeIndex(len(user_ids)))
        for c in USER_CATEGORICAL:
            out[c] = pd.Categorical(u[c].to_numpy(), categories=self.categories[c])
        for c in ("follow_user_num", "fans_user_num", "friend_user_num"):
            out[f"log_{c}"] = np.log1p(u[c].to_numpy(dtype=float))
        out["register_days"] = u["register_days"].to_numpy(dtype=float)
        us = s.user.reindex(user_ids)
        out["user_per_day"] = us["per_day"].fillna(0.0).to_numpy()
        out["user_click_rate"] = us["click_rate"].fillna(s.mu_click).to_numpy()
        out["user_long_rate"] = us["long_rate"].fillna(s.mu_long).to_numpy()

        for c in ITEM_CATEGORICAL:
            out[c] = pd.Categorical(i[c].to_numpy(), categories=self.categories[c])
        out["video_duration"] = i["video_duration"].to_numpy(dtype=float)
        cutoff_day = pd.Timestamp(splits.to_date(s.cutoff))
        upload = pd.to_datetime(i["upload_dt"], format="%Y-%m-%d", errors="coerce")
        out["days_since_upload"] = (cutoff_day - upload).dt.days.to_numpy(dtype=float)
        its = s.item.reindex(item_ids)
        out["item_per_day"] = its["per_day"].fillna(0.0).to_numpy()
        out["item_click_rate"] = its["click_rate"].fillna(s.mu_click).to_numpy()
        out["item_long_rate"] = its["long_rate"].fillna(s.mu_long).to_numpy()

        authors = i["author_id"].to_numpy()
        tags = i["first_tag"].to_numpy()
        out["user_author_click_rate"] = (
            s.user_author.reindex(pd.MultiIndex.from_arrays([user_ids, authors]))
            .fillna(s.mu_click)
            .to_numpy()
        )
        out["user_tag_click_rate"] = (
            s.user_tag.reindex(pd.MultiIndex.from_arrays([user_ids, tags]))
            .fillna(s.mu_click)
            .to_numpy()
        )
        out["popularity_rank"] = (
            s.popularity_rank.reindex(item_ids).fillna(len(s.popularity_rank) + 1).to_numpy()
        )
        out["tower_score"] = self._tower_scores(user_ids, item_ids)
        return out

    def _tower_scores(self, user_ids: IntArray, item_ids: IntArray) -> npt.NDArray[np.float32]:
        uniq_u, inv_u = np.unique(user_ids, return_inverse=True)
        uniq_i, inv_i = np.unique(item_ids, return_inverse=True)
        uv = self.tower.user_vectors(uniq_u)
        iv = self.tower.item_vectors(uniq_i)
        scores: npt.NDArray[np.float32] = np.einsum("nd,nd->n", uv[inv_u], iv[inv_i])
        return scores
