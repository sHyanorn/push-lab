"""Two-tower retrieval model with in-batch softmax and log-Q correction (METHODS Section 3)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

USER_FIELDS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "register_days_range",
    *[f"onehot_feat{i}" for i in range(18)],
]
ITEM_FIELDS = ["video_id", "author_id", "first_tag", "upload_type", "video_type", "duration_bucket"]
# Fields whose vocabulary comes from the training window; the others are static side information
WINDOW_FIELDS = {"user_id", "video_id", "author_id"}


@dataclass(frozen=True)
class Config:
    dim: int = 32
    field_dim: int = 16
    hidden: int = 64
    tau: float = 0.05
    lr: float = 1e-3
    batch: int = 1024
    epochs: int = 5
    seed: int = 0
    threads: int = 4


def item_side_features(videos: pd.DataFrame) -> pd.DataFrame:
    """Static item features: first tag and duration decile bucket."""
    out = videos[["video_id", "author_id", "upload_type", "video_type"]].copy()
    out["first_tag"] = videos["tag"].fillna("").str.split(",").str[0]
    duration = videos["video_duration"]
    edges = np.nanquantile(duration, np.linspace(0.1, 0.9, 9))
    bucket = np.digitize(duration.fillna(-1.0), edges) + 1
    out["duration_bucket"] = np.where(duration.isna(), 0, bucket)
    return out


class Vocab:
    """Maps raw values to indices; index 0 is shared by values not seen at fit time."""

    def __init__(self, values: Sequence[object]) -> None:
        unique = pd.unique(pd.Series(list(values), dtype=object).dropna())
        self.index = {value: i + 1 for i, value in enumerate(unique)}

    def __len__(self) -> int:
        return len(self.index) + 1

    def encode(self, values: pd.Series) -> npt.NDArray[np.int64]:
        return values.map(self.index).fillna(0).to_numpy(dtype=np.int64)


class Tower(nn.Module):
    def __init__(self, cardinalities: Sequence[int], config: Config) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, config.field_dim) for n in cardinalities)
        self.mlp = nn.Sequential(
            nn.Linear(len(cardinalities) * config.field_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fields = [emb(x[:, k]) for k, emb in enumerate(self.embeddings)]
        out: torch.Tensor = nn.functional.normalize(self.mlp(torch.cat(fields, dim=1)), dim=1)
        return out


def logq_softmax_loss(
    users: torch.Tensor,
    items: torch.Tensor,
    item_ids: torch.Tensor,
    log_q: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """In-batch softmax with log-Q correction and duplicate-item masking (Yi et al., 2019).

    ``users`` and ``items`` are (B, d) vectors of the B positive pairs, ``item_ids`` their
    item identities and ``log_q`` the log sampling probability of each row's item.
    """
    logits = users @ items.T / tau - log_q.unsqueeze(0)
    duplicate = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)
    off_diagonal = ~torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(duplicate & off_diagonal, float("-inf"))
    return -(logits.diagonal() - torch.logsumexp(logits, dim=1)).mean()


class TwoTower:
    """A fitted two-tower model at one cutoff."""

    def __init__(
        self, users: pd.DataFrame, items: pd.DataFrame, window: pd.DataFrame, config: Config
    ) -> None:
        self.config = config
        self.users = users.set_index("user_id", drop=False)
        self.items = items.set_index("video_id", drop=False)
        clicked = window.loc[window["is_click"] == 1, ["user_id", "video_id"]]
        # Training pairs; their ids define the id vocabularies, so unseen ids share index 0
        self.pairs = clicked.merge(items[["video_id", "author_id"]], on="video_id", how="left")
        self.vocabs: dict[str, Vocab] = {}
        for field in USER_FIELDS:
            source = self.pairs if field in WINDOW_FIELDS else users
            self.vocabs[field] = Vocab(source[field].tolist())
        for field in ITEM_FIELDS:
            source = self.pairs if field in WINDOW_FIELDS else items
            self.vocabs[field] = Vocab(source[field].tolist())
        # Seed before the towers draw their initial weights
        torch.manual_seed(config.seed)
        self.user_tower = Tower([len(self.vocabs[f]) for f in USER_FIELDS], config)
        self.item_tower = Tower([len(self.vocabs[f]) for f in ITEM_FIELDS], config)

    def _encode(self, table: pd.DataFrame, fields: Sequence[str]) -> torch.Tensor:
        columns = [self.vocabs[f].encode(table[f]) for f in fields]
        return torch.from_numpy(np.stack(columns, axis=1))

    def encode_users(self, user_ids: npt.NDArray[np.int64]) -> torch.Tensor:
        return self._encode(self.users.reindex(user_ids).assign(user_id=user_ids), USER_FIELDS)

    def encode_items(self, item_ids: npt.NDArray[np.int64]) -> torch.Tensor:
        return self._encode(self.items.reindex(item_ids).assign(video_id=item_ids), ITEM_FIELDS)

    def fit(self) -> list[float]:
        """Train on the window's clicked pairs; returns the mean loss of each epoch."""
        cfg = self.config
        torch.set_num_threads(cfg.threads)
        item_ids = self.pairs["video_id"].to_numpy(dtype=np.int64)
        counts = pd.Series(item_ids).value_counts()
        log_q = np.log(pd.Series(item_ids).map(counts).to_numpy(dtype=np.float64) / len(item_ids))
        x_user = self.encode_users(self.pairs["user_id"].to_numpy(dtype=np.int64))
        x_item = self.encode_items(item_ids)
        ids = torch.from_numpy(item_ids.copy())
        logq = torch.from_numpy(log_q).float()
        params = [*self.user_tower.parameters(), *self.item_tower.parameters()]
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
        generator = torch.Generator().manual_seed(cfg.seed)
        losses = []
        for _ in range(cfg.epochs):
            order = torch.randperm(len(ids), generator=generator)
            total, batches = 0.0, 0
            for start in range(0, len(order), cfg.batch):
                rows = order[start : start + cfg.batch]
                loss = logq_softmax_loss(
                    self.user_tower(x_user[rows]),
                    self.item_tower(x_item[rows]),
                    ids[rows],
                    logq[rows],
                    cfg.tau,
                )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                total += loss.item()
                batches += 1
            losses.append(total / batches)
        return losses

    @torch.no_grad()
    def user_vectors(self, user_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        self.user_tower.eval()
        out: npt.NDArray[np.float32] = self.user_tower(self.encode_users(user_ids)).numpy()
        return out

    @torch.no_grad()
    def item_vectors(self, item_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        self.item_tower.eval()
        out: npt.NDArray[np.float32] = self.item_tower(self.encode_items(item_ids)).numpy()
        return out
