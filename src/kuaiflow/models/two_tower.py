"""PyTorch two-tower retrieval with in-batch negative training."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from kuaiflow.models.common import make_id_map, positive_pairs, top_k_indices


class _Tower(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(self.embedding(indices)), dim=-1)


class TwoTowerRecommender:
    """ID-only retrieval baseline trained with an in-batch softmax objective."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        epochs: int = 10,
        batch_size: int = 512,
        seed: int = 2026,
        device: str = "cpu",
    ) -> None:
        if min(embedding_dim, hidden_dim, epochs, batch_size) <= 0:
            raise ValueError(
                "embedding_dim, hidden_dim, epochs, and batch_size must be positive"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)

    def fit(
        self,
        interactions: pd.DataFrame,
        label_col: str = "is_click",
        user_col: str = "user_id",
        item_col: str = "video_id",
    ) -> "TwoTowerRecommender":
        positives = positive_pairs(interactions, label_col, user_col, item_col)
        if positives.empty:
            raise ValueError("Two-tower retrieval requires positive interactions")

        self.user_ids, self.user_to_index = make_id_map(interactions[user_col])
        self.item_ids, self.item_to_index = make_id_map(interactions[item_col])
        users = positives[user_col].map(self.user_to_index).to_numpy(dtype=np.int64)
        items = positives[item_col].map(self.item_to_index).to_numpy(dtype=np.int64)
        self.user_seen: dict[Hashable, set[int]] = {
            user: set(group[item_col].map(self.item_to_index).tolist())
            for user, group in positives.groupby(user_col, sort=False)
        }
        self.popularity = np.bincount(items, minlength=len(self.item_ids)).astype(float)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.user_tower = _Tower(
            len(self.user_ids), self.embedding_dim, self.hidden_dim
        ).to(self.device)
        self.item_tower = _Tower(
            len(self.item_ids), self.embedding_dim, self.hidden_dim
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            [*self.user_tower.parameters(), *self.item_tower.parameters()],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        order = np.arange(len(users))
        self.training_history: list[float] = []
        self.user_tower.train()
        self.item_tower.train()
        for _ in range(self.epochs):
            rng.shuffle(order)
            total_loss = 0.0
            total_examples = 0
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                batch_users = torch.as_tensor(users[batch], device=self.device)
                batch_items = torch.as_tensor(items[batch], device=self.device)
                user_vectors = self.user_tower(batch_users)
                item_vectors = self.item_tower(batch_items)
                logits = user_vectors @ item_vectors.T / self.temperature

                # Repeated positive items are not negatives for one another.
                duplicate_items = batch_items[:, None].eq(batch_items[None, :])
                duplicate_items.fill_diagonal_(False)
                logits = logits.masked_fill(duplicate_items, -torch.inf)

                targets = torch.arange(len(batch), device=self.device)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach()) * len(batch)
                total_examples += len(batch)
            self.training_history.append(total_loss / total_examples)

        self._refresh_item_vectors()
        return self

    @torch.no_grad()
    def _refresh_item_vectors(self) -> None:
        self.item_tower.eval()
        indices = torch.arange(len(self.item_ids), device=self.device)
        self.item_vectors = self.item_tower(indices).cpu().numpy()

    @torch.no_grad()
    def recommend(
        self, user_ids: Iterable[Hashable], k: int
    ) -> dict[Hashable, list[Hashable]]:
        if not hasattr(self, "item_vectors"):
            raise RuntimeError("Call fit before recommend")
        output: dict[Hashable, list[Hashable]] = {}
        popularity_scale = max(float(self.popularity.max()), 1.0)
        fallback = self.popularity / popularity_scale
        self.user_tower.eval()

        for user in user_ids:
            if user in self.user_seen:
                index = torch.tensor([self.user_to_index[user]], device=self.device)
                user_vector = self.user_tower(index).cpu().numpy()[0]
                scores = self.item_vectors @ user_vector
                scores = scores + 1e-8 * fallback
            else:
                scores = fallback.copy()
            seen = self.user_seen.get(user, set())
            if seen:
                scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
            selected = top_k_indices(scores, k)
            output[user] = [self.item_ids[index] for index in selected]
        return output
