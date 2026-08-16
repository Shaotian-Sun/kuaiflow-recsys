"""Feature- and history-aware PyTorch two-tower retrieval."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from kuaiflow.models.common import make_id_map, positive_pairs, top_k_indices

USER_CATEGORICAL = ("user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author", "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range")
USER_NUMERIC = ("follow_user_num", "fans_user_num", "friend_user_num", "register_days")
VIDEO_CATEGORICAL = ("author_id", "video_type", "upload_type", "visible_status", "music_id", "music_type", "tag")
VIDEO_NUMERIC = ("video_duration", "server_width", "server_height")


class _FeatureEncoder(nn.Module):
    def __init__(self, cardinalities: Sequence[int], numeric_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, feature_dim) for n in cardinalities)
        input_dim = feature_dim * len(cardinalities) + numeric_dim
        self.output_dim = feature_dim if input_dim else 0
        self.projection = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.ReLU()) if input_dim else None

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        parts = [embedding(categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        if numeric.shape[1]:
            parts.append(numeric)
        return self.projection(torch.cat(parts, dim=-1))


class _ItemTower(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, hidden_dim: int, features: _FeatureEncoder) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(num_items, embedding_dim)
        self.features = features
        self.network = nn.Sequential(nn.Linear(embedding_dim + features.output_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))

    def forward(self, indices: torch.Tensor, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        parts = [self.id_embedding(indices)]
        if self.features.output_dim:
            parts.append(self.features(categorical, numeric))
        return F.normalize(self.network(torch.cat(parts, dim=-1)), dim=-1)


class _UserTower(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int, hidden_dim: int, features: _FeatureEncoder, use_history: bool) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(num_users, embedding_dim)
        self.features = features
        self.use_history = use_history
        history_dim = embedding_dim if use_history else 0
        self.network = nn.Sequential(nn.Linear(embedding_dim + history_dim + features.output_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))

    def forward(self, indices: torch.Tensor, categorical: torch.Tensor, numeric: torch.Tensor, history_vectors: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        mask = history_mask.unsqueeze(-1)
        history = (history_vectors * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        parts = [self.id_embedding(indices)]
        if self.use_history:
            parts.append(history)
        if self.features.output_dim:
            parts.append(self.features(categorical, numeric))
        return F.normalize(self.network(torch.cat(parts, dim=-1)), dim=-1)


class TwoTowerRecommender:
    """Two towers using IDs, static metadata, and prior positive item history.

    Training histories are causal: a target interaction can only see earlier
    positive interactions from the same user. Retrieval uses the latest training
    history. The item tower is shared by target and historical videos.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, feature_dim: int = 16, max_history: int = 20, use_history: bool = True, learning_rate: float = 1e-3, weight_decay: float = 1e-5, temperature: float = 0.07, epochs: int = 10, batch_size: int = 512, seed: int = 2026, device: str = "cpu") -> None:
        if min(embedding_dim, hidden_dim, feature_dim, max_history, epochs, batch_size) <= 0:
            raise ValueError("dimensions, max_history, epochs, and batch_size must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.embedding_dim, self.hidden_dim, self.feature_dim = embedding_dim, hidden_dim, feature_dim
        self.max_history, self.learning_rate, self.weight_decay = max_history, learning_rate, weight_decay
        self.use_history = use_history
        self.temperature, self.epochs, self.batch_size, self.seed = temperature, epochs, batch_size, seed
        self.device = torch.device(device)

    @staticmethod
    def _prepare_features(ids: Sequence[Hashable], features: pd.DataFrame | None, id_col: str, categorical_columns: Sequence[str], numeric_columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        frame = pd.DataFrame({id_col: ids})
        if features is not None:
            if id_col not in features:
                raise ValueError(f"Feature table is missing {id_col!r}")
            frame = frame.merge(features.drop_duplicates(id_col), on=id_col, how="left")
        categorical, cardinalities = [], []
        for column in categorical_columns:
            if column in frame:
                codes, uniques = pd.factorize(frame[column].fillna("__MISSING__").astype(str), sort=True)
                categorical.append(codes.astype(np.int64))
                cardinalities.append(len(uniques))
        numeric = []
        for column in numeric_columns:
            if column in frame:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
                finite = np.isfinite(values)
                values[~finite] = float(np.median(values[finite])) if finite.any() else 0.0
                values = np.sign(values) * np.log1p(np.abs(values))
                scale = float(values.std())
                numeric.append((values - float(values.mean())) / (scale if scale > 1e-6 else 1.0))
        cats = np.column_stack(categorical) if categorical else np.empty((len(ids), 0), np.int64)
        nums = np.column_stack(numeric).astype(np.float32) if numeric else np.empty((len(ids), 0), np.float32)
        return cats, nums, cardinalities

    def fit(self, interactions: pd.DataFrame, label_col: str = "is_click", user_col: str = "user_id", item_col: str = "video_id", time_col: str = "time_ms", user_features: pd.DataFrame | None = None, video_features: pd.DataFrame | None = None) -> "TwoTowerRecommender":
        positives = positive_pairs(interactions, label_col, user_col, item_col)
        if positives.empty:
            raise ValueError("Two-tower retrieval requires positive interactions")
        if time_col in interactions:
            positives = (
                interactions.loc[interactions[label_col] > 0]
                .sort_values(time_col, kind="stable")
                .drop_duplicates([user_col, item_col], keep="first")
            )
        self.user_ids, self.user_to_index = make_id_map(interactions[user_col])
        self.item_ids, self.item_to_index = make_id_map(interactions[item_col])
        users = positives[user_col].map(self.user_to_index).to_numpy(np.int64)
        items = positives[item_col].map(self.item_to_index).to_numpy(np.int64)

        histories = np.full((len(items), self.max_history), -1, np.int64)
        running: dict[Hashable, list[int]] = {}
        for row, (user, item) in enumerate(zip(positives[user_col], items)):
            prior = running.setdefault(user, [])
            tail = prior[-self.max_history:]
            if tail:
                histories[row, -len(tail):] = tail
            prior.append(int(item))
        self.user_history = {user: np.asarray(history[-self.max_history:], np.int64) for user, history in running.items()}
        self.user_seen = {user: set(history) for user, history in running.items()}
        self.popularity = np.bincount(items, minlength=len(self.item_ids)).astype(float)
        self.user_categorical, self.user_numeric, user_cards = self._prepare_features(self.user_ids, user_features, user_col, USER_CATEGORICAL, USER_NUMERIC)
        self.item_categorical, self.item_numeric, item_cards = self._prepare_features(self.item_ids, video_features, item_col, VIDEO_CATEGORICAL, VIDEO_NUMERIC)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.user_tower = _UserTower(len(self.user_ids), self.embedding_dim, self.hidden_dim, _FeatureEncoder(user_cards, self.user_numeric.shape[1], self.feature_dim), self.use_history).to(self.device)
        self.item_tower = _ItemTower(len(self.item_ids), self.embedding_dim, self.hidden_dim, _FeatureEncoder(item_cards, self.item_numeric.shape[1], self.feature_dim)).to(self.device)
        optimizer = torch.optim.AdamW([*self.user_tower.parameters(), *self.item_tower.parameters()], lr=self.learning_rate, weight_decay=self.weight_decay)
        order = np.arange(len(users))
        self.training_history = []
        for _ in range(self.epochs):
            rng.shuffle(order)
            total_loss = 0.0
            self.user_tower.train(); self.item_tower.train()
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                batch_users, batch_items = self._long(users[batch]), self._long(items[batch])
                history = self._long(histories[batch]); mask = history.ge(0)
                if not self.use_history:
                    mask = torch.zeros_like(mask)
                history_vectors = self._encode_items(history.clamp_min(0).reshape(-1)).reshape(len(batch), self.max_history, self.embedding_dim)
                user_vectors = self.user_tower(batch_users, self._long(self.user_categorical[users[batch]]), self._float(self.user_numeric[users[batch]]), history_vectors, mask.float())
                item_vectors = self._encode_items(batch_items)
                logits = user_vectors @ item_vectors.T / self.temperature
                duplicates = batch_items[:, None].eq(batch_items[None, :]); duplicates.fill_diagonal_(False)
                logits = logits.masked_fill(duplicates, -torch.inf)
                loss = F.cross_entropy(logits, torch.arange(len(batch), device=self.device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            self.training_history.append(total_loss / len(order))
        self._refresh_item_vectors()
        return self

    def _long(self, values: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def _float(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _encode_items(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        positions = flat.detach().cpu().numpy()
        return self.item_tower(flat, self._long(self.item_categorical[positions]), self._float(self.item_numeric[positions]))

    @torch.no_grad()
    def _refresh_item_vectors(self) -> None:
        self.item_tower.eval()
        self.item_vectors = self._encode_items(torch.arange(len(self.item_ids), device=self.device)).cpu().numpy()

    @torch.no_grad()
    def recommend(self, user_ids: Iterable[Hashable], k: int) -> dict[Hashable, list[Hashable]]:
        if not hasattr(self, "item_vectors"):
            raise RuntimeError("Call fit before recommend")
        output = {}; fallback = self.popularity / max(float(self.popularity.max()), 1.0)
        self.user_tower.eval(); self.item_tower.eval()
        for user in user_ids:
            if user in self.user_to_index:
                index = self.user_to_index[user]; values = self.user_history.get(user, np.empty(0, np.int64))
                padded = np.full(self.max_history, -1, np.int64)
                if len(values): padded[-len(values):] = values
                history = self._long(padded[None, :])
                vectors = self._encode_items(history.clamp_min(0).reshape(-1)).reshape(1, self.max_history, self.embedding_dim)
                history_mask = history.ge(0) if self.use_history else torch.zeros_like(history, dtype=torch.bool)
                user_vector = self.user_tower(self._long(np.asarray([index])), self._long(self.user_categorical[[index]]), self._float(self.user_numeric[[index]]), vectors, history_mask.float()).cpu().numpy()[0]
                scores = self.item_vectors @ user_vector + 1e-8 * fallback
            else:
                scores = fallback.copy()
            seen = self.user_seen.get(user, set())
            if seen: scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
            output[user] = [self.item_ids[i] for i in top_k_indices(scores, k)]
        return output
