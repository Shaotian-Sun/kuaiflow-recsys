"""Strictly prior, train-only click histories shared by both DIN variants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class ClickHistory:
    def __init__(self, max_length=30):
        if isinstance(max_length, bool) or int(max_length) != max_length or max_length <= 0:
            raise ValueError("history_max_length must be a positive integer")
        self.max_length = int(max_length)
        self.events = {}

    def fit(self, train, encoder):
        if "video_id" not in encoder.categorical:
            raise ValueError("DIN requires video_id in categorical_features")
        timestamps = self._timestamps(train)
        if not len(timestamps):
            raise ValueError("DIN history requires non-empty training data")
        if train[["user_id", "video_id"]].isna().any().any():
            raise ValueError("History user and video IDs cannot be missing")
        positive = train.loc[train.is_click > 0, ["user_id", "video_id", "time_ms"]].copy()
        positive["time_ms"] = timestamps[(train.is_click > 0).to_numpy()]
        positive["user_id"] = positive.user_id.astype(str)
        positive["item"] = encoder._categorical_values(positive.video_id).map(
            encoder.vocabularies["video_id"]).fillna(0).astype(np.int32)
        positive = positive.loc[positive.item > 1].sort_values(
            ["user_id", "time_ms", "video_id"], kind="stable")
        self.events = {user: (group.time_ms.to_numpy(np.int64), group.item.to_numpy(np.int32))
                       for user, group in positive.groupby("user_id", sort=False)}
        self.cutoff = int(timestamps.max())
        return self

    @staticmethod
    def _timestamps(frame):
        if "time_ms" not in frame:
            raise ValueError("DIN requires time_ms for causal history construction")
        values = pd.to_numeric(frame.time_ms, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("History timestamps must be finite integer milliseconds")
        return values.astype(np.int64)

    def transform(self, frame, *, causal=False):
        times = self._timestamps(frame) if causal else None
        output = np.zeros((len(frame), self.max_length), dtype=np.int32)
        offsets = np.arange(self.max_length, 0, -1)
        # Positional group indices remain correct for sliced/nonconsecutive indices.
        for user, positions in frame.user_id.astype(str).groupby(
                frame.user_id.astype(str), sort=False).indices.items():
            if user not in self.events:
                continue
            event_times, items = self.events[user]
            ends = (np.searchsorted(event_times, times[positions], side="left") if causal
                    else np.full(len(positions), len(items)))
            indices = ends[:, None] - offsets
            output[positions] = np.where(indices >= 0, items[np.maximum(indices, 0)], 0)
        return output

    def to_dict(self):
        return {"max_length": self.max_length, "cutoff_time_ms": self.cutoff,
                "signal": "is_click > 0", "training": "strictly earlier timestamp; equal-time events excluded",
                "evaluation": "train-only frozen history for validation, test and candidates",
                "history_users": len(self.events)}

    def save(self, path):
        users, times, items = [], [], []
        for user, (event_times, event_items) in self.events.items():
            users.extend([user] * len(event_times))
            times.extend(event_times)
            items.extend(event_items)
        np.savez_compressed(path, users=np.asarray(users, dtype=str),
                            times=np.asarray(times, dtype=np.int64), items=np.asarray(items, dtype=np.int32),
                            max_length=self.max_length, cutoff=self.cutoff)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            history = cls(int(data['max_length']))
            history.cutoff = int(data['cutoff'])
            frame = pd.DataFrame({"user": data['users'], "time": data['times'], "item": data['items']})
        history.events = {user: (group.time.to_numpy(), group.item.to_numpy())
                          for user, group in frame.groupby('user', sort=False)}
        return history


def prepare_histories(splits, encoder, model_config):
    history = ClickHistory(model_config.get("history_max_length", 30)).fit(splits.train, encoder)
    for frame in (splits.validation, splits.test):
        if np.any(history._timestamps(frame) <= history.cutoff):
            raise ValueError("DIN evaluation timestamps must be strictly after training")
    arrays = {"train": history.transform(splits.train, causal=True),
              "validation": history.transform(splits.validation),
              "test": history.transform(splits.test)}
    return history, arrays


def history_args(history, selection, device):
    return () if history is None else (torch.as_tensor(history[selection], dtype=torch.long, device=device),)


def din_kwargs(model_config, categorical):
    if "video_id" not in categorical:
        raise ValueError("DIN requires video_id in categorical_features")
    return {"video_field_index": categorical.index("video_id"),
            "attention_hidden_dims": tuple(model_config.get("attention_hidden_dims", [64, 32]))}
