"""Utilities used by baseline recommenders."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd


def positive_pairs(
    interactions: pd.DataFrame,
    label_col: str,
    user_col: str,
    item_col: str,
) -> pd.DataFrame:
    return (
        interactions.loc[interactions[label_col] > 0, [user_col, item_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def make_id_map(values: pd.Series) -> tuple[list[Hashable], dict[Hashable, int]]:
    ids = values.drop_duplicates().tolist()
    return ids, {value: index for index, value in enumerate(ids)}


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return np.array([], dtype=np.int64)
    actual_k = min(k, finite.size)
    finite_scores = scores[finite]
    if actual_k == finite.size:
        order = np.argsort(-finite_scores, kind="stable")
    else:
        selected = np.argpartition(finite_scores, -actual_k)[-actual_k:]
        order = selected[np.argsort(-finite_scores[selected], kind="stable")]
    return finite[order]

