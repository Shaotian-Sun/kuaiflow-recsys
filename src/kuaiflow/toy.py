"""A tiny deterministic dataset for smoke-testing the complete pipeline."""

from __future__ import annotations

import pandas as pd

from kuaiflow.data import Week1Splits


def make_toy_splits() -> Week1Splits:
    train_rows = [
        (0, 10, 1, 1), (0, 11, 2, 1), (0, 12, 3, 0),
        (1, 10, 4, 1), (1, 11, 5, 1), (1, 13, 6, 0),
        (2, 12, 7, 1), (2, 13, 8, 1), (2, 10, 9, 0),
        (3, 12, 10, 1), (3, 13, 11, 1), (3, 11, 12, 0),
    ]
    validation_rows = [(0, 13, 20, 1), (1, 12, 21, 1), (2, 11, 22, 1), (3, 10, 23, 1)]
    test_rows = [(0, 13, 30, 1), (1, 12, 31, 1), (2, 11, 32, 1), (3, 10, 33, 1)]
    columns = ["user_id", "video_id", "time_ms", "is_click"]
    train = pd.DataFrame(train_rows, columns=columns)
    validation = pd.DataFrame(validation_rows, columns=columns)
    test = pd.DataFrame(test_rows, columns=columns)
    return Week1Splits(train, validation, test, test.copy())


def make_multitask_toy_splits() -> Week1Splits:
    """Add every Week 3 outcome to the tiny deterministic data set.

    This is deliberately separate from :func:`make_toy_splits`: the earlier
    retrieval and DeepFM examples should keep their small four-column contract.
    The patterns below give every binary task both classes and exercise zero
    watch time, missing duration, and replayed viewing.
    """

    base = make_toy_splits()

    def add_targets(frame: pd.DataFrame, offset: int) -> pd.DataFrame:
        output = frame.copy()
        index = pd.Series(range(offset, offset + len(output)), index=output.index)
        output["is_click"] = (index % 2 == 0).astype(int)
        output["is_like"] = (index % 3 == 0).astype(int)
        output["is_follow"] = (index % 4 == 0).astype(int)
        output["is_comment"] = (index % 3 == 1).astype(int)
        output["is_forward"] = (index % 4 == 1).astype(int)
        output["is_hate"] = (index % 5 == 0).astype(int)
        output["long_view"] = (index % 2 == 1).astype(int)
        output["is_profile_enter"] = (index % 3 == 2).astype(int)

        watch_pattern = [0, 2_000, 8_000, 25_000, 45_000, 80_000]
        duration_pattern = [10_000, 10_000, 20_000, 20_000, 40_000, 0]
        output["play_time_ms"] = [
            watch_pattern[value % len(watch_pattern)] for value in index
        ]
        output["duration_ms"] = [
            duration_pattern[value % len(duration_pattern)] for value in index
        ]
        return output

    train = add_targets(base.train, 0)
    validation = add_targets(base.validation, len(train))
    test = add_targets(base.test, len(train) + len(validation))
    return Week1Splits(train, validation, test, test.copy())
