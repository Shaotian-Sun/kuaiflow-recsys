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

