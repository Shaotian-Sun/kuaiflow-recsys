import tempfile
from pathlib import Path
import unittest

import pandas as pd

from kuaiflow.data import (
    FUTURE_FILE,
    HISTORY_FILE,
    RANDOM_FILE,
    prepare_week1_data,
    split_future_by_time,
)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        columns = ["user_id", "video_id", "time_ms", "is_click"]
        self.history = pd.DataFrame([(0, 10, 1, 1), (1, 11, 2, 0)], columns=columns)
        self.future = pd.DataFrame(
            [(0, 11, 10, 1), (1, 10, 11, 1), (0, 12, 12, 0), (1, 12, 13, 1)],
            columns=columns,
        )
        self.random = pd.DataFrame([(0, 12, 20, 1)], columns=columns)

    def test_split_is_strictly_chronological(self) -> None:
        splits = split_future_by_time(self.history, self.future, self.random, 0.5)
        self.assertLess(splits.train.time_ms.max(), splits.validation.time_ms.min())
        self.assertLess(splits.validation.time_ms.max(), splits.test.time_ms.min())

    def test_prepare_writes_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            processed = root / "processed"
            raw.mkdir()
            self.history.to_csv(raw / HISTORY_FILE, index=False)
            self.future.to_csv(raw / FUTURE_FILE, index=False)
            self.random.to_csv(raw / RANDOM_FILE, index=False)
            summary = prepare_week1_data(raw, processed)
            self.assertEqual(summary["splits"]["train"]["interactions"], 2)
            for name in ("train", "validation", "test", "random_audit"):
                self.assertTrue((processed / f"{name}.csv.gz").exists())

    def test_split_removes_overlapping_future_rows(self) -> None:
        columns = ["user_id", "video_id", "time_ms", "is_click"]

        history = pd.DataFrame(
            [(0, 10, 10, 1), (1, 11, 20, 1)],
            columns=columns,
        )
        future = pd.DataFrame(
            [
                (0, 12, 19, 1),
                (0, 13, 20, 1),
                (0, 14, 21, 1),
                (0, 15, 22, 1),
                (0, 16, 23, 1),
                (0, 17, 24, 1),
            ],
            columns=columns,
        )

        splits = split_future_by_time(
            history,
            future,
            self.random,
            validation_fraction=0.5,
        )
        retained_future = pd.concat(
            [splits.validation, splits.test],
            ignore_index=True,
        )

        self.assertEqual(
            retained_future["time_ms"].tolist(),
            [21, 22, 23, 24],
        )
        self.assertLess(
            splits.train["time_ms"].max(),
            splits.validation["time_ms"].min(),
        )

if __name__ == "__main__":
    unittest.main()

