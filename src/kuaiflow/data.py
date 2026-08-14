"""Loading and leakage-aware splitting for KuaiRand-Pure logs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_FILE = "log_standard_4_08_to_4_21_pure.csv"
FUTURE_FILE = "log_standard_4_22_to_5_08_pure.csv"
RANDOM_FILE = "log_random_4_22_to_5_08_pure.csv"
REQUIRED_COLUMNS = {"user_id", "video_id", "time_ms", "is_click"}


@dataclass(frozen=True)
class Week1Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    random_audit: pd.DataFrame


def _find_unique_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {root}. Run `kuaiflow download` first."
        )
    if len(matches) > 1:
        raise RuntimeError(f"Found more than one {filename!r} under {root}: {matches}")
    return matches[0]


def read_log(path: str | Path) -> pd.DataFrame:
    """Read one KuaiRand log and validate columns used in week one."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no interactions")
    frame = frame.sort_values("time_ms", kind="stable").reset_index(drop=True)
    return frame


def load_kuairand_pure(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load historical standard, future standard, and randomized exposure logs."""
    root = Path(raw_dir)
    history = read_log(_find_unique_file(root, HISTORY_FILE))
    future = read_log(_find_unique_file(root, FUTURE_FILE))
    random_audit = read_log(_find_unique_file(root, RANDOM_FILE))
    return history, future, random_audit


def split_future_by_time(
    history: pd.DataFrame,
    future: pd.DataFrame,
    random_audit: pd.DataFrame,
    validation_fraction: float = 0.5,
) -> Week1Splits:
    """Use the earlier logging period for training and split the later period by time.

    This global chronological split is deliberately stricter than a random row split:
    no future interaction is allowed into the training set.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    history_end = history["time_ms"].max()
    future = future.loc[future["time_ms"] > history_end].copy()
    if history["time_ms"].max() >= future["time_ms"].min():
        raise ValueError("Historical and future logs overlap; cannot make a strict time split")

    ordered = future.sort_values("time_ms", kind="stable").reset_index(drop=True)
    cut = int(np.floor(len(ordered) * validation_fraction))
    cut = min(max(cut, 1), len(ordered) - 1)
    validation = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()

    return Week1Splits(
        train=history.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        random_audit=random_audit.reset_index(drop=True),
    )


def prepare_week1_data(
    raw_dir: str | Path,
    processed_dir: str | Path,
    validation_fraction: float = 0.5,
) -> dict[str, object]:
    """Create compressed CSV splits and a small data summary."""
    history, future, random_audit = load_kuairand_pure(raw_dir)
    splits = split_future_by_time(history, future, random_audit, validation_fraction)
    output = Path(processed_dir)
    output.mkdir(parents=True, exist_ok=True)

    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
        "random_audit": splits.random_audit,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv.gz", index=False, compression="gzip")

    summary: dict[str, object] = {
        "splits": {
            name: {
                "interactions": int(len(frame)),
                "users": int(frame["user_id"].nunique()),
                "items": int(frame["video_id"].nunique()),
                "start_time_ms": int(frame["time_ms"].min()),
                "end_time_ms": int(frame["time_ms"].max()),
                "click_rate": float(frame["is_click"].mean()),
            }
            for name, frame in frames.items()
        }
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_prepared(processed_dir: str | Path) -> Week1Splits:
    root = Path(processed_dir)

    def load(name: str) -> pd.DataFrame:
        path = root / f"{name}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run `kuaiflow prepare` first")
        return pd.read_csv(path)

    return Week1Splits(
        train=load("train"),
        validation=load("validation"),
        test=load("test"),
        random_audit=load("random_audit"),
    )

