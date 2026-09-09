"""Single-task DeepFM/DIN training and Week 2 candidate reranking."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn

from kuaiflow.models.din import DIN
from kuaiflow.history import ClickHistory, prepare_histories, history_args, din_kwargs
from kuaiflow.data import Week1Splits, load_kuairand_features
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.models.deepfm import DeepFM


MISSING_TOKEN = "__MISSING__"
UNKNOWN_INDEX = 0
MISSING_INDEX = 1


class DeepFMFeatures:
    """Train-only categorical vocabularies and numerical transformations."""

    def fit(
        self,
        frame: pd.DataFrame,
        categorical: list[str],
        numeric: list[str],
    ) -> "DeepFMFeatures":
        missing = set(categorical + numeric).difference(frame.columns)
        if missing:
            raise ValueError(f"DeepFM data is missing features: {sorted(missing)}")
        if len(set(categorical + numeric)) != len(categorical + numeric):
            raise ValueError("Categorical and numeric feature names must be unique")

        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.vocabularies: dict[str, dict[str, int]] = {}
        for column in self.categorical:
            values = self._categorical_values(frame[column])
            observed = sorted(value for value in values.unique() if value != MISSING_TOKEN)
            self.vocabularies[column] = {
                value: index + 2 for index, value in enumerate(observed)
            }

        self.numeric_stats: dict[str, tuple[float, float, float]] = {}
        for column in self.numeric:
            raw = self._numeric_values(frame[column])
            finite = np.isfinite(raw)
            median = float(np.median(raw[finite])) if finite.any() else 0.0
            clean = np.where(finite, raw, median)
            transformed = np.log1p(np.maximum(clean, 0.0))
            mean = float(transformed.mean())
            scale = float(transformed.std())
            self.numeric_stats[column] = (
                median,
                mean,
                scale if scale > 1e-6 else 1.0,
            )
        return self

    @staticmethod
    def _categorical_values(series: pd.Series) -> pd.Series:
        values = series.astype("string").fillna(MISSING_TOKEN)
        return values.mask(values == "-124", MISSING_TOKEN).astype(str)

    @staticmethod
    def _numeric_values(series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=float, copy=True
        )
        values[values == -124] = np.nan
        return values

    @property
    def cardinalities(self) -> list[int]:
        # Index 0 is UNKNOWN and index 1 is MISSING for every field.
        return [len(self.vocabularies[column]) + 2 for column in self.categorical]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = set(self.categorical + self.numeric).difference(frame.columns)
        if missing:
            raise ValueError(f"DeepFM data is missing features: {sorted(missing)}")

        categorical_columns = []
        for column in self.categorical:
            values = self._categorical_values(frame[column])
            vocabulary = self.vocabularies[column]
            encoded = values.map(vocabulary).fillna(UNKNOWN_INDEX).to_numpy(
                dtype=np.int64, copy=True
            )
            encoded[values.to_numpy() == MISSING_TOKEN] = MISSING_INDEX
            categorical_columns.append(encoded)
        categorical = (
            np.column_stack(categorical_columns)
            if categorical_columns
            else np.empty((len(frame), 0), dtype=np.int64)
        )

        numeric_columns = []
        for column in self.numeric:
            median, mean, scale = self.numeric_stats[column]
            values = self._numeric_values(frame[column])
            values = np.where(np.isfinite(values), values, median)
            transformed = np.log1p(np.maximum(values, 0.0))
            numeric_columns.append(((transformed - mean) / scale).astype(np.float32))
        numeric = (
            np.column_stack(numeric_columns)
            if numeric_columns
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        return categorical, numeric

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown_index": UNKNOWN_INDEX,
            "missing_index": MISSING_INDEX,
            "categorical": self.categorical,
            "numeric": self.numeric,
            "vocabularies": self.vocabularies,
            "numeric_stats": {
                name: {"median": values[0], "log_mean": values[1], "log_scale": values[2]}
                for name, values in self.numeric_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeepFMFeatures":
        encoder = cls()
        encoder.categorical = list(payload["categorical"])
        encoder.numeric = list(payload["numeric"])
        encoder.vocabularies = {
            name: {value: int(index) for value, index in vocabulary.items()}
            for name, vocabulary in payload["vocabularies"].items()
        }
        encoder.numeric_stats = {
            name: (
                float(values["median"]),
                float(values["log_mean"]),
                float(values["log_scale"]),
            )
            for name, values in payload["numeric_stats"].items()
        }
        return encoder


@dataclass
class DeepFMRun:
    results: dict[str, Any]
    model: DeepFM | DIN
    encoder: DeepFMFeatures
    reranked_candidates: pd.DataFrame
    history_index: ClickHistory | None = None


def _attach_static_features(
    pairs: pd.DataFrame,
    user_features: pd.DataFrame | None,
    video_features: pd.DataFrame | None,
    feature_names: list[str],
) -> pd.DataFrame:
    """Join only requested static fields, preventing accidental label leakage."""
    base_columns = list(
        dict.fromkeys(
            ["user_id", "video_id"]
            + [column for column in feature_names if column in pairs.columns]
        )
    )
    frame = pairs[base_columns].copy()
    requested = set(feature_names)
    if user_features is not None:
        columns = [
            column
            for column in user_features.columns
            if column == "user_id" or (column in requested and column not in frame)
        ]
        frame = frame.merge(
            user_features[columns].drop_duplicates("user_id"),
            on="user_id",
            how="left",
            validate="many_to_one",
        )
    if video_features is not None:
        columns = [
            column
            for column in video_features.columns
            if column == "video_id" or (column in requested and column not in frame)
        ]
        frame = frame.merge(
            video_features[columns].drop_duplicates("video_id"),
            on="video_id",
            how="left",
            validate="many_to_one",
        )
    missing = requested.difference(frame.columns)
    if missing:
        raise ValueError(f"Static feature tables are missing: {sorted(missing)}")
    return frame


def _binary_labels(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target not in frame:
        raise ValueError(f"Ranking data is missing target column {target!r}")
    return (
        pd.to_numeric(frame[target], errors="coerce")
        .fillna(0)
        .gt(0)
        .to_numpy(np.float32)
    )


def _predict_logits(
    model: DeepFM | DIN,
    categorical: np.ndarray,
    numeric: np.ndarray,
    batch_size: int,
    device: torch.device,
    sequence: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(categorical), batch_size):
            rows.append(
                model(
                    torch.as_tensor(
                        categorical[start : start + batch_size],
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        numeric[start : start + batch_size],
                        dtype=torch.float32,
                        device=device,
                    ),
                    *history_args(sequence, slice(start, start + batch_size), device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.float32)


def _classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    has_both_classes = len(np.unique(labels)) == 2
    return {
        "examples": len(labels),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(labels, probabilities))
        if labels.sum()
        else None,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _validate_candidates(
    candidates: pd.DataFrame,
    candidate_k: int,
    expected_users_per_split: int | None = None,
) -> None:
    required = {"split", "user_id", "video_id", "retrieval_rank"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate data is missing columns: {sorted(missing)}")
    unknown_splits = set(candidates["split"]).difference({"validation", "test"})
    if unknown_splits:
        raise ValueError(f"Candidate data has unknown splits: {sorted(unknown_splits)}")
    if set(candidates["split"]) != {"validation", "test"}:
        raise ValueError("Candidate data must contain validation and test splits")
    if candidates.duplicated(["split", "user_id", "video_id"]).any():
        raise ValueError("Candidate data contains duplicate user-video pairs")
    sizes = candidates.groupby(["split", "user_id"], sort=False).size()
    if sizes.empty or not sizes.eq(candidate_k).all():
        raise ValueError(f"Every candidate user must have exactly {candidate_k} items")
    if expected_users_per_split is not None:
        user_counts = candidates.groupby("split")["user_id"].nunique()
        if not user_counts.eq(expected_users_per_split).all():
            raise ValueError(
                "Candidate user counts do not match expected_users_per_split"
            )
    expected_ranks = list(range(1, candidate_k + 1))
    ranks_valid = (
        candidates.sort_values("retrieval_rank")
        .groupby(["split", "user_id"], sort=False)["retrieval_rank"]
        .apply(list)
        .map(lambda values: values == expected_ranks)
        .all()
    )
    if not ranks_valid:
        raise ValueError(f"Every candidate list must contain ranks 1 through {candidate_k}")


def _recommendation_map(
    frame: pd.DataFrame, rank_column: str
) -> dict[Any, list[Any]]:
    ordered = frame.sort_values(["user_id", rank_column], kind="stable")
    return {
        user: group["video_id"].tolist()
        for user, group in ordered.groupby("user_id", sort=False)
    }


def _candidate_metrics(
    candidates: pd.DataFrame,
    splits: Week1Splits,
    k_values: list[int],
    model_name: str = "deepfm",
) -> dict[str, Any]:
    catalog = splits.train["video_id"].drop_duplicates().tolist()
    training_seen = {
        user: set(group["video_id"].tolist())
        for user, group in splits.train.loc[
            splits.train["is_click"] > 0, ["user_id", "video_id"]
        ].groupby("user_id", sort=False)
    }
    output: dict[str, Any] = {}
    for split_name, interactions in (
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        split_candidates = candidates.loc[candidates["split"] == split_name]
        ground_truth = build_ground_truth(
            interactions,
            label_col="is_click",
            catalog=catalog,
            exclude=training_seen,
        )
        retrieval = _recommendation_map(split_candidates, "retrieval_rank")
        deepfm = _recommendation_map(split_candidates, f"{model_name}_rank")
        output[split_name] = {
            "candidate_users": len(retrieval),
            "candidate_rows": len(split_candidates),
            "retrieval_order": {},
            f"{model_name}_order": {},
        }
        for k in k_values:
            baseline = evaluate_recommendations(retrieval, ground_truth, k, catalog)
            reranked = evaluate_recommendations(deepfm, ground_truth, k, catalog)
            output[split_name]["retrieval_order"][str(k)] = baseline
            output[split_name][f"{model_name}_order"][str(k)] = reranked
            output[split_name].setdefault("ndcg_delta", {})[str(k)] = (
                reranked[f"ndcg@{k}"] - baseline[f"ndcg@{k}"]
            )
    return output


def run_deepfm_ranking(
    splits: Week1Splits,
    config: dict[str, Any],
    candidates: pd.DataFrame | None = None,
    user_features: pd.DataFrame | None = None,
    video_features: pd.DataFrame | None = None,
) -> DeepFMRun:
    """Train DeepFM on logged impressions, then rerank stored candidates."""
    seed = int(config.get("seed", 2026))
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    model_name = model_config.get("architecture", "deepfm")
    if model_name not in ("deepfm", "din"):
        raise ValueError("Unsupported ranking architecture: " + str(model_name))
    use_din = model_name == "din"
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})
    target = str(data_config.get("target_column", "is_click"))
    if target != "is_click":
        raise ValueError("The standalone DeepFM stage currently supports is_click only")
    categorical = list(data_config.get("categorical_features", ["user_id", "video_id"]))
    numeric = list(data_config.get("numeric_features", []))
    feature_names = categorical + numeric

    forbidden = {
        "is_click", "is_like", "is_follow", "is_comment", "is_forward",
        "is_hate", "long_view", "play_time_ms", "profile_stay_time",
        "comment_stay_time", "is_profile_enter", "retrieval_rank", "is_rand",
        "retrieval_score", "split", "date", "hourmin", "time_ms", "tab",
        "duration_ms",
    }
    leaked = forbidden.intersection(feature_names)
    if leaked:
        raise ValueError(f"Post-exposure or policy fields cannot be features: {sorted(leaked)}")

    if user_features is None or video_features is None:
        raw_dir = data_config.get("raw_dir")
        needs_static = any(name not in {"user_id", "video_id"} for name in feature_names)
        if needs_static and not raw_dir:
            raise ValueError("data.raw_dir is required for configured static features")
        if raw_dir:
            loaded_users, loaded_videos = load_kuairand_features(raw_dir)
            user_features = loaded_users if user_features is None else user_features
            video_features = loaded_videos if video_features is None else video_features

    if candidates is None:
        candidate_path = data_config.get("candidates_path")
        if not candidate_path:
            raise ValueError("data.candidates_path is required")
        candidates = pd.read_csv(candidate_path)
    candidate_k = int(evaluation_config.get("candidate_k", 100))
    expected_users = evaluation_config.get("expected_users_per_split")
    _validate_candidates(
        candidates,
        candidate_k,
        int(expected_users) if expected_users is not None else None,
    )

    train_features = _attach_static_features(
        splits.train, user_features, video_features, feature_names
    )
    encoder = DeepFMFeatures().fit(train_features, categorical, numeric)
    encoded = {"train": encoder.transform(train_features)}
    del train_features
    for name, frame in (
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        split_features = _attach_static_features(
            frame, user_features, video_features, feature_names
        )
        encoded[name] = encoder.transform(split_features)
        del split_features
    labels = {
        "train": _binary_labels(splits.train, target),
        "validation": _binary_labels(splits.validation, target),
        "test": _binary_labels(splits.test, target),
    }

    history_index, histories = (prepare_histories(splits, encoder, model_config)
                                if use_din else (None, {}))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(model_config.get("device", "cpu"))
    model = (DIN if use_din else DeepFM)(
        **(din_kwargs(model_config, encoder.categorical) if use_din else {}),
        cardinalities=encoder.cardinalities,
        numeric_dim=len(numeric),
        embedding_dim=int(model_config.get("embedding_dim", 16)),
        hidden_dims=tuple(model_config.get("hidden_dims", [128, 64])),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 1e-6))
    batch_size = int(training_config.get("batch_size", 2048))
    epochs = int(training_config.get("epochs", 5))
    patience = int(training_config.get("early_stopping_patience", 2))
    if min(batch_size, epochs) <= 0 or patience < 0:
        raise ValueError("batch_size and epochs must be positive and patience non-negative")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    train_categorical, train_numeric = encoded["train"]
    order = np.arange(len(train_categorical))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = model(
                torch.as_tensor(
                    train_categorical[batch], dtype=torch.long, device=device
                ),
                torch.as_tensor(
                    train_numeric[batch], dtype=torch.float32, device=device
                ),
                *history_args(histories.get("train"), batch, device),
            )
            targets = torch.as_tensor(
                labels["train"][batch], dtype=torch.float32, device=device
            )
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)

        validation_logits = _predict_logits(
            model, *encoded["validation"], batch_size, device, histories.get("validation")
        )
        validation_metrics = _classification_metrics(
            labels["validation"], validation_logits
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / len(order),
                "validation_log_loss": validation_metrics["log_loss"],
                "validation_roc_auc": validation_metrics["roc_auc"],
            }
        )
        if validation_metrics["log_loss"] < best_loss:
            best_loss = validation_metrics["log_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    pointwise = {}
    for split_name in ("validation", "test"):
        logits = _predict_logits(model, *encoded[split_name], batch_size, device, histories.get(split_name))
        pointwise[split_name] = _classification_metrics(labels[split_name], logits)

    # Candidate scoring is chunked so the million-row handoff never needs one
    # additional full-size encoded matrix in memory.
    encoded.pop("train")
    histories.pop("train", None)
    candidate_logits = np.empty(len(candidates), dtype=np.float32)
    candidate_chunk_size = int(evaluation_config.get("candidate_chunk_size", 100_000))
    if candidate_chunk_size <= 0:
        raise ValueError("evaluation.candidate_chunk_size must be positive")
    for start in range(0, len(candidates), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(candidates))
        chunk = _attach_static_features(
            candidates.iloc[start:stop], user_features, video_features, feature_names
        )
        chunk_encoded = encoder.transform(chunk)
        candidate_logits[start:stop] = _predict_logits(
            model, *chunk_encoded, batch_size, device,
            history_index.transform(candidates.iloc[start:stop]) if history_index else None
        )
    reranked = candidates.copy()
    reranked[f"{model_name}_score"] = 1.0 / (
        1.0 + np.exp(-np.clip(candidate_logits, -40.0, 40.0))
    )
    reranked = reranked.sort_values(
        ["split", "user_id", f"{model_name}_score", "retrieval_rank"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    reranked[f"{model_name}_rank"] = (
        reranked.groupby(["split", "user_id"], sort=False).cumcount() + 1
    )
    reranked = reranked.sort_values(
        ["split", "user_id", f"{model_name}_rank"], kind="stable"
    ).reset_index(drop=True)

    k_values = sorted({int(k) for k in evaluation_config.get("k_values", [10, 20, 50, 100])})
    if not k_values or min(k_values) <= 0 or max(k_values) > candidate_k:
        raise ValueError("evaluation.k_values must be between 1 and candidate_k")
    results = {
        "week": 3,
        "model": model_name,
        "objective": target,
        "seed": seed,
        "training_examples": len(splits.train),
        "candidate_source": str(data_config.get("candidates_path", "in-memory")),
        "candidate_k": candidate_k,
        "features": {
            "categorical": categorical,
            "numeric": numeric,
            "retrieval_rank_used_as_feature": False,
        },
        "sequence": history_index.to_dict() if history_index else None,
        "architecture": {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "embedding_dim": model.embedding_dim,
            "hidden_dims": list(model.hidden_dims),
            "logit": "MLP(fields + DIN_interest)" if use_din else "linear + fm_second_order + deep",
        },
        "optimization": {
            "loss": "unweighted_binary_cross_entropy_with_logits",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "training_seconds": training_seconds,
            "history": history,
        },
        "pointwise": pointwise,
        "candidate_ranking": _candidate_metrics(reranked, splits, k_values, model_name),
    }
    return DeepFMRun(results, model, encoder, reranked, history_index)


def save_deepfm_run(
    run: DeepFMRun,
    config: dict[str, Any],
    artifacts_dir: str | Path,
    reranked_candidates_path: str | Path,
) -> None:
    """Persist metrics/model metadata and the reranked candidate dataset."""
    model_name = run.results["model"]
    artifact_name = model_name
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    reranked_output = Path(reranked_candidates_path)
    reranked_output.parent.mkdir(parents=True, exist_ok=True)
    run.reranked_candidates.to_csv(
        reranked_output, index=False, compression="gzip"
    )

    with (output / f"week3_{artifact_name}_encoder.json").open("w", encoding="utf-8") as handle:
        json.dump(run.encoder.to_dict(), handle, indent=2)
    if run.history_index is not None:
        run.history_index.save(output / f"week3_{artifact_name}_history.npz")
    torch.save(
        {
            "state_dict": run.model.state_dict(),
            "history_file": f"week3_{artifact_name}_history.npz" if run.history_index else None,
            "model_config": config.get("model", {}),
            "categorical_features": run.encoder.categorical,
            "numeric_features": run.encoder.numeric,
            "cardinalities": run.encoder.cardinalities,
        },
        output / f"week3_{artifact_name}_model.pt",
    )

    results = copy.deepcopy(run.results)
    results["outputs"] = {
        "reranked_candidates": str(reranked_output),
        "model": str(output / f"week3_{artifact_name}_model.pt"),
        "encoder": str(output / f"week3_{artifact_name}_encoder.json"),
    }
    if run.history_index is not None:
        results["outputs"]["history"] = str(output / f"week3_{artifact_name}_history.npz")
    with (output / f"week3_{artifact_name}_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows: list[dict[str, Any]] = []
    for split, split_result in results["candidate_ranking"].items():
        for ordering in ("retrieval_order", f"{model_name}_order"):
            for k, metrics in split_result[ordering].items():
                rows.append(
                    {
                        "split": split,
                        "ordering": ordering,
                        "k": int(k),
                        "recall": metrics[f"recall@{k}"],
                        "hit_rate": metrics[f"hit_rate@{k}"],
                        "ndcg": metrics[f"ndcg@{k}"],
                        "coverage": metrics[f"coverage@{k}"],
                        "evaluated_users": metrics["evaluated_users"],
                    }
                )
    keys = list(rows[0]) if rows else []
    with (output / f"week3_{artifact_name}_ranking_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_deepfm_artifacts(
    model_path: str | Path,
    encoder_path: str | Path,
    device: str = "cpu",
) -> tuple[DeepFM | DIN, DeepFMFeatures]:
    """Restore DeepFM or DIN; DIN models also expose their saved history_index."""
    with Path(encoder_path).open(encoding="utf-8") as handle:
        encoder = DeepFMFeatures.from_dict(json.load(handle))
    checkpoint = torch.load(
        model_path, map_location=torch.device(device), weights_only=True
    )
    model_config = checkpoint["model_config"]
    use_din = model_config.get("architecture", "deepfm") == "din"
    model = (DIN if use_din else DeepFM)(
        **(din_kwargs(model_config, encoder.categorical) if use_din else {}),
        cardinalities=checkpoint["cardinalities"],
        numeric_dim=len(checkpoint["numeric_features"]),
        embedding_dim=int(model_config.get("embedding_dim", 16)),
        hidden_dims=tuple(model_config.get("hidden_dims", [128, 64])),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    if use_din:
        model.history_index = ClickHistory.load(Path(model_path).parent / checkpoint["history_file"])
    model.eval()
    return model, encoder
