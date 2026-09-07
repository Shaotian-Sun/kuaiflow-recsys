"""Training and evaluation pipeline for Week 2 retrieval models with FAISS support."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from kuaiflow.data import Week1Splits, load_kuairand_features
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.models import TwoTowerRecommender
from kuaiflow.models.common import top_k_indices

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. FAISS features will be disabled.")


class FAISSRetriever:
    """FAISS-powered candidate retrieval for two-tower models."""

    def __init__(
        self,
        index_type: str = "flat",
        n_lists: int = 100,
        n_probe: int = 10,
        hnsw_m: int = 32,
        use_gpu: bool = False,
        seed: int = 2026,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Please install with: pip install faiss-cpu")

        self.index_type = index_type
        self.n_lists = n_lists
        self.n_probe = n_probe
        self.hnsw_m = hnsw_m
        self.use_gpu = use_gpu
        self.seed = seed

        self.index = None
        self.video_ids = None
        self.dimension = None

    def build_index(self, embeddings: np.ndarray, ids: list[Any]) -> dict[str, Any]:
        """Build FAISS index from embeddings."""
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got {embeddings.ndim}D")

        self.dimension = embeddings.shape[1]
        embeddings = embeddings.astype('float32').copy()

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index based on type
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.n_lists, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = self.n_probe
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index.add(embeddings)
        self.video_ids = np.array(ids)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )

        return self.get_index_stats()

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int,
        batch_size: int = 1024,
    ) -> tuple[list[list[Any]], list[np.ndarray], float]:
        """Search for top-k similar items."""
        if self.index is None:
            raise RuntimeError("Build index before searching")

        query_embeddings = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings)

        all_indices = []
        all_scores = []
        search_time = 0.0

        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i:i + batch_size]

            start_time = time.perf_counter()
            scores, indices = self.index.search(batch, k)
            search_time += time.perf_counter() - start_time

            all_indices.append(indices)
            all_scores.append(scores)

        if all_indices:
            indices = np.vstack(all_indices)
            scores = np.vstack(all_scores)
        else:
            indices = np.array([])
            scores = np.array([])

        recommendations = []
        score_list = []
        for i, row in enumerate(indices):
            valid_mask = row >= 0
            valid_indices = row[valid_mask]
            valid_scores = scores[i][valid_mask] if len(scores) > 0 else np.array([])

            item_ids = self.video_ids[valid_indices].tolist()
            recommendations.append(item_ids)
            score_list.append(valid_scores)

        return recommendations, score_list, search_time

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the index."""
        if self.index is None:
            return {"index_built": False}

        return {
            "index_built": True,
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "memory_mb": self.index.ntotal * self.index.d * 4 / (1024 * 1024),
            "parameters": {
                "n_lists": self.n_lists if self.index_type == "ivf" else None,
                "n_probe": self.n_probe if self.index_type == "ivf" else None,
                "hnsw_m": self.hnsw_m if self.index_type == "hnsw" else None,
            }
        }


class ExactRetriever:
    """Batched NumPy inner-product retrieval with the FAISS search interface."""

    def __init__(self, embeddings: np.ndarray, ids: list[Any]):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.ids = np.asarray(ids)

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int,
        batch_size: int = 1024,
    ) -> tuple[list[list[Any]], list[np.ndarray], float]:
        """Return exact top-k candidates for each query."""
        recommendations: list[list[Any]] = []
        score_rows: list[np.ndarray] = []
        search_seconds = 0.0

        for start in range(0, len(query_embeddings), batch_size):
            batch = np.asarray(
                query_embeddings[start : start + batch_size], dtype=np.float32
            )
            search_started = time.perf_counter()
            scores = batch @ self.embeddings.T
            indices = np.vstack([top_k_indices(row, k) for row in scores])
            search_seconds += time.perf_counter() - search_started
            for row_number, row_indices in enumerate(indices):
                recommendations.append(self.ids[row_indices].tolist())
                score_rows.append(scores[row_number, row_indices])

        return recommendations, score_rows, search_seconds


def _faiss_search_k(
    model: TwoTowerRecommender, user_ids: list[Any], requested_k: int
) -> int:
    """Over-fetch enough candidates to retain K after seen-item filtering."""
    max_seen = max((len(model.user_seen.get(user, ())) for user in user_ids), default=0)
    return min(len(model.item_ids), requested_k + max_seen)


def _postprocess_faiss_recommendations(
    model: TwoTowerRecommender,
    user_ids: list[Any],
    raw_recommendations: list[list[Any]],
    k: int,
) -> dict[Any, list[Any]]:
    """Apply the same cold-start and seen-item policy as exact retrieval."""
    popular_indices = np.argsort(-model.popularity)
    output: dict[Any, list[Any]] = {}

    for user, raw_recs in zip(user_ids, raw_recommendations):
        if user not in model.user_to_index:
            output[user] = [model.item_ids[index] for index in popular_indices[:k]]
            continue

        # user_seen contains internal item indices; FAISS returns external video IDs.
        seen_ids = {model.item_ids[index] for index in model.user_seen.get(user, ())}
        filtered = [item for item in raw_recs if item not in seen_ids]

        if len(filtered) < k:
            for index in popular_indices:
                item = model.item_ids[index]
                if item not in seen_ids and item not in filtered:
                    filtered.append(item)
                if len(filtered) >= k:
                    break

        output[user] = filtered[:k]

    return output


def _latency_summary(samples: list[float], user_count: int) -> dict[str, Any]:
    """Summarize repeated wall-clock measurements as per-user latency."""
    milliseconds = np.asarray(samples, dtype=float) * 1000 / max(user_count, 1)
    return {
        "median_ms_per_user": float(np.median(milliseconds)),
        "p95_ms_per_user": float(np.percentile(milliseconds, 95)),
        "samples_ms_per_user": milliseconds.tolist(),
    }


def _approximation_fidelity(
    exact_candidates: list[list[Any]],
    approximate_candidates: list[list[Any]],
    exact_final: dict[Any, list[Any]],
    approximate_final: dict[Any, list[Any]],
    user_ids: list[Any],
    k_values: list[int],
    candidate_eligible_users: set[Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare ANN candidates and final lists directly against exact retrieval."""
    if len(exact_candidates) != len(approximate_candidates) or len(user_ids) != len(
        exact_candidates
    ):
        raise ValueError("Exact, approximate, and user result counts must match")

    fidelity: dict[str, dict[str, float]] = {}
    for k in k_values:
        candidate_recalls: list[float] = []
        candidate_ndcgs: list[float] = []
        final_overlaps: list[float] = []
        final_exact_matches: list[float] = []

        for position, user in enumerate(user_ids):
            exact_row = exact_candidates[position][:k]
            approximate_row = approximate_candidates[position][:k]
            if candidate_eligible_users is None or user in candidate_eligible_users:
                exact_set = set(exact_row)
                candidate_recalls.append(
                    len(exact_set.intersection(approximate_row))
                    / max(len(exact_row), 1)
                )

                # Exact ranks define graded relevance: the nearest exact item has gain K,
                # the Kth has gain 1. This penalizes missing and reordered neighbors.
                gains = {
                    item: len(exact_row) - rank for rank, item in enumerate(exact_row)
                }
                discounts = np.log2(np.arange(2, len(exact_row) + 2, dtype=float))
                ideal_dcg = sum(
                    gains[item] / discounts[rank]
                    for rank, item in enumerate(exact_row)
                )
                approximate_dcg = sum(
                    gains.get(item, 0.0) / np.log2(rank + 2)
                    for rank, item in enumerate(approximate_row)
                )
                candidate_ndcgs.append(
                    approximate_dcg / ideal_dcg if ideal_dcg else 1.0
                )

            exact_list = exact_final[user][:k]
            approximate_list = approximate_final[user][:k]
            final_overlaps.append(
                len(set(exact_list).intersection(approximate_list))
                / max(len(exact_list), 1)
            )
            final_exact_matches.append(float(approximate_list == exact_list))

        fidelity[str(k)] = {
            f"candidate_recall@{k}": float(np.mean(candidate_recalls)),
            f"candidate_ndcg@{k}": float(np.mean(candidate_ndcgs)),
            f"final_overlap@{k}": float(np.mean(final_overlaps)),
            f"final_exact_match_rate@{k}": float(np.mean(final_exact_matches)),
            "candidate_evaluated_users": float(len(candidate_recalls)),
            "final_evaluated_users": float(len(user_ids)),
        }
    return fidelity


def _benchmark_retriever(
    model: TwoTowerRecommender,
    user_ids: list[Any],
    retriever: Any,
    k: int,
    precomputed_embeddings: np.ndarray,
    warmup_runs: int,
    measured_runs: int,
    embedding_batch_size: int = 256,
    search_batch_size: int = 1024,
) -> tuple[dict[Any, list[Any]], dict[str, Any]]:
    """Measure identical search, retrieval-pipeline, and end-to-end boundaries."""
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be >= 0 and measured_runs must be > 0")

    search_k = _faiss_search_k(model, user_ids, k)

    def retrieve(query_embeddings: np.ndarray) -> tuple[dict[Any, list[Any]], float, float]:
        started = time.perf_counter()
        raw_recommendations, _, _ = retriever.search(
            query_embeddings, k=search_k, batch_size=search_batch_size
        )
        search_seconds = time.perf_counter() - started

        started = time.perf_counter()
        recommendations = _postprocess_faiss_recommendations(
            model, user_ids, raw_recommendations, k
        )
        postprocess_seconds = time.perf_counter() - started
        return recommendations, search_seconds, postprocess_seconds

    for _ in range(warmup_runs):
        retrieve(precomputed_embeddings)

    search_samples: list[float] = []
    postprocess_samples: list[float] = []
    pipeline_samples: list[float] = []
    recommendations: dict[Any, list[Any]] = {}
    for _ in range(measured_runs):
        recommendations, search_seconds, postprocess_seconds = retrieve(
            precomputed_embeddings
        )
        search_samples.append(search_seconds)
        postprocess_samples.append(postprocess_seconds)
        pipeline_samples.append(search_seconds + postprocess_seconds)

    # End-to-end starts before user encoding and stops after shared post-processing.
    for _ in range(warmup_runs):
        embeddings = _get_user_embeddings_batch(
            model, user_ids, batch_size=embedding_batch_size
        )
        retrieve(embeddings)

    end_to_end_samples: list[float] = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        embeddings = _get_user_embeddings_batch(
            model, user_ids, batch_size=embedding_batch_size
        )
        recommendations, _, _ = retrieve(embeddings)
        end_to_end_samples.append(time.perf_counter() - started)

    latency = {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "users": len(user_ids),
        "requested_k": k,
        "candidate_k": search_k,
        "search_only": _latency_summary(search_samples, len(user_ids)),
        "postprocess_only": _latency_summary(postprocess_samples, len(user_ids)),
        "retrieval_pipeline": _latency_summary(pipeline_samples, len(user_ids)),
        "end_to_end": _latency_summary(end_to_end_samples, len(user_ids)),
    }
    return recommendations, latency


def _evaluation_users(
    ground_truth: dict[Any, set[Any]], max_users: int | None, seed: int
) -> list[Any]:
    """Sample users for evaluation."""
    users = list(ground_truth)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(users), size=max_users, replace=False))
        users = [users[position] for position in positions]
    return users


def _get_user_embeddings_batch(
    model: TwoTowerRecommender,
    user_ids: list[Any],
    batch_size: int = 256,
) -> np.ndarray:
    """Generate user embeddings in batch."""
    if not hasattr(model, "user_tower"):
        raise RuntimeError("Call fit before getting user embeddings")

    model.user_tower.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]

            for user in batch_users:
                if user in model.user_to_index:
                    index = model.user_to_index[user]
                    values = model.user_history.get(user, np.empty(0, np.int64))
                    padded = np.full(model.max_history, -1, np.int64)
                    if len(values):
                        padded[-len(values):] = values

                    history = model._long(padded[None, :])
                    vectors = model._encode_items(
                        history.clamp_min(0).reshape(-1)
                    ).reshape(1, model.max_history, model.embedding_dim)

                    history_mask = history.ge(0) if model.use_history else torch.zeros_like(
                        history, dtype=torch.bool
                    )

                    user_vector = model.user_tower(
                        model._long(np.asarray([index])),
                        model._long(model.user_categorical[[index]]),
                        model._float(model.user_numeric[[index]]),
                        vectors,
                        history_mask.float(),
                    ).cpu().numpy()[0]
                    all_embeddings.append(user_vector)
                else:
                    # Cold start: use zero vector
                    all_embeddings.append(np.zeros(model.embedding_dim, dtype=np.float32))

    return np.array(all_embeddings, dtype=np.float32)


def _candidate_rows(
    split: str,
    user_ids: list[Any],
    recommendations: dict[Any, list[Any]],
) -> list[dict[str, Any]]:
    """Flatten final retrieval lists for persistence and downstream ranking."""
    rows: list[dict[str, Any]] = []
    for user in user_ids:
        items = recommendations[user]
        rows.extend(
            {
                "split": split,
                "user_id": user,
                "video_id": item,
                "retrieval_rank": rank,
            }
            for rank, item in enumerate(items, start=1)
        )
    return rows


def run_week2_retrieval(
    splits: Week1Splits, config: dict[str, Any]
) -> dict[str, Any]:
    """Run Week 2 retrieval evaluation (original + optional FAISS)."""
    seed = int(config.get("seed", 2026))
    label_col = config["data"].get("positive_column", "is_click")
    k_values = sorted({int(k) for k in config["evaluation"]["k_values"]})
    if not k_values or min(k_values) <= 0:
        raise ValueError("evaluation.k_values must contain positive integers")
    max_users = config["evaluation"].get("max_users")

    # Build catalog and ground truth
    catalog = splits.train["video_id"].drop_duplicates().tolist()
    training_positives = splits.train.loc[
        splits.train[label_col] > 0, ["user_id", "video_id"]
    ].drop_duplicates()
    training_seen = {
        user: set(group["video_id"].tolist())
        for user, group in training_positives.groupby("user_id", sort=False)
    }
    ground_truth = {
        "validation": build_ground_truth(
            splits.validation,
            label_col=label_col,
            catalog=catalog,
            exclude=training_seen,
        ),
        "test": build_ground_truth(
            splits.test,
            label_col=label_col,
            catalog=catalog,
            exclude=training_seen,
        ),
    }
    users = {
        "validation": _evaluation_users(ground_truth["validation"], max_users, seed),
        "test": _evaluation_users(ground_truth["test"], max_users, seed + 1),
    }

    # Load features
    experiment = config.get("experiment", {})
    variant = str(experiment.get("variant", "feature_history"))
    use_user_features = bool(experiment.get("use_user_features", True))
    use_video_features = bool(experiment.get("use_video_features", True))
    use_history = bool(experiment.get("use_history", True))
    user_features = video_features = None
    raw_dir = config["data"].get("raw_dir")
    if raw_dir and (use_user_features or use_video_features):
        loaded_users, loaded_videos = load_kuairand_features(raw_dir)
        user_features = loaded_users if use_user_features else None
        video_features = loaded_videos if use_video_features else None

    # Train model
    model = TwoTowerRecommender(
        seed=seed, use_history=use_history, **config.get("model", {})
    )
    started = time.perf_counter()
    model.fit(
        splits.train,
        label_col=label_col,
        user_features=user_features,
        video_features=video_features,
    )
    fit_seconds = time.perf_counter() - started

    # Initialize results
    results: dict[str, Any] = {
        "model": "two_tower",
        "variant": variant,
        "features": {
            "user_static": user_features is not None,
            "video_basic": video_features is not None,
            "causal_history": use_history,
            "max_history": model.max_history,
        },
        "label": label_col,
        "fit_seconds": fit_seconds,
        "training_loss": model.training_history,
        "candidate_k": max(k_values),
        "candidates": {},
        "splits": {},
    }

    largest_k = max(k_values)
    warmup_runs = int(config["evaluation"].get("latency_warmup_runs", 1))
    measured_runs = int(config["evaluation"].get("latency_measured_runs", 5))
    faiss_config = config.get("faiss")

    if faiss_config:
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is configured but faiss-cpu is not installed")
        print(f"Using FAISS retrieval (type={faiss_config.get('index_type', 'flat')})")
        retriever: Any = FAISSRetriever(
            index_type=faiss_config.get("index_type", "flat"),
            n_lists=faiss_config.get("n_lists", 100),
            n_probe=faiss_config.get("n_probe", 10),
            hnsw_m=faiss_config.get("hnsw_m", 32),
            use_gpu=faiss_config.get("use_gpu", False),
            seed=seed,
        )
        build_start = time.perf_counter()
        index_stats = retriever.build_index(model.item_vectors, model.item_ids)
        build_time = time.perf_counter() - build_start
        results["model"] = "two_tower_faiss"
        results["variant"] = f"{variant}_faiss_{faiss_config.get('index_type', 'flat')}"
        results["faiss"] = {
            "index_type": faiss_config.get("index_type", "flat"),
            "index_stats": index_stats,
            "index_build_seconds": build_time,
        }
    else:
        retriever = ExactRetriever(model.item_vectors, model.item_ids)

    for split_name in ("validation", "test"):
        user_embeddings = _get_user_embeddings_batch(
            model, users[split_name], batch_size=256
        )
        recommendations, latency = _benchmark_retriever(
            model=model,
            user_ids=users[split_name],
            retriever=retriever,
            k=largest_k,
            precomputed_embeddings=user_embeddings,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        split_results: dict[str, Any] = {
            "latency": latency,
            "milliseconds_per_user": latency["end_to_end"]["median_ms_per_user"],
            "metrics": {},
        }
        for k in k_values:
            split_results["metrics"][str(k)] = evaluate_recommendations(
                recommendations,
                ground_truth[split_name],
                k=k,
                catalog=catalog,
            )
        results["candidates"][split_name] = _candidate_rows(
            split_name,
            users[split_name],
            recommendations,
        )
        results["splits"][split_name] = split_results

    return results


def run_faiss_tradeoff_analysis(
    splits: Week1Splits,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run tradeoff analysis with different FAISS configurations."""

    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")

    # ============================================================
    # 1. Experiment settings
    # ============================================================
    seed = int(config.get("seed", 2026))
    label_col = config["data"].get("positive_column", "is_click")
    k_values = sorted(
        {int(k) for k in config["evaluation"]["k_values"]}
    )
    max_users = config["evaluation"].get("max_users")
    largest_k = max(k_values)
    warmup_runs = int(config["evaluation"].get("latency_warmup_runs", 1))
    measured_runs = int(config["evaluation"].get("latency_measured_runs", 5))

    # ============================================================
    # 2. Build ground truth
    # ============================================================
    catalog = (
        splits.train["video_id"]
        .drop_duplicates()
        .tolist()
    )

    training_positives = splits.train.loc[
        splits.train[label_col] > 0,
        ["user_id", "video_id"],
    ].drop_duplicates()

    training_seen = {
        user: set(group["video_id"].tolist())
        for user, group in training_positives.groupby(
            "user_id",
            sort=False,
        )
    }

    ground_truth = {
        "validation": build_ground_truth(
            splits.validation,
            label_col=label_col,
            catalog=catalog,
            exclude=training_seen,
        ),
        "test": build_ground_truth(
            splits.test,
            label_col=label_col,
            catalog=catalog,
            exclude=training_seen,
        ),
    }

    users = {
        "validation": _evaluation_users(
            ground_truth["validation"],
            max_users,
            seed,
        ),
        "test": _evaluation_users(
            ground_truth["test"],
            max_users,
            seed + 1,
        ),
    }

    # ============================================================
    # 3. Load features
    # ============================================================
    experiment = config.get("experiment", {})

    use_user_features = bool(
        experiment.get("use_user_features", True)
    )
    use_video_features = bool(
        experiment.get("use_video_features", True)
    )
    use_history = bool(
        experiment.get("use_history", True)
    )

    user_features = None
    video_features = None

    raw_dir = config["data"].get("raw_dir")

    if raw_dir and (
        use_user_features or use_video_features
    ):
        loaded_users, loaded_videos = (
            load_kuairand_features(raw_dir)
        )

        user_features = (
            loaded_users
            if use_user_features
            else None
        )

        video_features = (
            loaded_videos
            if use_video_features
            else None
        )

    # ============================================================
    # 4. TRAIN TWO-TOWER EXACTLY ONCE
    # ============================================================
    print("Training Two-Tower model...")

    model = TwoTowerRecommender(
        seed=seed,
        use_history=use_history,
        **config.get("model", {}),
    )

    train_start = time.perf_counter()

    model.fit(
        splits.train,
        label_col=label_col,
        user_features=user_features,
        video_features=video_features,
    )

    fit_seconds = time.perf_counter() - train_start

    # ============================================================
    # 5. Generate embeddings once for the search-only comparison
    # ============================================================
    video_embeddings = model.item_vectors
    video_ids = model.item_ids

    print("Generating user embeddings...")

    user_embeddings_by_split = {
        split_name: _get_user_embeddings_batch(
            model,
            users[split_name],
            batch_size=256,
        )
        for split_name in ("validation", "test")
    }

    # ============================================================
    # 6. Exact baseline using the same batched retrieval interface
    # ============================================================
    print("Benchmarking exact retrieval...")
    exact_retriever = ExactRetriever(video_embeddings, video_ids)
    baseline_results = {}
    exact_candidates_by_split: dict[str, list[list[Any]]] = {}
    exact_final_by_split: dict[str, dict[Any, list[Any]]] = {}

    for split_name in ("validation", "test"):
        recommendations, latency = _benchmark_retriever(
            model=model,
            user_ids=users[split_name],
            retriever=exact_retriever,
            k=largest_k,
            precomputed_embeddings=user_embeddings_by_split[split_name],
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        split_results = {
            "latency": latency,
            "milliseconds_per_user": latency["end_to_end"]["median_ms_per_user"],
            "metrics": {},
        }
        for k in k_values:
            split_results["metrics"][str(k)] = evaluate_recommendations(
                recommendations,
                ground_truth[split_name],
                k=k,
                catalog=catalog,
            )
        exact_final_by_split[split_name] = recommendations
        search_k = _faiss_search_k(model, users[split_name], largest_k)
        exact_candidates_by_split[split_name], _, _ = exact_retriever.search(
            user_embeddings_by_split[split_name], k=search_k, batch_size=1024
        )
        split_results["approximation"] = _approximation_fidelity(
            exact_candidates=exact_candidates_by_split[split_name],
            approximate_candidates=exact_candidates_by_split[split_name],
            exact_final=recommendations,
            approximate_final=recommendations,
            user_ids=users[split_name],
            k_values=k_values,
            candidate_eligible_users=set(model.user_to_index),
        )
        baseline_results[split_name] = split_results

    # ============================================================
    # 7. FAISS configurations
    # ============================================================
    index_configs = [
        {
            "index_type": "flat",
            "label": "FAISS-Flat",
        },
        {
            "index_type": "ivf",
            "n_lists": 50,
            "n_probe": 5,
            "label": "FAISS-IVF-50-5",
        },
        {
            "index_type": "ivf",
            "n_lists": 100,
            "n_probe": 10,
            "label": "FAISS-IVF-100-10",
        },
        {
            "index_type": "ivf",
            "n_lists": 200,
            "n_probe": 20,
            "label": "FAISS-IVF-200-20",
        },
        {
            "index_type": "hnsw",
            "hnsw_m": 16,
            "label": "FAISS-HNSW-16",
        },
        {
            "index_type": "hnsw",
            "hnsw_m": 32,
            "label": "FAISS-HNSW-32",
        },
    ]

    tradeoff_results = {
        "fit_seconds": fit_seconds,
        "latency_protocol": {
            "clock": "time.perf_counter",
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "search_batch_size": 1024,
            "user_embedding_batch_size": 256,
            "catalog_vectors": len(video_ids),
            "embedding_dimension": int(video_embeddings.shape[1]),
            "environment": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "faiss_threads": faiss.omp_get_max_threads(),
            },
            "search_only_definition": "precomputed user embeddings through backend candidate search",
            "retrieval_pipeline_definition": "backend search plus shared seen-item filtering and fallback",
            "end_to_end_definition": "user embedding generation through search and shared post-processing",
            "index_build_excluded": True,
        },
        "baseline": baseline_results,
        "faiss_configs": {},
    }

    # ============================================================
    # 8. Test every retrieval backend
    # ============================================================
    for idx_config in index_configs:
        label = idx_config["label"]

        print(f"Testing {label}...")

        retriever = FAISSRetriever(
            index_type=idx_config["index_type"],
            n_lists=idx_config.get(
                "n_lists",
                100,
            ),
            n_probe=idx_config.get(
                "n_probe",
                10,
            ),
            hnsw_m=idx_config.get(
                "hnsw_m",
                32,
            ),
            seed=seed,
        )

        # Build index
        build_start = time.perf_counter()

        index_stats = retriever.build_index(
            video_embeddings,
            video_ids,
        )

        build_time = (
            time.perf_counter() - build_start
        )

        config_results = {
            "index_stats": index_stats,
            "build_seconds": build_time,
            "splits": {},
        }

        for split_name in (
            "validation",
            "test",
        ):
            # IMPORTANT:
            # reuse embeddings generated once above
            user_embeddings = (
                user_embeddings_by_split[split_name]
            )

            recommendations, latency = _benchmark_retriever(
                model=model,
                user_ids=users[split_name],
                retriever=retriever,
                k=largest_k,
                precomputed_embeddings=user_embeddings,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
            )
            search_k = _faiss_search_k(model, users[split_name], largest_k)
            approximate_candidates, _, _ = retriever.search(
                user_embeddings, k=search_k, batch_size=1024
            )

            split_results = {
                "latency": latency,
                "milliseconds_per_user": latency["end_to_end"]["median_ms_per_user"],
                "approximation": _approximation_fidelity(
                    exact_candidates=exact_candidates_by_split[split_name],
                    approximate_candidates=approximate_candidates,
                    exact_final=exact_final_by_split[split_name],
                    approximate_final=recommendations,
                    user_ids=users[split_name],
                    k_values=k_values,
                    candidate_eligible_users=set(model.user_to_index),
                ),
                "metrics": {},
            }

            for k in k_values:
                split_results["metrics"][str(k)] = (
                    evaluate_recommendations(
                        recommendations,
                        ground_truth[split_name],
                        k=k,
                        catalog=catalog,
                    )
                )

            config_results["splits"][
                split_name
            ] = split_results

        tradeoff_results["faiss_configs"][
            label
        ] = config_results

    return tradeoff_results


def save_week2_results(
    results: dict[str, Any],
    artifacts_dir: str | Path,
    candidates_path: str | Path | None = None,
) -> None:
    """Save compact metrics plus reusable retrieval candidates."""
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    variant = re.sub(r"[^a-zA-Z0-9_-]+", "_", results.get("variant", "two_tower"))
    stem = f"week2_{variant}_results"

    # Keep the metrics JSON compact. Candidate rows are a separate, typed table
    # that Week 3 can load directly without parsing a very large nested JSON file.
    summary = {key: value for key, value in results.items() if key != "candidates"}
    candidates = [
        row
        for split_rows in results.get("candidates", {}).values()
        for row in split_rows
    ]
    candidate_output = Path(candidates_path) if candidates_path is not None else None
    summary["candidate_data"] = (
        str(candidate_output) if candidates and candidate_output is not None else None
    )
    summary["candidate_rows"] = len(candidates)
    with (output / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if candidates and candidate_output is not None:
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(candidates).to_csv(
            candidate_output, index=False, compression="gzip"
        )

    # Save CSV
    rows: list[dict[str, Any]] = []
    for split, split_result in results["splits"].items():
        for k, metrics in split_result["metrics"].items():
            rows.append({
                "model": results["model"],
                "variant": results.get("variant", "two_tower"),
                "user_static": results.get("features", {}).get("user_static", False),
                "video_basic": results.get("features", {}).get("video_basic", False),
                "causal_history": results.get("features", {}).get("causal_history", False),
                "split": split,
                "k": int(k),
                "recall": metrics[f"recall@{k}"],
                "hit_rate": metrics[f"hit_rate@{k}"],
                "ndcg": metrics[f"ndcg@{k}"],
                "coverage": metrics[f"coverage@{k}"],
                "evaluated_users": metrics["evaluated_users"],
                "milliseconds_per_user": split_result["milliseconds_per_user"],
            })
    pd.DataFrame(rows).to_csv(output / f"{stem}.csv", index=False)


def save_tradeoff_results(
    tradeoff_results: dict[str, Any], artifacts_dir: str | Path
) -> None:
    """Save tradeoff analysis results."""
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with (output / "faiss_tradeoff_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(tradeoff_results, handle, indent=2)

    # Create comparison CSV
    rows = []

    def latency_columns(split_data: dict[str, Any]) -> dict[str, float]:
        latency = split_data["latency"]
        return {
            "search_median_ms_per_user": latency["search_only"]["median_ms_per_user"],
            "search_p95_ms_per_user": latency["search_only"]["p95_ms_per_user"],
            "pipeline_median_ms_per_user": latency["retrieval_pipeline"]["median_ms_per_user"],
            "pipeline_p95_ms_per_user": latency["retrieval_pipeline"]["p95_ms_per_user"],
            "end_to_end_median_ms_per_user": latency["end_to_end"]["median_ms_per_user"],
            "end_to_end_p95_ms_per_user": latency["end_to_end"]["p95_ms_per_user"],
        }

    def fidelity_columns(split_data: dict[str, Any], k: str) -> dict[str, float]:
        fidelity = split_data["approximation"][k]
        return {
            "candidate_recall_vs_exact": fidelity[f"candidate_recall@{k}"],
            "candidate_ndcg_vs_exact": fidelity[f"candidate_ndcg@{k}"],
            "final_overlap_vs_exact": fidelity[f"final_overlap@{k}"],
            "final_exact_match_rate": fidelity[f"final_exact_match_rate@{k}"],
        }

    # Add baseline
    for split, split_data in tradeoff_results["baseline"].items():
        for k, metrics in split_data["metrics"].items():
            rows.append({
                "method": "Exact-NumPy",
                "split": split,
                "k": int(k),
                "recall": metrics[f"recall@{k}"],
                "hit_rate": metrics[f"hit_rate@{k}"],
                "ndcg": metrics[f"ndcg@{k}"],
                **latency_columns(split_data),
                **fidelity_columns(split_data, k),
            })

    # Add FAISS configs
    for method, config_data in tradeoff_results["faiss_configs"].items():
        for split, split_data in config_data["splits"].items():
            for k, metrics in split_data["metrics"].items():
                rows.append({
                    "method": method,
                    "split": split,
                    "k": int(k),
                    "recall": metrics[f"recall@{k}"],
                    "hit_rate": metrics[f"hit_rate@{k}"],
                    "ndcg": metrics[f"ndcg@{k}"],
                    **latency_columns(split_data),
                    **fidelity_columns(split_data, k),
                })

    pd.DataFrame(rows).to_csv(output / "faiss_tradeoff_comparison.csv", index=False)
