import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import kuaiflow.retrieval as retrieval_module
from kuaiflow.cli import _validate_retrieval_mode
from kuaiflow.retrieval import (
    FAISS_AVAILABLE,
    FAISSRetriever,
    _approximation_fidelity,
    _faiss_search_k,
    _postprocess_faiss_recommendations,
    run_week2_retrieval,
    save_week2_results,
)
from kuaiflow.toy import make_toy_splits


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if FAISS_AVAILABLE:
            cls.faiss_threads = retrieval_module.faiss.omp_get_max_threads()
            retrieval_module.faiss.omp_set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.torch_threads)
        if FAISS_AVAILABLE:
            retrieval_module.faiss.omp_set_num_threads(cls.faiss_threads)

    def test_week2_toy_pipeline(self) -> None:
        config = {
            "seed": 1,
            "data": {"positive_column": "is_click"},
            "evaluation": {"k_values": [1, 2], "max_users": None},
            "model": {
                "embedding_dim": 8,
                "hidden_dim": 16,
                "learning_rate": 0.01,
                "epochs": 5,
                "batch_size": 4,
            },
        }
        results = run_week2_retrieval(make_toy_splits(), config)

        self.assertEqual(results["model"], "two_tower")
        self.assertEqual(results["variant"], "feature_history")
        self.assertEqual(len(results["training_loss"]), 5)
        for split in ("validation", "test"):
            self.assertEqual(set(results["splits"][split]["metrics"]), {"1", "2"})
            latency = results["splits"][split]["latency"]
            self.assertEqual(latency["measured_runs"], 5)
            for boundary in ("search_only", "retrieval_pipeline", "end_to_end"):
                self.assertGreaterEqual(latency[boundary]["median_ms_per_user"], 0)
                self.assertEqual(len(latency[boundary]["samples_ms_per_user"]), 5)
            self.assertEqual(
                results["splits"][split]["metrics"]["2"]["evaluated_users"],
                4.0,
            )

    def test_variant_controls_and_artifact_names(self) -> None:
        config = {
            "seed": 1,
            "experiment": {
                "variant": "id_only",
                "use_user_features": False,
                "use_video_features": False,
                "use_history": False,
            },
            "data": {"positive_column": "is_click"},
            "evaluation": {"k_values": [2], "max_users": None},
            "model": {
                "embedding_dim": 8, "hidden_dim": 16, "epochs": 1,
                "batch_size": 4,
            },
        }
        results = run_week2_retrieval(make_toy_splits(), config)
        self.assertEqual(results["variant"], "id_only")
        self.assertFalse(results["features"]["user_static"])
        self.assertFalse(results["features"]["video_basic"])
        self.assertFalse(results["features"]["causal_history"])
        with TemporaryDirectory() as directory:
            candidate_path = Path(
                directory, "processed", "week2_id_only_top2.csv.gz"
            )
            save_week2_results(results, directory, candidate_path)
            self.assertTrue(Path(directory, "week2_id_only_results.json").exists())
            self.assertTrue(Path(directory, "week2_id_only_results.csv").exists())
            self.assertTrue(candidate_path.exists())
            candidates = retrieval_module.pd.read_csv(candidate_path)
            self.assertEqual(
                list(candidates.columns),
                [
                    "split", "user_id", "video_id", "retrieval_rank",
                ],
            )
            self.assertEqual(len(candidates), 16)
            self.assertTrue(candidates.groupby(["split", "user_id"]).size().eq(2).all())
            self.assertTrue(
                candidates.groupby(["split", "user_id"])["retrieval_rank"]
                .apply(list)
                .map(lambda ranks: ranks == [1, 2])
                .all()
            )

    @unittest.skipUnless(FAISS_AVAILABLE, "faiss is not installed")
    def test_faiss_flat_matches_exact_metrics_after_seen_filtering(self) -> None:
        config = {
            "seed": 1,
            "experiment": {
                "variant": "id_only",
                "use_user_features": False,
                "use_video_features": False,
                "use_history": False,
            },
            "data": {"positive_column": "is_click"},
            "evaluation": {"k_values": [1, 2], "max_users": None},
            "model": {
                "embedding_dim": 8,
                "hidden_dim": 16,
                "epochs": 1,
                "batch_size": 4,
            },
        }
        exact = run_week2_retrieval(make_toy_splits(), config)
        faiss_flat = run_week2_retrieval(
            make_toy_splits(), {**config, "faiss": {"index_type": "flat"}}
        )

        for split in ("validation", "test"):
            for k in ("1", "2"):
                exact_metrics = exact["splits"][split]["metrics"][k]
                flat_metrics = faiss_flat["splits"][split]["metrics"][k]
                self.assertEqual(exact_metrics[f"recall@{k}"], flat_metrics[f"recall@{k}"])
                self.assertEqual(exact_metrics[f"ndcg@{k}"], flat_metrics[f"ndcg@{k}"])

    def test_overfetch_and_seen_filtering_use_internal_indices_correctly(self) -> None:
        model = SimpleNamespace(
            item_ids=["a", "b", "c", "d", "e"],
            user_to_index={"known": 0},
            user_seen={"known": {0, 1, 2}},
            popularity=np.asarray([5, 4, 3, 2, 1], dtype=float),
        )
        self.assertEqual(_faiss_search_k(model, ["known"], requested_k=2), 5)
        recommendations = _postprocess_faiss_recommendations(
            model,
            ["known"],
            [["a", "b", "c", "d", "e"]],
            k=2,
        )
        self.assertEqual(recommendations["known"], ["d", "e"])

    def test_cold_user_uses_popularity_fallback(self) -> None:
        model = SimpleNamespace(
            item_ids=["a", "b", "c"],
            user_to_index={},
            user_seen={},
            popularity=np.asarray([1, 3, 2], dtype=float),
        )
        recommendations = _postprocess_faiss_recommendations(
            model, ["cold"], [["a", "c"]], k=2
        )
        self.assertEqual(recommendations["cold"], ["b", "c"])

    @unittest.skipUnless(FAISS_AVAILABLE, "faiss is not installed")
    def test_approximate_indexes_return_valid_catalog_items(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(160, 8)).astype(np.float32)
        queries = rng.normal(size=(4, 8)).astype(np.float32)
        ids = list(range(100, 260))

        configurations = (
            {"index_type": "ivf", "n_lists": 4, "n_probe": 2},
            {"index_type": "hnsw", "hnsw_m": 8},
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                retriever = FAISSRetriever(**configuration)
                retriever.build_index(embeddings, ids)
                recommendations, scores, _ = retriever.search(queries, k=5)
                self.assertEqual(len(recommendations), len(queries))
                for row, score_row in zip(recommendations, scores):
                    self.assertEqual(len(row), 5)
                    self.assertEqual(len(score_row), 5)
                    self.assertTrue(set(row).issubset(ids))

    def test_missing_faiss_dependency_has_actionable_error(self) -> None:
        with patch("kuaiflow.retrieval.FAISS_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "pip install faiss-cpu"):
                FAISSRetriever()

    def test_cli_retrieval_modes_reject_ambiguous_configs(self) -> None:
        _validate_retrieval_mode("standard", {})
        _validate_retrieval_mode("faiss", {"faiss": {"index_type": "hnsw"}})
        _validate_retrieval_mode("tradeoff", {})

        invalid = (
            ("standard", {"faiss": {"index_type": "hnsw"}}),
            ("faiss", {}),
            ("tradeoff", {"faiss": {"index_type": "hnsw"}}),
        )
        for mode, config in invalid:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    _validate_retrieval_mode(mode, config)

    def test_approximation_fidelity_compares_candidates_and_final_lists(self) -> None:
        fidelity = _approximation_fidelity(
            exact_candidates=[["a", "b", "c"], ["d", "e", "f"]],
            approximate_candidates=[["a", "c", "x"], ["d", "e", "f"]],
            exact_final={"u1": ["a", "b"], "u2": ["d", "e"]},
            approximate_final={"u1": ["a", "c"], "u2": ["d", "e"]},
            user_ids=["u1", "u2"],
            k_values=[2, 3],
        )
        self.assertAlmostEqual(fidelity["3"]["candidate_recall@3"], 5 / 6)
        self.assertAlmostEqual(fidelity["2"]["final_overlap@2"], 0.75)
        self.assertAlmostEqual(fidelity["2"]["final_exact_match_rate@2"], 0.5)
        self.assertGreater(fidelity["3"]["candidate_ndcg@3"], 0)
        self.assertLess(fidelity["3"]["candidate_ndcg@3"], 1)

        known_only = _approximation_fidelity(
            exact_candidates=[["a"], ["b"]],
            approximate_candidates=[["x"], ["b"]],
            exact_final={"cold": ["a"], "known": ["b"]},
            approximate_final={"cold": ["a"], "known": ["b"]},
            user_ids=["cold", "known"],
            k_values=[1],
            candidate_eligible_users={"known"},
        )["1"]
        self.assertEqual(known_only["candidate_recall@1"], 1.0)
        self.assertEqual(known_only["candidate_evaluated_users"], 1.0)
        self.assertEqual(known_only["final_evaluated_users"], 2.0)

        with self.assertRaises(ValueError):
            _approximation_fidelity(
                exact_candidates=[["a"]],
                approximate_candidates=[],
                exact_final={"u": ["a"]},
                approximate_final={"u": ["a"]},
                user_ids=["u"],
                k_values=[1],
            )


if __name__ == "__main__":
    unittest.main()
