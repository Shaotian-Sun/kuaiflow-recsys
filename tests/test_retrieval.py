import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from kuaiflow.retrieval import run_week2_retrieval, save_week2_results
from kuaiflow.toy import make_toy_splits


class RetrievalTests(unittest.TestCase):
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
            save_week2_results(results, directory)
            self.assertTrue(Path(directory, "week2_id_only_results.json").exists())
            self.assertTrue(Path(directory, "week2_id_only_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
