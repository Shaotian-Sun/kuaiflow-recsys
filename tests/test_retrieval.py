import unittest

from kuaiflow.retrieval import run_week2_retrieval
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
        self.assertEqual(len(results["training_loss"]), 5)
        for split in ("validation", "test"):
            self.assertEqual(set(results["splits"][split]["metrics"]), {"1", "2"})
            self.assertEqual(
                results["splits"][split]["metrics"]["2"]["evaluated_users"],
                4.0,
            )


if __name__ == "__main__":
    unittest.main()
