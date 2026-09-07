import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from kuaiflow.models.deepfm import DeepFM
from kuaiflow.ranking import (
    DeepFMFeatures,
    load_deepfm_artifacts,
    run_deepfm_ranking,
    save_deepfm_run,
)
from kuaiflow.toy import make_toy_splits


class DeepFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.torch_threads)

    def test_logit_is_sum_of_three_trainable_branches(self) -> None:
        torch.manual_seed(1)
        model = DeepFM(
            cardinalities=[4, 5],
            numeric_dim=1,
            embedding_dim=3,
            hidden_dims=[4],
            dropout=0.0,
        )
        categorical = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
        numeric = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
        components = model.component_logits(categorical, numeric)
        expected = components["linear"] + components["fm"] + components["deep"]
        self.assertTrue(torch.allclose(model(categorical, numeric), expected))
        for embedding in model.feature_embeddings:
            self.assertTrue(torch.equal(embedding.weight[0], torch.zeros(3)))

        fields = model._field_embeddings(categorical, numeric)
        pairwise = sum(
            (fields[:, left] * fields[:, right]).sum(dim=1)
            for left in range(fields.shape[1])
            for right in range(left + 1, fields.shape[1])
        )
        self.assertTrue(torch.allclose(components["fm"], pairwise, atol=1e-7))

        model(categorical, numeric).sum().backward()
        self.assertIsNotNone(model.first_order_embeddings[0].weight.grad)
        self.assertIsNotNone(model.feature_embeddings[0].weight.grad)
        self.assertIsNotNone(model.numeric_embeddings.grad)
        self.assertIsNotNone(model.deep[0].weight.grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.step()
        for embedding in model.feature_embeddings:
            self.assertTrue(torch.equal(embedding.weight[0], torch.zeros(3)))

    def test_encoder_separates_missing_and_unknown(self) -> None:
        train = pd.DataFrame({"category": ["a", None], "number": [0.0, 3.0]})
        encoder = DeepFMFeatures().fit(train, ["category"], ["number"])
        categorical, numeric = encoder.transform(
            pd.DataFrame({"category": ["new", None], "number": [1.0, None]})
        )
        self.assertEqual(categorical[:, 0].tolist(), [0, 1])
        self.assertTrue(np.isfinite(numeric).all())

    @staticmethod
    def _toy_candidates() -> pd.DataFrame:
        rows = []
        per_user = {
            0: [12, 13],
            1: [13, 12],
            2: [10, 11],
            3: [11, 10],
        }
        for split in ("validation", "test"):
            for user, items in per_user.items():
                for rank, item in enumerate(items, start=1):
                    rows.append((split, user, item, rank))
        return pd.DataFrame(
            rows, columns=["split", "user_id", "video_id", "retrieval_rank"]
        )

    def test_pipeline_trains_on_impressions_and_reranks_same_candidates(self) -> None:
        splits = make_toy_splits()
        config = {
            "seed": 3,
            "data": {
                "target_column": "is_click",
                "categorical_features": ["user_id", "video_id"],
                "numeric_features": [],
            },
            "model": {
                "embedding_dim": 4,
                "hidden_dims": [8],
                "dropout": 0.0,
                "device": "cpu",
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 0.01,
                "early_stopping_patience": 0,
            },
            "evaluation": {
                "candidate_k": 2,
                "expected_users_per_split": 4,
                "candidate_chunk_size": 3,
                "k_values": [1, 2],
            },
        }
        candidates = self._toy_candidates()
        run = run_deepfm_ranking(splits, config, candidates=candidates)

        self.assertEqual(run.results["training_examples"], len(splits.train))
        self.assertEqual(len(run.results["optimization"]["history"]), 2)
        self.assertEqual(len(run.reranked_candidates), len(candidates))
        self.assertEqual(
            set(map(tuple, run.reranked_candidates[["split", "user_id", "video_id"]].to_numpy())),
            set(map(tuple, candidates[["split", "user_id", "video_id"]].to_numpy())),
        )
        self.assertTrue(
            run.reranked_candidates.groupby(["split", "user_id"])["deepfm_rank"]
            .apply(lambda values: sorted(values.tolist()) == [1, 2])
            .all()
        )
        self.assertIn("deepfm_order", run.results["candidate_ranking"]["test"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reranked = root / "processed" / "deepfm.csv.gz"
            save_deepfm_run(run, config, root / "artifacts", reranked)
            self.assertTrue(reranked.exists())
            self.assertTrue((root / "artifacts" / "week3_deepfm_model.pt").exists())
            self.assertTrue((root / "artifacts" / "week3_deepfm_encoder.json").exists())
            self.assertTrue((root / "artifacts" / "week3_deepfm_results.json").exists())
            loaded_model, loaded_encoder = load_deepfm_artifacts(
                root / "artifacts" / "week3_deepfm_model.pt",
                root / "artifacts" / "week3_deepfm_encoder.json",
            )
            sample = pd.DataFrame({"user_id": [0, 99], "video_id": [12, 13]})
            original_inputs = run.encoder.transform(sample)
            loaded_inputs = loaded_encoder.transform(sample)
            self.assertTrue(np.array_equal(original_inputs[0], loaded_inputs[0]))
            with torch.no_grad():
                original_scores = run.model(
                    torch.as_tensor(original_inputs[0], dtype=torch.long),
                    torch.as_tensor(original_inputs[1], dtype=torch.float32),
                )
                loaded_scores = loaded_model(
                    torch.as_tensor(loaded_inputs[0], dtype=torch.long),
                    torch.as_tensor(loaded_inputs[1], dtype=torch.float32),
                )
            self.assertTrue(torch.equal(original_scores, loaded_scores))

    def test_post_exposure_features_are_rejected(self) -> None:
        splits = make_toy_splits()
        config = {
            "data": {
                "categorical_features": ["user_id", "is_like"],
                "numeric_features": [],
            },
            "evaluation": {"candidate_k": 2},
        }
        with self.assertRaisesRegex(ValueError, "cannot be features"):
            run_deepfm_ranking(splits, config, candidates=self._toy_candidates())

    def test_incomplete_candidate_cohort_is_rejected(self) -> None:
        splits = make_toy_splits()
        candidates = self._toy_candidates()
        candidates = candidates.loc[
            ~((candidates["split"] == "test") & (candidates["user_id"] == 3))
        ]
        config = {
            "data": {
                "categorical_features": ["user_id", "video_id"],
                "numeric_features": [],
            },
            "evaluation": {
                "candidate_k": 2,
                "expected_users_per_split": 4,
            },
        }
        with self.assertRaisesRegex(ValueError, "expected_users_per_split"):
            run_deepfm_ranking(splits, config, candidates=candidates)


if __name__ == "__main__":
    unittest.main()
