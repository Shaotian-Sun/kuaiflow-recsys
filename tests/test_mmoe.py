import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from kuaiflow.models.mmoe import DeepFMMMoE
from kuaiflow.multitask_objectives import (
    DurationCompletionCurve,
    completion_targets,
    masked_soft_bce_with_logits,
    watch_time_targets,
    weighted_watch_time_bce,
)
from kuaiflow.multitask_ranking import (
    _configured_weights,
    _parse_binary_tasks,
    _parse_negative_tasks,
    load_mmoe_artifacts,
    run_mmoe_ranking,
    save_mmoe_run,
)
from kuaiflow.toy import make_multitask_toy_splits


class MMoEModelTests(unittest.TestCase):
    def test_deepfm_components_and_task_gates(self) -> None:
        torch.manual_seed(7)
        model = DeepFMMMoE(
            cardinalities=[4, 5],
            numeric_dim=1,
            task_names=["click", "forward", "watch_time"],
            embedding_dim=3,
            num_experts=2,
            expert_hidden_dims=[6, 4],
            tower_hidden_dim=3,
            dropout=0.0,
        )
        categorical = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
        numeric = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
        components = model.component_logits(categorical, numeric)

        self.assertEqual(model(categorical, numeric).shape, (2, 3))
        self.assertEqual(components["gate_weights"].shape, (2, 3, 2))
        self.assertTrue(torch.allclose(
            components["gate_weights"].sum(dim=-1), torch.ones(2, 3)
        ))
        expected = components["linear"] + components["fm"] + components["mmoe"]
        self.assertTrue(torch.allclose(model(categorical, numeric), expected))

        fields = model._field_embeddings(categorical, numeric)
        pairwise = sum(
            (fields[:, left] * fields[:, right]).sum(dim=1)
            for left in range(fields.shape[1])
            for right in range(left + 1, fields.shape[1])
        )
        self.assertTrue(torch.allclose(components["shared_fm"], pairwise, atol=1e-7))

        model(categorical, numeric).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.step()
        for embedding in model.feature_embeddings:
            self.assertTrue(torch.equal(embedding.weight[0], torch.zeros(3)))


class MultiTaskObjectiveTests(unittest.TestCase):
    def test_watch_time_weighted_logistic_identity(self) -> None:
        target, weight, seconds = watch_time_targets([0, 1_000, 3_000])
        np.testing.assert_allclose(target, [0.0, 0.5, 0.75])
        np.testing.assert_allclose(weight, [1.0, 2.0, 4.0])
        np.testing.assert_allclose(seconds, [0.0, 1.0, 3.0])

        # A featureless head is optimal at z=log(mean(t)), so exp(z)=mean(t).
        logit = torch.tensor(
            np.log(seconds.mean()), dtype=torch.float32, requires_grad=True
        )
        loss = weighted_watch_time_bce(
            logit.expand(3),
            torch.tensor(target),
            torch.tensor(weight),
            mean_weight=float(weight.mean()),
        )
        loss.backward()
        self.assertAlmostEqual(float(logit.grad), 0.0, places=6)
        self.assertAlmostEqual(
            float(torch.exp(logit).detach()), float(seconds.mean()), places=6
        )

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            weighted_watch_time_bce(
                torch.zeros(1), torch.tensor([2.0]), torch.ones(1)
            )

        one_example_default = weighted_watch_time_bce(
            torch.zeros(1), torch.tensor([0.75]), torch.tensor([4.0])
        )
        one_example_fixed = weighted_watch_time_bce(
            torch.zeros(1),
            torch.tensor([0.75]),
            torch.tensor([4.0]),
            mean_weight=2.0,
        )
        self.assertAlmostEqual(
            float(one_example_fixed), 2.0 * float(one_example_default), places=6
        )

        # Fixed train normalization preserves long-watch influence even for a
        # one-row mini-batch; a per-batch weighted mean would cancel this weight.
        one = weighted_watch_time_bce(
            torch.zeros(1),
            torch.tensor([0.75]),
            torch.tensor([4.0]),
            mean_weight=2.0,
        )
        self.assertAlmostEqual(float(one), 2.0 * np.log(2.0), places=6)

    def test_completion_clips_replays_masks_zero_and_round_trips_curve(self) -> None:
        completion, mask = completion_targets(
            [0, 5_000, 30_000, 4_000], [0, 10_000, 20_000, 8_000]
        )
        np.testing.assert_allclose(completion, [0.0, 0.5, 1.0, 0.5])
        self.assertEqual(mask.tolist(), [False, True, True, True])

        curve = DurationCompletionCurve(n_bins=3).fit(
            [5_000, 10_000, 20_000, 40_000],
            [0.8, 0.6, 0.4, 0.2],
        )
        prediction = curve.predict([5_000, 15_000, 40_000, 0])
        self.assertTrue(np.all(np.diff(prediction[:3]) <= 1e-7))
        restored = DurationCompletionCurve.from_dict(curve.to_dict())
        np.testing.assert_allclose(restored.predict([5_000, 15_000, 40_000, 0]), prediction)

        value = masked_soft_bce_with_logits(
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([False, False]),
        )
        self.assertEqual(float(value), 0.0)


class MMoEPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.torch_threads)

    @staticmethod
    def _candidates() -> pd.DataFrame:
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

    @staticmethod
    def _video_features() -> pd.DataFrame:
        return pd.DataFrame({
            "video_id": [10, 11, 12, 13],
            "video_duration": [10_000.0, 20_000.0, 40_000.0, 80_000.0],
        })

    @staticmethod
    def _config() -> dict:
        return {
            "seed": 3,
            "data": {
                "categorical_features": ["user_id", "video_id"],
                "numeric_features": ["video_duration"],
            },
            "model": {
                "embedding_dim": 4,
                "num_experts": 2,
                "expert_hidden_dims": [8, 4],
                "tower_hidden_dim": 3,
                "dropout": 0.0,
                "device": "cpu",
            },
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.01,
                "early_stopping_patience": 0,
            },
            "ranking": {"duration_curve_bins": 3},
            "evaluation": {
                "candidate_k": 2,
                "expected_users_per_split": 4,
                "candidate_chunk_size": 3,
                "watch_time_prediction_cap_seconds": 200,
                "k_values": [1, 2],
            },
        }

    def test_pipeline_uses_all_targets_and_preserves_top_k_membership(self) -> None:
        splits = make_multitask_toy_splits()
        candidates = self._candidates()
        run = run_mmoe_ranking(
            splits,
            self._config(),
            candidates=candidates,
            video_features=self._video_features(),
        )

        expected_tasks = {
            "click", "like", "follow", "comment", "forward", "long_view",
            "profile_enter", "hate", "watch_time", "completion",
        }
        self.assertEqual(set(run.model.task_names), expected_tasks)
        self.assertFalse(run.results["tasks"]["collection"]["used"])
        self.assertEqual(len(run.reranked_candidates), len(candidates))
        before = set(map(tuple, candidates[["split", "user_id", "video_id"]].to_numpy()))
        after = set(map(tuple, run.reranked_candidates[
            ["split", "user_id", "video_id"]
        ].to_numpy()))
        self.assertEqual(before, after)

        for task in expected_tasks.difference({"watch_time", "completion"}):
            values = run.reranked_candidates[f"p_{task}"].to_numpy()
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((values >= 0.0) & (values <= 1.0)).all())
        self.assertTrue((run.reranked_candidates["estimated_watch_seconds"] >= 0).all())
        self.assertTrue((run.reranked_candidates["completion_lift"] >= 0).all())
        self.assertTrue(
            run.reranked_candidates.groupby(["split", "user_id"])["mmoe_rank"]
            .apply(lambda values: sorted(values.tolist()) == [1, 2])
            .all()
        )
        self.assertIn(
            "task_head_order",
            run.results["candidate_ranking"]["test"]["targets"]["forward"],
        )
        self.assertEqual(
            run.results["candidate_ranking"]["test"]["targets"]["hate"]["direction"],
            "lower_is_better",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "ranking" / "mmoe.csv.gz"
            save_mmoe_run(run, self._config(), root / "artifacts", output)
            loaded, encoder, curve, metadata = load_mmoe_artifacts(
                root / "artifacts" / "week3_mmoe_model.pt",
                root / "artifacts" / "week3_mmoe_encoder.json",
                root / "artifacts" / "week3_mmoe_duration_curve.json",
            )
            sample = pd.DataFrame({
                "user_id": [0, 99],
                "video_id": [12, 13],
                "video_duration": [40_000.0, 80_000.0],
            })
            inputs = run.encoder.transform(sample)
            loaded_inputs = encoder.transform(sample)
            with torch.no_grad():
                original_logits = run.model(
                    torch.as_tensor(inputs[0]), torch.as_tensor(inputs[1])
                )
                loaded_logits = loaded(
                    torch.as_tensor(loaded_inputs[0]),
                    torch.as_tensor(loaded_inputs[1]),
                )
            self.assertTrue(torch.equal(original_logits, loaded_logits))
            np.testing.assert_array_equal(
                curve.predict([10_000, 40_000]),
                run.duration_curve.predict([10_000, 40_000]),
            )
            self.assertEqual(metadata["task_names"], list(run.model.task_names))
            self.assertTrue(output.exists())

    def test_outcome_features_and_unavailable_collection_are_rejected(self) -> None:
        splits = make_multitask_toy_splits()
        candidates = self._candidates()
        config = self._config()
        config["data"]["categorical_features"].append("is_like")
        with self.assertRaisesRegex(ValueError, "cannot be features"):
            run_mmoe_ranking(
                splits,
                config,
                candidates=candidates,
                video_features=self._video_features(),
            )

        config = self._config()
        # The task alias itself is rejected even if somebody maps it to a real
        # but semantically different label.
        config["tasks"] = {"binary": {"collect": "is_like"}}
        with self.assertRaisesRegex(ValueError, "no per-impression collection"):
            run_mmoe_ranking(
                splits,
                config,
                candidates=candidates,
                video_features=self._video_features(),
            )
        for alias in ("is_collect", "collect_rate", "is_save", "bookmark"):
            config = self._config()
            config["tasks"] = {"binary": {alias: "is_like"}}
            with self.assertRaisesRegex(ValueError, "no per-impression collection"):
                run_mmoe_ranking(
                    splits,
                    config,
                    candidates=candidates,
                    video_features=self._video_features(),
                )

    def test_negative_tasks_keep_safe_partial_utility_defaults(self) -> None:
        config = {
            "tasks": {
                "binary": {"click": "is_click", "safety": "is_hate"},
                "negative": ["safety"],
            }
        }
        binary = _parse_binary_tasks(config)
        negative = _parse_negative_tasks(config, binary)
        self.assertEqual(negative, {"safety"})
        defaults = {"click": 1.0, "safety": -1.0}
        weights = _configured_weights(
            {"click": 2.0},
            ["click", "safety"],
            allow_negative=True,
            defaults=defaults,
        )
        self.assertEqual(weights, {"click": 2.0, "safety": -1.0})

    def test_missing_candidate_duration_fails_before_training(self) -> None:
        config = self._config()
        config["data"]["numeric_features"] = []
        with self.assertRaisesRegex(ValueError, "Candidate scoring requires"):
            run_mmoe_ranking(
                make_multitask_toy_splits(),
                config,
                candidates=self._candidates(),
            )


if __name__ == "__main__":
    unittest.main()
