"""Week 4 cohort isolation and frozen-checkpoint integration tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from kuaiflow.calibration import binary_metrics
from kuaiflow.data import Week1Splits
from kuaiflow.history import ClickHistory
from kuaiflow.models.deepfm import DeepFM
from kuaiflow.models.din import DIN
from kuaiflow.ranking import DeepFMFeatures
from kuaiflow.week4 import (
    evaluate_predictions,
    run_week4_audit,
    score_cohorts,
    split_audit_cohorts,
    training_segments,
)


def _frame(times, randomized=False):
    n = len(times)
    return pd.DataFrame({
        "user_id": [1 + i % 2 for i in range(n)],
        "video_id": [10 + i % 2 for i in range(n)],
        "time_ms": times,
        "is_click": [i % 2 for i in range(n)],
        "is_rand": [int(randomized)] * n,
        "tab": [1 if i % 3 else 2 for i in range(n)],
    })


def _splits():
    return Week1Splits(
        train=_frame([1, 2, 3]),
        validation=_frame([10, 20]),
        test=_frame([21, 25, 30, 35, 40]),
        random_audit=_frame([3, 11, 20, 20, 22, 25, 35, 37, 45], True),
    )


class Week4CohortTests(unittest.TestCase):
    def test_calibration_ties_training_exclusion_and_common_interval(self):
        splits = _splits()
        original = splits.random_audit.copy(deep=True)
        cohorts, protocol = split_audit_cohorts(splits)
        self.assertEqual(cohorts["random_calibration"].time_ms.tolist(), [11, 20, 20])
        self.assertEqual(cohorts["standard_test"].time_ms.tolist(), [25, 30, 35, 40])
        self.assertEqual(cohorts["random_test"].time_ms.tolist(), [22, 25, 35, 37])
        self.assertEqual(protocol["test_start_ms"], 22)
        self.assertEqual(protocol["test_end_ms"], 40)
        self.assertEqual(protocol["dropped_random_before_or_at_training"], 1)
        self.assertEqual(protocol["dropped_outside_common_test"],
                         {"standard_test": 1, "random_test": 1})
        for name in ("standard_test", "random_test"):
            self.assertTrue(cohorts[name].time_ms.gt(20).all())
        pd.testing.assert_frame_equal(splits.random_audit, original)

    def test_input_order_and_indices_do_not_change_membership(self):
        splits = _splits()
        unordered = Week1Splits(**{
            name: getattr(splits, name).sample(frac=1, random_state=7)
            for name in ("train", "validation", "test", "random_audit")
        })
        first, _ = split_audit_cohorts(splits)
        second, _ = split_audit_cohorts(unordered)
        for name in first:
            self.assertEqual(second[name].index.tolist(), list(range(len(second[name]))))
            keys = ["time_ms", "user_id", "video_id", "is_click"]
            pd.testing.assert_frame_equal(
                first[name].sort_values(keys).reset_index(drop=True),
                second[name].sort_values(keys).reset_index(drop=True),
            )

    def test_rejects_standard_test_tie_at_calibration_boundary(self):
        splits = _splits()
        splits.test.loc[0, "time_ms"] = 20
        with self.assertRaisesRegex(ValueError, "strictly after.*calibration"):
            split_audit_cohorts(splits)

    def test_rejects_invalid_flags_labels_and_timestamps(self):
        for name, column, value in (
            ("random_audit", "is_rand", 0),
            ("validation", "is_rand", 1),
            ("test", "is_click", 0.5),
            ("random_audit", "time_ms", np.nan),
            ("random_audit", "time_ms", 2.5),
        ):
            with self.subTest(name=name, column=column, value=value):
                splits = _splits()
                frame = getattr(splits, name)
                frame[column] = frame[column].astype(float)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    split_audit_cohorts(splits)

    def test_rejects_missing_random_calibration_or_common_support(self):
        splits = _splits()
        for random in (_frame([30, 40], True), _frame([10, 100, 110], True),
                       _frame([], True)):
            with self.subTest(times=random.time_ms.tolist()):
                changed = Week1Splits(splits.train, splits.validation, splits.test, random)
                with self.assertRaises(ValueError):
                    split_audit_cohorts(changed)

    def test_segments_use_training_counts_and_keep_unknowns_separate(self):
        train = pd.DataFrame({
            "user_id": [1] * 5 + [2] * 2 + [3],
            "video_id": [10] * 4 + [11] * 2 + [12, 13],
            "is_click": [0] * 8,
        })
        frame = pd.DataFrame({"user_id": [1, 2, 99, 3],
                              "video_id": [10, 13, 20, 11], "tab": [1, 2, 1, 1],
                              "is_click": [1] * 4})
        masks = training_segments(train, frame)
        self.assertEqual(masks["item=head"].tolist(), [True, False, False, False])
        self.assertEqual(masks["item=tail"].tolist(), [False, True, False, True])
        self.assertEqual(masks["item=unseen"].tolist(), [False, False, True, False])
        self.assertEqual(masks["user=low_activity"].tolist(), [False, True, False, True])
        self.assertEqual(masks["user=high_activity"].tolist(), [True, False, False, False])
        self.assertEqual(masks["warm_tab=1"].tolist(), [True, False, False, True])
        expanded = pd.concat([frame, frame.iloc[[2]]] * 8, ignore_index=True)
        repeated = training_segments(train.assign(is_click=1), expanded)
        for name in masks:
            np.testing.assert_array_equal(masks[name], repeated[name][:len(frame)])


class Week4EvaluationTests(unittest.TestCase):
    def test_rejects_invalid_run_controls_before_reading_data(self):
        for config in (
            {"models": []}, {"models": ["deepfm", "deepfm"]}, {"models": ["unknown"]},
            {"evaluation": {"batch_size": True}}, {"evaluation": {"batch_size": 1.5}},
            {"evaluation": {"calibration_bins": 0}}, {"evaluation": {"calibration_bins": 2.5}},
            {"evaluation": {"bootstrap_repeats": -1}},
            {"evaluation": {"bootstrap_repeats": False}},
            {"seed": -1}, {"seed": 1.5}, {"save_logits": "false"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_week4_audit(config)

    def test_final_labels_cannot_change_calibration_parameters(self):
        splits = _splits()
        cohorts, _ = split_audit_cohorts(splits)
        predictions = {"deepfm": {name: {"click": np.linspace(-1, 1, len(frame))}
                                    for name, frame in cohorts.items()}}
        kwargs = {"n_bins": 3, "bootstrap_repeats": 8, "seed": 11}
        first = evaluate_predictions(splits.train, cohorts, predictions,
                                     {"deepfm": {"click": "is_click"}}, **kwargs)
        modified = {name: frame.copy(deep=True) for name, frame in cohorts.items()}
        for name in ("standard_test", "random_test"):
            modified[name]["is_click"] = 1 - modified[name].is_click
        second = evaluate_predictions(splits.train, modified, predictions,
                                      {"deepfm": {"click": "is_click"}}, **kwargs)
        self.assertEqual(first["calibrators"], second["calibrators"])
        self.assertNotEqual(first["metrics"], second["metrics"])
        self.assertEqual(len(first["uncertainty"]), 4)
        for row in first["metrics"]:
            if row["segment"] == "all":
                self.assertEqual(row["rows"], len(cohorts[row["cohort"]]))
        for cohort in ("standard_test", "random_test"):
            for variant in ("raw", "standard_platt", "random_platt"):
                total = sum(row["count"] for row in first["reliability"]
                            if row["cohort"] == cohort and row["variant"] == variant)
                self.assertEqual(total, len(cohorts[cohort]))

    def test_rejects_unaligned_or_nonfinite_predictions(self):
        splits = _splits()
        cohorts, _ = split_audit_cohorts(splits)
        base = {"deepfm": {name: {"click": np.linspace(-1, 1, len(frame))}
                           for name, frame in cohorts.items()}}
        for cohort in cohorts:
            for invalid in (np.array([0.1]), np.full(len(cohorts[cohort]), np.nan)):
                with self.subTest(cohort=cohort, invalid=invalid):
                    predictions = copy.deepcopy(base)
                    predictions["deepfm"][cohort]["click"] = invalid
                    with self.assertRaises(ValueError):
                        evaluate_predictions(splits.train, cohorts, predictions,
                                             {"deepfm": {"click": "is_click"}},
                                             bootstrap_repeats=0)


class Week4ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def _checkpoint(self, root, train, architecture, feature_names=None):
        torch.manual_seed(13)
        encoder = DeepFMFeatures().fit(train, feature_names or ["user_id", "video_id"], [])
        config = {"architecture": architecture, "embedding_dim": 3,
                  "hidden_dims": [4], "dropout": 0.0, "attention_hidden_dims": [4]}
        args = dict(cardinalities=encoder.cardinalities, numeric_dim=0,
                    embedding_dim=3, hidden_dims=[4], dropout=0.0)
        model = (DIN(**args, video_field_index=1, attention_hidden_dims=[4])
                 if architecture == "din" else DeepFM(**args))
        prefix = root / f"week3_{architecture}"
        history = ClickHistory(3).fit(train, encoder) if architecture == "din" else None
        if history is not None:
            history.save(str(prefix) + "_history.npz")
        torch.save({"state_dict": model.state_dict(), "model_config": config,
                    "cardinalities": encoder.cardinalities,
                    "categorical_features": encoder.categorical,
                    "numeric_features": encoder.numeric,
                    "history_file": f"week3_{architecture}_history.npz" if history else None},
                   str(prefix) + "_model.pt")
        Path(str(prefix) + "_encoder.json").write_text(json.dumps(encoder.to_dict()))
        validation = _splits().validation
        cat, num = encoder.transform(validation)
        inputs = [torch.tensor(cat), torch.tensor(num)]
        if history is not None:
            inputs.append(torch.tensor(history.transform(validation), dtype=torch.long))
        model.eval()
        with torch.no_grad():
            logits = model(*inputs).numpy().astype(float)
        Path(str(prefix) + "_results.json").write_text(json.dumps({
            "model": architecture,
            "pointwise": {"validation": {
                **binary_metrics(validation.is_click, expit(logits)),
                "examples": len(validation),
            }},
        }))
        return model.eval(), encoder, history

    def test_reloaded_predictions_match_direct_model_with_unknown_ids_and_batches(self):
        splits = _splits()
        frame = _frame([21, 22, 23, 24, 25])
        frame.loc[2, ["user_id", "video_id"]] = [99, 999]
        frame.index = [31, 20, 19, 17, 13]
        for architecture in ("deepfm", "din"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model, encoder, history = self._checkpoint(root, splits.train, architecture)
                cat, num = encoder.transform(frame)
                args = [torch.tensor(cat), torch.tensor(num)]
                if history:
                    args.append(torch.tensor(history.transform(frame), dtype=torch.long))
                with torch.no_grad():
                    expected = model(*args).numpy()
                cohorts = {"standard_calibration": splits.validation, "random_test": frame}
                first, tasks, audit = score_cohorts(architecture, root, cohorts,
                                                    None, None, 3, batch_size=2)
                second, _, _ = score_cohorts(architecture, root, cohorts,
                                             None, None, 3, batch_size=20)
                np.testing.assert_allclose(first["random_test"]["click"], expected, atol=1e-7)
                np.testing.assert_allclose(second["random_test"]["click"], expected, atol=1e-7)
                self.assertEqual(tasks, {"click": "is_click"})
                self.assertEqual(audit["feature_oov"]["random_test"]["user_id"]["unknown_rows"], 1)
                self.assertEqual(audit["feature_oov"]["random_test"]["video_id"]["unknown_rows"], 1)
                changed = frame.assign(is_click=1-frame.is_click, is_rand=1, tab=99)
                perturbed, _, _ = score_cohorts(architecture, root,
                                                {**cohorts, "random_test": changed},
                                                None, None, 3, batch_size=2)
                np.testing.assert_array_equal(first["random_test"]["click"],
                                              perturbed["random_test"]["click"])

    def test_scoring_rejects_post_exposure_features_and_history_cutoff_mismatch(self):
        splits = _splits()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root, splits.train, "deepfm", ["user_id", "video_id", "is_click"])
            with self.assertRaisesRegex(ValueError, "Unsafe checkpoint features"):
                score_cohorts("deepfm", root, {"random_test": splits.test}, None, None, 3)
            self._checkpoint(root, splits.train, "din")
            with self.assertRaisesRegex(ValueError, "history cutoff"):
                score_cohorts("din", root, {"random_test": splits.test}, None, None, 2)
            with self.assertRaisesRegex(ValueError, "strictly after training"):
                score_cohorts("din", root, {"random_test": splits.train}, None, None, 3)

    def test_scoring_rejects_inconsistent_prior_validation_result(self):
        splits = _splits()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint(root, splits.train, "deepfm")
            path = root / "week3_deepfm_results.json"
            results = json.loads(path.read_text())
            results["pointwise"]["validation"]["log_loss"] += 0.1
            path.write_text(json.dumps(results))
            with self.assertRaisesRegex(ValueError, "validation log_loss does not reproduce"):
                score_cohorts("deepfm", root, {"standard_calibration": splits.validation},
                              None, None, 3)
            results["model"] = "din"
            path.write_text(json.dumps(results))
            with self.assertRaisesRegex(ValueError, "architecture"):
                score_cohorts("deepfm", root, {"standard_calibration": splits.validation},
                              None, None, 3)


if __name__ == "__main__":
    unittest.main()
