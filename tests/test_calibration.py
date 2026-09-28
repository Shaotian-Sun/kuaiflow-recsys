import json
import math
import unittest

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from kuaiflow.calibration import (
    apply_calibrator,
    binary_metrics,
    fit_calibrator,
    paired_cluster_bootstrap,
    reliability_bins,
)


class CalibrationTests(unittest.TestCase):
    def test_recovers_probability_distortion_and_improves_held_out_losses(self) -> None:
        rng = np.random.default_rng(52)
        latent = rng.normal(size=24000)
        labels = rng.binomial(1, expit(latent))
        logits = 3.0 * latent + 1.7
        fitted = fit_calibrator(logits[:16000], labels[:16000])
        self.assertAlmostEqual(fitted["slope"], 1.0 / 3.0, delta=0.03)
        self.assertAlmostEqual(fitted["intercept"], -1.7 / 3.0, delta=0.06)
        raw = binary_metrics(labels[16000:], expit(logits[16000:]))
        calibrated = binary_metrics(labels[16000:], apply_calibrator(logits[16000:], fitted))
        self.assertLess(calibrated["log_loss"], raw["log_loss"] - 0.1)
        self.assertLess(calibrated["brier"], raw["brier"] - 0.02)
        self.assertLess(calibrated["ece"], raw["ece"] / 3.0)
        self.assertAlmostEqual(raw["roc_auc"], calibrated["roc_auc"])
        self.assertAlmostEqual(raw["pr_auc"], calibrated["pr_auc"])

    def test_json_reload_and_positive_slope_preserve_ranking(self) -> None:
        logits = np.array([-3.0, -0.1, 0.2, 0.6, 1.2, 2.0, 4.0])
        fitted = fit_calibrator(logits, [0, 0, 1, 0, 1, 0, 1])
        reloaded = json.loads(json.dumps(fitted, allow_nan=False))
        predictions = apply_calibrator(logits, reloaded)
        self.assertGreater(fitted["slope"], 0.0)
        np.testing.assert_array_equal(np.argsort(predictions), np.argsort(logits))
        np.testing.assert_array_equal(predictions, apply_calibrator(logits, fitted))

    def test_negative_relationship_cannot_reverse_ranking(self) -> None:
        fitted = fit_calibrator([-2, -1, 1, 2], [1, 1, 0, 0])
        self.assertAlmostEqual(fitted["slope"], 1e-6)
        self.assertTrue(np.all(np.diff(apply_calibrator([-2, -1, 1, 2], fitted)) > 0))

    def test_single_class_uses_explicit_smoothed_constant(self) -> None:
        for labels, expected in [([0, 0, 0], 0.125), ([1, 1, 1], 0.875)]:
            with self.subTest(labels=labels):
                fitted = fit_calibrator([-4, 0, 4], labels)
                self.assertEqual(fitted["method"], "smoothed_constant")
                self.assertEqual(fitted["slope"], 0.0)
                np.testing.assert_allclose(apply_calibrator([-1e6, 0, 1e6], fitted), expected)
                self.assertIsNone(binary_metrics(labels, [expected] * 3)["roc_auc"])
                self.assertIsNone(binary_metrics(labels, [expected] * 3)["pr_auc"])

    def test_known_metrics_and_impression_weighted_ece(self) -> None:
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], n_bins=2)
        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_prediction"], 0.4125)
        self.assertEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["pr_auc"], 5 / 6)
        self.assertAlmostEqual(metrics["brier"], 0.158125)
        self.assertAlmostEqual(metrics["log_loss"], -math.log(0.9 * 0.6 * 0.35 * 0.8) / 4)
        self.assertAlmostEqual(metrics["ece"], 0.0875)

    def test_discrimination_ties_match_reference_metrics(self) -> None:
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1])
        probabilities = np.array([0.8, 0.8, 0.1, 0.1, 0.3, 0.3, 0.3, 0.8, 0.6])
        metrics = binary_metrics(labels, probabilities)
        self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(labels, probabilities))
        self.assertAlmostEqual(metrics["pr_auc"], average_precision_score(labels, probabilities))
        constant = binary_metrics(labels, np.full(len(labels), 0.3))
        self.assertAlmostEqual(constant["roc_auc"], 0.5)
        self.assertAlmostEqual(constant["pr_auc"], labels.mean())

    def test_bin_boundaries_empty_bins_and_endpoint_losses(self) -> None:
        bins = reliability_bins([0, 1, 0, 1], [0.0, 0.1, 0.2, 1.0])
        self.assertEqual([entry["count"] for entry in bins], [1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(bins[9]["upper"], 1.0)
        self.assertIsNone(bins[4]["positive_rate"])
        self.assertIsNone(bins[4]["mean_prediction"])
        metrics = binary_metrics([0, 1], [1.0, 0.0])
        self.assertEqual(metrics["brier"], 1.0)
        self.assertTrue(math.isfinite(metrics["log_loss"]))
        self.assertEqual(metrics["ece"], 1.0)
        json.dumps({"bins": bins, "metrics": metrics}, allow_nan=False)

    def test_empty_handling_is_explicit(self) -> None:
        self.assertEqual(binary_metrics([], []), {
            "rows": 0, "positive_rate": None, "mean_prediction": None,
            "roc_auc": None, "pr_auc": None, "log_loss": None,
            "brier": None, "ece": None,
        })
        self.assertTrue(all(entry["count"] == 0 for entry in reliability_bins([], [])))
        self.assertEqual(apply_calibrator([], {"slope": 1, "intercept": 0}).shape, (0,))
        with self.assertRaisesRegex(ValueError, "empty"):
            fit_calibrator([], [])
        with self.assertRaisesRegex(ValueError, "empty"):
            paired_cluster_bootstrap([], [], [], [])

    def test_invalid_inputs_are_rejected_before_statistics(self) -> None:
        invalid_pairs = [
            ([0, 2], [0.1, 0.2]),
            ([0, np.nan], [0.1, 0.2]),
            ([0, 1], [0.1, np.inf]),
            ([0, 1], [-0.01, 0.2]),
            ([0, 1], [0.1, 1.01]),
            ([[0, 1]], [0.1, 0.2]),
            ([0, 1], [[0.1, 0.2]]),
            ([0, 1], [0.1]),
        ]
        for labels, probabilities in invalid_pairs:
            with self.subTest(labels=labels, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    binary_metrics(labels, probabilities)
        for n_bins in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                reliability_bins([1], [0.5], n_bins=n_bins)
        with self.assertRaises(ValueError):
            fit_calibrator([np.inf, 0], [0, 1])
        with self.assertRaises(ValueError):
            fit_calibrator([0], [0, 1])
        for parameters in [
            {"slope": -1, "intercept": 0}, {"slope": 0, "intercept": 0},
            {"slope": np.nan, "intercept": 0}, {"slope": 1},
        ]:
            with self.assertRaises(ValueError):
                apply_calibrator([0], parameters)


class PairedClusterBootstrapTests(unittest.TestCase):
    def test_identical_predictions_have_exact_zero_paired_intervals(self) -> None:
        result = paired_cluster_bootstrap(
            [0, 1, 1, 0], [0.1, 0.7, 0.4, 0.8], [0.1, 0.7, 0.4, 0.8],
            [4, 4, 8, 10], repeats=200,
        )
        self.assertEqual(result["users"], 3)
        for metric in ["log_loss", "brier"]:
            self.assertEqual(result[metric], {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0})
        json.dumps(result, allow_nan=False)

    def test_cluster_resampling_retains_all_rows_and_impression_weighting(self) -> None:
        # A has three impressions, B one. Mean user loss differences would be
        # -0.35; the required impression-weighted estimate is -0.21.
        labels = np.array([0, 0, 0, 0])
        raw = np.array([0.4, 0.4, 0.4, 0.8])
        calibrated = np.array([0.3, 0.3, 0.3, 0.1])
        result = paired_cluster_bootstrap(labels, raw, calibrated, ["A", "A", "A", "B"], repeats=1000, seed=4)
        differences = calibrated ** 2 - raw ** 2
        counts = np.array([3, 1])
        sums = np.array([differences[:3].sum(), differences[3]])
        rng = np.random.default_rng(4)
        expected_samples = []
        for _ in range(1000):
            selected = rng.integers(0, 2, size=2)
            expected_samples.append(sums[selected].sum() / counts[selected].sum())
        low, high = np.quantile(expected_samples, [0.025, 0.975])
        self.assertAlmostEqual(result["brier"]["estimate"], differences.mean())
        self.assertNotAlmostEqual(result["brier"]["estimate"], (differences[0] + differences[3]) / 2)
        self.assertAlmostEqual(result["brier"]["ci_lower"], low)
        self.assertAlmostEqual(result["brier"]["ci_upper"], high)
        self.assertEqual(result, paired_cluster_bootstrap(labels, raw, calibrated, ["A", "A", "A", "B"], repeats=1000, seed=4))

    def test_single_user_is_a_degenerate_interval(self) -> None:
        result = paired_cluster_bootstrap([0, 1], [0.8, 0.2], [0.2, 0.8], [3, 3], repeats=20)
        for metric in ["log_loss", "brier"]:
            self.assertLess(result[metric]["estimate"], 0.0)
            self.assertEqual(result[metric]["estimate"], result[metric]["ci_lower"])
            self.assertEqual(result[metric]["estimate"], result[metric]["ci_upper"])

    def test_invalid_cluster_inputs_fail_clearly(self) -> None:
        for users in [[1], [1, None], [1, float("nan")], [[1], [2]]]:
            with self.subTest(users=users):
                with self.assertRaises(ValueError):
                    paired_cluster_bootstrap([0, 1], [0.1, 0.9], [0.2, 0.8], users)
        for repeats in [0, 1.5, True]:
            with self.assertRaises(ValueError):
                paired_cluster_bootstrap([0, 1], [0.1, 0.9], [0.2, 0.8], [1, 2], repeats=repeats)
        for seed in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                paired_cluster_bootstrap([0, 1], [0.1, 0.9], [0.2, 0.8], [1, 2], seed=seed)


if __name__ == "__main__":
    unittest.main()
