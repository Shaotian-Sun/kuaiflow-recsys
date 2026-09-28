"""Frozen-model exposure audit with chronological probability calibration.

Randomized early-period labels fit only post-hoc calibrators. No random labels
enter a ranker, feature vocabulary, history, or checkpoint-selection decision.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from kuaiflow.calibration import (
    apply_calibrator, binary_metrics, fit_calibrator,
    paired_cluster_bootstrap, reliability_bins,
)
from kuaiflow.data import (
    Week1Splits, load_prepared, load_kuairand_features,
    _find_unique_file, USER_FEATURE_FILE, VIDEO_FEATURE_FILE,
)
from kuaiflow.history import ClickHistory
from kuaiflow.multitask_ranking import (
    load_mmoe_artifacts, _predict_logits as predict_multi,
    POST_EXPOSURE_OR_POLICY_COLUMNS, UNSAFE_AGGREGATE_FEATURES,
)
from kuaiflow.ranking import (
    _attach_static_features, _predict_logits as predict_single,
    load_deepfm_artifacts,
)

MODEL_NAMES = ("deepfm", "mmoe", "din", "din_mmoe")
CALIBRATION_COHORTS = ("standard_calibration", "random_calibration")
TEST_COHORTS = ("standard_test", "random_test")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_frame(frame, name, randomized):
    required = {"user_id", "video_id", "time_ms", "is_click", "is_rand", "tab"}
    if required.difference(frame):
        raise ValueError(f"{name}: missing columns {sorted(required.difference(frame))}")
    if frame.empty:
        raise ValueError(f"{name}: empty cohort")
    if frame[["user_id", "video_id", "tab"]].isna().any().any():
        raise ValueError(f"{name}: missing user, item, or scenario")
    ClickHistory._timestamps(frame)
    for column in ("is_click", "is_rand"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{name}: {column} must be binary without missing values")
    if not frame.is_rand.eq(int(randomized)).all():
        raise ValueError(f"{name}: incorrect is_rand exposure-policy flag")


def split_audit_cohorts(splits: Week1Splits):
    """Align random calibration to the existing standard validation boundary.

    Equal-time impressions never straddle calibration and final evaluation.
    Both final cohorts are clipped to their common observed time interval.
    """
    for name, frame in (("train", splits.train), ("validation", splits.validation),
                        ("test", splits.test), ("random", splits.random_audit)):
        _validate_frame(frame, name, name == "random")
    train_end = int(splits.train.time_ms.max())
    calibration_end = int(splits.validation.time_ms.max())
    if splits.validation.time_ms.min() <= train_end:
        raise ValueError("Validation must be strictly after training")
    if splits.test.time_ms.min() <= calibration_end:
        raise ValueError("Test must be strictly after the calibration cutoff (including ties)")
    random = splits.random_audit
    early = random.loc[(random.time_ms > train_end) & (random.time_ms <= calibration_end)]
    late = random.loc[random.time_ms > calibration_end]
    if early.empty or late.empty:
        raise ValueError("Random calibration and test cohorts must both be nonempty")
    start = int(max(splits.test.time_ms.min(), late.time_ms.min()))
    end = int(min(splits.test.time_ms.max(), late.time_ms.max()))
    if start > end:
        raise ValueError("Standard and random test cohorts have no common time interval")
    standard_test = splits.test.loc[splits.test.time_ms.between(start, end)]
    random_test = late.loc[late.time_ms.between(start, end)]
    cohorts = {
        "standard_calibration": splits.validation,
        "random_calibration": early,
        "standard_test": standard_test,
        "random_test": random_test,
    }
    for name, frame in cohorts.items():
        if frame.empty:
            raise ValueError(f"{name}: no rows in the common test interval")
    cohorts = {name: frame.sort_values("time_ms", kind="stable").reset_index(drop=True)
               for name, frame in cohorts.items()}
    return cohorts, {
        "train_cutoff_ms": train_end,
        "calibration_cutoff_ms": calibration_end,
        "test_start_ms": start,
        "test_end_ms": end,
        "dropped_random_before_or_at_training": int((random.time_ms <= train_end).sum()),
        "dropped_outside_common_test": {
            "standard_test": len(splits.test) - len(standard_test),
            "random_test": len(late) - len(random_test),
        },
        "rankers_retrained": False,
        "calibration_fitting": "standard validation or early randomized exposures only",
        "final_test_used_for_selection": False,
        "history": "frozen at the original training cutoff",
        "propensity_estimation_or_off_policy_value": False,
    }


def training_segments(train, frame):
    """Masks from original training counts only; row frequencies are exposures."""
    item_counts = train.groupby("video_id", sort=True).size().sort_values(ascending=False, kind="stable")
    head = item_counts.index[:max(1, int(np.ceil(len(item_counts) * 0.2)))]
    activity = train.groupby("user_id").size()
    counts = frame.user_id.map(activity).fillna(0).to_numpy()
    known_user = counts > 0
    known_item = frame.video_id.isin(item_counts.index).to_numpy()
    is_head = frame.video_id.isin(head).to_numpy()
    tab1 = frame.tab.eq(1).to_numpy()
    median = float(activity.median())
    return {
        "all": np.ones(len(frame), dtype=bool),
        "tab=1": tab1,
        "tab!=1": ~tab1,
        "warm_tab=1": tab1 & known_user & known_item,
        "user=unseen": ~known_user,
        "user=low_activity": known_user & (counts <= median),
        "user=high_activity": counts > median,
        "item=head": is_head,
        "item=tail": known_item & ~is_head,
        "item=unseen": ~known_item,
    }


def cohort_summary(frame, train):
    return {
        "rows": len(frame), "users": int(frame.user_id.nunique()),
        "items": int(frame.video_id.nunique()),
        "start_time_ms": int(frame.time_ms.min()), "end_time_ms": int(frame.time_ms.max()),
        "positive_rate": float(frame.is_click.mean()),
        "tab_distribution": {str(k): int(v) for k, v in frame.tab.value_counts().sort_index().items()},
        "train_unknown_user_rate": float((~frame.user_id.isin(train.user_id)).mean()),
        "train_unknown_item_rate": float((~frame.video_id.isin(train.video_id)).mean()),
    }


def exposure_shift(standard, random, train):
    """Descriptive marginal changes; not an identified causal policy effect."""
    output = {}
    for column in ("user_id", "video_id", "tab"):
        a = standard[column].value_counts(normalize=True)
        b = random[column].value_counts(normalize=True)
        support = a.index.union(b.index)
        p = a.reindex(support, fill_value=0).to_numpy(float)
        q = b.reindex(support, fill_value=0).to_numpy(float)
        mid = (p + q) / 2
        def kl(x):
            positive = x > 0
            return float((x[positive] * np.log2(x[positive] / mid[positive])).sum())
        output[column] = {"total_variation": float(np.abs(p-q).sum()/2),
                          "jensen_shannon_bits": (kl(p)+kl(q))/2,
                          "common_values": int(len(a.index.intersection(b.index)))}
    output["training_segments"] = {
        name: {key: int(mask.sum()) for key, mask in training_segments(train, frame).items()}
        for name, frame in (("standard_test", standard), ("random_test", random))
    }
    output["definitions"] = {
        "item_head": "top 20% of training items by training exposure count; ID tie-break",
        "user_low_activity": "positive training exposure count at or below training-user median",
        "training_user_exposure_median": float(train.groupby("user_id").size().median()),
        "tab": "logged scenario ID; tab=1 is a shared-scenario sensitivity, not a causal match",
    }
    return output


def evaluate_predictions(train, cohorts, predictions, task_columns, *, n_bins=10, bootstrap_repeats=500, seed=2026):
    """Evaluate supplied frozen logits; final labels cannot affect fitted maps.

    predictions[model][cohort][target] is a positional 1D logit array for the
    corresponding cohort. Kept separate from scoring for auditable tests.
    """
    n_bins = _integer(n_bins, "calibration_bins")
    bootstrap_repeats = _integer(bootstrap_repeats, "bootstrap_repeats", 0)
    seed = _integer(seed, "seed", 0)
    rows, bins, intervals, calibrators = [], [], [], {}
    segments = {name: training_segments(train, cohorts[name]) for name in TEST_COHORTS}
    for model, tasks in task_columns.items():
        calibrators[model] = {}
        for target, column in tasks.items():
            fitted = {}
            for source in ("standard", "random"):
                name = source + "_calibration"
                logits = predictions[model][name][target]
                fitted[source + "_platt"] = fit_calibrator(logits, cohorts[name][column].to_numpy())
                fitted[source + "_platt"]["source_cohort"] = name
                fitted[source + "_platt"]["target_column"] = column
                fitted[source + "_platt"]["cutoff_time_ms"] = int(cohorts[name].time_ms.max())
            calibrators[model][target] = fitted
            for name in TEST_COHORTS:
                frame = cohorts[name]
                logits = np.asarray(predictions[model][name][target], dtype=float)
                if logits.ndim != 1 or len(logits) != len(frame) or not np.isfinite(logits).all():
                    raise ValueError(f"{model}/{name}/{target}: invalid aligned logit array")
                labels = frame[column].to_numpy()
                variants = {"raw": expit(logits)}
                variants.update({key: apply_calibrator(logits, value) for key, value in fitted.items()})
                for source in ("standard", "random"):
                    calibration_labels = cohorts[source + "_calibration"][column].to_numpy()
                    # Jeffreys smoothing avoids degenerate estimates for sparse heads.
                    probability = (calibration_labels.sum() + 0.5) / (len(calibration_labels) + 1)
                    variants[source + "_constant"] = np.full(len(frame), probability)
                for variant, probabilities in variants.items():
                    masks = segments[name] if target == "click" and "constant" not in variant else {"all": segments[name]["all"]}
                    for segment, mask in masks.items():
                        if not mask.any():
                            continue
                        rows.append({"model": model, "target": target, "cohort": name,
                                     "variant": variant, "segment": segment,
                                     **binary_metrics(labels[mask], probabilities[mask], n_bins=n_bins)})
                    if target == "click" and "constant" not in variant:
                        bins.extend({"model": model, "target": target, "cohort": name, "variant": variant, **entry}
                                    for entry in reliability_bins(labels, probabilities, n_bins=n_bins))
                    if target == "click" and variant in fitted and bootstrap_repeats > 0:
                        intervals.append({"model": model, "cohort": name, "variant": variant,
                                          **paired_cluster_bootstrap(labels, variants["raw"], probabilities,
                                                                     frame.user_id.to_numpy(), repeats=bootstrap_repeats, seed=seed)})
    return {"metrics": rows, "reliability": bins, "uncertainty": intervals, "calibrators": calibrators}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_cohorts(model_name, artifacts_dir, cohorts, users, videos, train_cutoff, batch_size=4096):
    """Restore the existing Week 3 model and retain its exact input contract."""
    batch_size = _integer(batch_size, "batch_size")
    root = Path(artifacts_dir)
    prefix = root / f"week3_{model_name}"
    paths = {suffix: Path(str(prefix) + "_" + suffix) for suffix in ("model.pt", "encoder.json", "results.json")}
    provenance = {str(p): _hash_file(p) for p in paths.values()}
    saved_results = json.loads(paths["results.json"].read_text())
    expected_architecture = "deepfm_mmoe" if model_name == "mmoe" else model_name
    if saved_results["model"] != expected_architecture:
        raise ValueError("Saved Week 3 result architecture does not match the requested model")
    multi = model_name in ("mmoe", "din_mmoe")
    if multi:
        duration = Path(str(prefix) + "_duration_curve.json")
        model, encoder, _, metadata = load_mmoe_artifacts(paths["model.pt"], paths["encoder.json"], duration)
        tasks = metadata["binary_tasks"]
        indices = {task: metadata["task_names"].index(task) for task in tasks}
        provenance[str(duration)] = _hash_file(duration)
    else:
        model, encoder = load_deepfm_artifacts(paths["model.pt"], paths["encoder.json"])
        tasks = {"click": "is_click"}
    features = encoder.categorical + encoder.numeric
    forbidden = set(features) & (POST_EXPOSURE_OR_POLICY_COLUMNS | UNSAFE_AGGREGATE_FEATURES)
    if forbidden:
        raise ValueError(f"Unsafe checkpoint features: {sorted(forbidden)}")
    history = getattr(model, "history_index", None)
    if history is not None:
        if history.cutoff != train_cutoff:
            raise ValueError("Saved DIN history cutoff does not match the current training split")
        checkpoint = torch.load(paths["model.pt"], map_location="cpu", weights_only=True)
        path = paths["model.pt"].parent / checkpoint["history_file"]
        provenance[str(path)] = _hash_file(path)
    predictions, feature_audit = {}, {}
    for name, frame in cohorts.items():
        if frame.time_ms.min() <= train_cutoff:
            raise ValueError("Scored cohorts must be strictly after training")
        attached = _attach_static_features(frame, users, videos, features)
        if len(attached) != len(frame) or not attached[["user_id", "video_id"]].reset_index(drop=True).equals(frame[["user_id", "video_id"]].reset_index(drop=True)):
            raise ValueError("Static joins changed row order or cohort membership")
        cat, num = encoder.transform(attached)
        seq = history.transform(attached) if history is not None else None
        logits = (predict_multi if multi else predict_single)(model, cat, num, batch_size, torch.device("cpu"), seq)
        predictions[name] = {task: logits[:, indices[task]] if multi else logits for task in tasks}
        feature_audit[name] = {
            column: {"unknown_rows": int((cat[:, i] == 0).sum()), "missing_rows": int((cat[:, i] == 1).sum())}
            for i, column in enumerate(encoder.categorical)
        }
    # Reproduce the old validation metrics before accepting the checkpoint and
    # current data/feature combination. Legacy runs did not store input hashes.
    prior = saved_results["pointwise"]["validation"]
    if multi:
        prior = prior["binary"]["click"]
    current = binary_metrics(cohorts["standard_calibration"].is_click.to_numpy(),
                             expit(predictions["standard_calibration"]["click"].astype(float)))
    if current["rows"] != prior["examples"]:
        raise ValueError("Saved Week 3 validation cohort size does not match the current split")
    deltas = {}
    for key in ("log_loss", "roc_auc", "pr_auc"):
        if prior[key] is not None and current[key] is not None:
            deltas[key] = current[key] - prior[key]
            if abs(deltas[key]) > 1e-5:
                raise ValueError(f"{model_name}: saved validation {key} does not reproduce; verify data and checkpoint")
    return predictions, tasks, {"sha256": provenance, "features": features, "feature_oov": feature_audit,
                                "validation_reproduction_deltas": deltas}


def run_week4_audit(config):
    started = time.perf_counter()
    model_names = config.get("models", list(MODEL_NAMES))
    if not model_names or len(set(model_names)) != len(model_names) or set(model_names).difference(MODEL_NAMES):
        raise ValueError(f"models must be unique members of {MODEL_NAMES}")
    evaluation = config.get("evaluation", {})
    batch_size = _integer(evaluation.get("batch_size", 4096), "batch_size")
    n_bins = _integer(evaluation.get("calibration_bins", 10), "calibration_bins")
    repeats = _integer(evaluation.get("bootstrap_repeats", 500), "bootstrap_repeats", 0)
    seed = _integer(config.get("seed", 2026), "seed", 0)
    if not isinstance(config.get("save_logits", True), bool):
        raise ValueError("save_logits must be a boolean")
    input_paths = {name: Path(config["data"]["processed_dir"]) / f"{name}.csv.gz"
                   for name in ("train", "validation", "test", "random_audit")}
    for filename in (USER_FEATURE_FILE, VIDEO_FEATURE_FILE):
        input_paths[filename] = _find_unique_file(Path(config["data"]["raw_dir"]), filename)
    input_hashes = {name: _hash_file(path) for name, path in input_paths.items()}
    splits = load_prepared(config["data"]["processed_dir"])
    cohorts, protocol = split_audit_cohorts(splits)
    users, videos = load_kuairand_features(config["data"]["raw_dir"])
    predictions, tasks, provenance = {}, {}, {}
    output = Path(config.get("artifacts_dir", "artifacts"))
    output.mkdir(parents=True, exist_ok=True)
    for model in model_names:
        print(f"Week 4: scoring frozen {model} on {sum(len(f) for f in cohorts.values()):,} impressions", flush=True)
        predictions[model], tasks[model], provenance[model] = score_cohorts(
            model, config.get("checkpoints_dir", "artifacts"), cohorts, users, videos,
            protocol["train_cutoff_ms"], batch_size,
        )
        if config.get("save_logits", True):
            np.savez_compressed(output / f"week4_{model}_logits.npz",
                                **{f"{name}__{target}": values for name, scores in predictions[model].items() for target, values in scores.items()})
    print("Week 4: fitting calibrators and evaluating held-out exposures", flush=True)
    results = evaluate_predictions(
        splits.train, cohorts, predictions, tasks,
        n_bins=n_bins, bootstrap_repeats=repeats, seed=seed,
    )
    if any(_hash_file(path) != input_hashes[name] for name, path in input_paths.items()):
        raise ValueError("Input data changed during the audit; rerun with frozen inputs")
    if any(_hash_file(path) != digest for entry in provenance.values() for path, digest in entry["sha256"].items()):
        raise ValueError("Checkpoint artifacts changed during the audit")
    results.update({
        "schema_version": 1,
        "protocol": protocol,
        "cohorts": {name: cohort_summary(frame, splits.train) for name, frame in {"train": splits.train, **cohorts}.items()},
        "exposure_shift": exposure_shift(cohorts["standard_test"], cohorts["random_test"], splits.train),
        "model_provenance": provenance,
        "target_columns": tasks,
        "data_sha256": input_hashes,
        "configuration": config,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch_threads": torch.get_num_threads(),
                    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "torch", "scipy", "scikit-learn")}},
        "elapsed_seconds": time.perf_counter() - started,
    })
    # Save cohort membership in positional order for exact alignment with NPZ scores.
    membership = pd.concat([frame[["user_id", "video_id", "time_ms", "is_click", "tab"]].assign(cohort=name, row_index=np.arange(len(frame)))
                            for name, frame in cohorts.items()], ignore_index=True)
    membership.to_csv(output / "week4_cohort_membership.csv.gz", index=False)
    (output / "week4_results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (output / "week4_calibrators.json").write_text(json.dumps(results["calibrators"], indent=2, allow_nan=False) + "\n")
    pd.DataFrame(results["metrics"]).to_csv(output / "week4_metrics.csv", index=False)
    pd.DataFrame(results["reliability"]).to_csv(output / "week4_reliability.csv", index=False)
    from kuaiflow.week4_report import generate_report
    generate_report(results, Path(config.get("report_root", ".")))
    return results
