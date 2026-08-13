"""Command-line entry points for the first project week."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from kuaiflow.benchmark import run_week1_benchmark, save_benchmark_results
from kuaiflow.data import load_prepared, prepare_week1_data
from kuaiflow.download import download_kuairand_pure
from kuaiflow.toy import make_toy_splits


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kuaiflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("download", "prepare", "benchmark"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", default="configs/week1.yaml")
    benchmark = subparsers.choices["benchmark"]
    benchmark.add_argument(
        "--models", nargs="+", choices=["popularity", "itemcf", "bpr"]
    )

    subparsers.add_parser("demo")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "demo":
        config = {
            "seed": 2026,
            "data": {"positive_column": "is_click"},
            "evaluation": {"k": 2, "max_users": None},
            "models": {
                "itemcf": {"neighbor_k": 3},
                "bpr": {
                    "factors": 8,
                    "learning_rate": 0.05,
                    "regularization": 0.0001,
                    "epochs": 10,
                    "batch_size": 4,
                },
            },
        }
        print(json.dumps(run_week1_benchmark(make_toy_splits(), config), indent=2))
        return

    config = _load_config(args.config)
    if args.command == "download":
        path = download_kuairand_pure(config["data"]["raw_dir"])
        print(f"Dataset extracted under {path}")
    elif args.command == "prepare":
        summary = prepare_week1_data(
            raw_dir=config["data"]["raw_dir"],
            processed_dir=config["data"]["processed_dir"],
            validation_fraction=config["data"].get(
                "validation_fraction_of_future", 0.5
            ),
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "benchmark":
        results = run_week1_benchmark(
            load_prepared(config["data"]["processed_dir"]),
            config,
            selected_models=args.models,
        )
        save_benchmark_results(results, config.get("artifacts_dir", "artifacts"))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

