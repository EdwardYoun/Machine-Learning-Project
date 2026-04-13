from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from pre_snap_motion.config import ProjectConfig, load_config
from pre_snap_motion.data.tracking import (
    load_tracking_play_features,
    resolve_tracking_input_paths,
    summarize_tracking_play_features,
    tracking_cache_is_stale,
    tracking_features_path,
)
from pre_snap_motion.io import project_artifacts_dir
from pre_snap_motion.pipeline import fetch, prepare, run, train

COMMAND_HANDLERS: dict[str, Callable[[ProjectConfig], object]] = {
    "fetch": fetch,
    "prepare": prepare,
    "train": train,
    "run": run,
}


def available_config_paths(config_dir: str | Path = "configs") -> list[Path]:
    return sorted(Path(config_dir).glob("*.yaml"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Centralized runner for pre-snap motion experiments."
    )
    parser.add_argument(
        "--command",
        choices=["inspect", "fetch", "prepare", "train", "run"],
        default="run",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Path to a YAML config file. Repeat to run multiple configs.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run every YAML config found in the configs directory.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip printing metric summaries after train or run commands.",
    )
    return parser


def _config_paths_from_args(args: argparse.Namespace) -> list[Path]:
    if args.all_configs:
        paths = available_config_paths()
        if paths:
            return paths
    if args.config:
        return [Path(path) for path in args.config]
    return [Path("configs/default.yaml")]


def _print_header(config_path: Path, config: ProjectConfig, command: str) -> None:
    print("=" * 72)
    print(f"Project: {config.project_name}")
    print(f"Config: {config_path}")
    print(f"Command: {command}")


def _print_tracking_status(config: ProjectConfig) -> None:
    if not config.tracking.enabled:
        print("Tracking: disabled")
        return

    input_paths = resolve_tracking_input_paths(config)
    cache_path = tracking_features_path(config)
    print(f"Tracking inputs discovered: {len(input_paths)}")
    if input_paths:
        sample_paths = [str(path) for path in input_paths[:3]]
        print(f"Tracking input sample: {sample_paths}")
        if len(input_paths) > 3:
            print(f"Tracking input sample truncated: +{len(input_paths) - 3} more")
    print(f"Tracking cache: {cache_path}")
    print(f"Tracking cache stale: {tracking_cache_is_stale(config)}")

    if not input_paths:
        return

    tracking = load_tracking_play_features(config)
    if tracking is None:
        return
    summary = summarize_tracking_play_features(tracking)
    if summary.empty:
        return
    print("Tracking coverage by inferred NFL season:")
    for _, row in summary.iterrows():
        print(
            f"  - season {int(row['season'])}: "
            f"{int(row['unique_games'])} games, {int(row['unique_plays'])} plays"
        )


def _print_outputs(outputs: object) -> None:
    if isinstance(outputs, dict):
        print("Outputs:")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")
        return
    print(f"Output: {outputs}")


def _print_metric_summary(config: ProjectConfig) -> None:
    metrics_dir = project_artifacts_dir(config) / "metrics"
    dataset_summary_path = metrics_dir / "dataset_summary.json"
    best_models_path = metrics_dir / "best_models.csv"

    if dataset_summary_path.exists():
        payload = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        print("Dataset summary:")
        print(
            f"  - rows: train {payload.get('train_rows', 0):,}, "
            f"test {payload.get('test_rows', 0):,}, total {payload.get('total_rows', 0):,}"
        )
        if "train_tracking_coverage_rate" in payload:
            print(
                "  - tracking coverage: "
                f"train {payload['train_tracking_coverage_rate']:.1%}, "
                f"test {payload.get('test_tracking_coverage_rate', 0.0):.1%}"
            )
        elif "tracking_coverage_rate" in payload:
            print(f"  - tracking coverage: {payload['tracking_coverage_rate']:.1%}")

    if best_models_path.exists():
        best_models = pd.read_csv(best_models_path)
        if best_models.empty:
            return
        print("Best models:")
        columns = [
            column
            for column in [
                "evaluation_slice",
                "task",
                "target",
                "model_name",
                "feature_set",
                "selection_metric",
            ]
            if column in best_models.columns
        ]
        for _, row in best_models.loc[:, columns].iterrows():
            slice_prefix = (
                f"[{row['evaluation_slice']}] "
                if "evaluation_slice" in row and pd.notna(row["evaluation_slice"])
                else ""
            )
            print(
                f"  - {slice_prefix}{row['task']} / {row['target']}: "
                f"{row['model_name']} with {row['feature_set']}"
            )


def run_config(
    config_path: Path,
    command: str,
    skip_summary: bool,
) -> None:
    config = load_config(config_path)
    _print_header(config_path, config, command)
    _print_tracking_status(config)

    if command == "inspect":
        return

    outputs = COMMAND_HANDLERS[command](config)
    _print_outputs(outputs)
    if command in {"train", "run"} and not skip_summary:
        _print_metric_summary(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for config_path in _config_paths_from_args(args):
        run_config(
            config_path=config_path,
            command=args.command,
            skip_summary=args.skip_summary,
        )


if __name__ == "__main__":
    main()
