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
from pre_snap_motion.io import project_artifacts_dir, write_frame, write_text
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
        choices=["inspect", "compare", "fetch", "prepare", "train", "run"],
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
    selected_models_path = metrics_dir / "selected_models.csv"
    motion_effect_path = metrics_dir / "motion_effect_overall.csv"
    defensive_reaction_path = metrics_dir / "defensive_reaction_overall.csv"

    if dataset_summary_path.exists():
        payload = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        print("Dataset summary:")
        print(
            f"  - rows: train {payload.get('train_rows', 0):,}, "
            f"validation {payload.get('validation_rows', 0):,}, "
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

    if selected_models_path.exists():
        selected_models = pd.read_csv(selected_models_path)
        if not selected_models.empty:
            print("Selected holdout models:")
            columns = [
                column
                for column in [
                    "evaluation_slice",
                    "task",
                    "target",
                    "model_name",
                    "feature_set",
                    "selected_threshold",
                ]
                if column in selected_models.columns
            ]
            for _, row in selected_models.loc[:, columns].iterrows():
                slice_prefix = (
                    f"[{row['evaluation_slice']}] "
                    if "evaluation_slice" in row and pd.notna(row["evaluation_slice"])
                    else ""
                )
                threshold_suffix = (
                    f", threshold {row['selected_threshold']:.2f}"
                    if "selected_threshold" in row and pd.notna(row["selected_threshold"])
                    else ""
                )
                print(
                    f"  - {slice_prefix}{row['task']} / {row['target']}: "
                    f"{row['model_name']} with {row['feature_set']}{threshold_suffix}"
                )

    if motion_effect_path.exists():
        motion_effect = pd.read_csv(motion_effect_path)
        motion_effect = motion_effect.loc[motion_effect["dataset_split"] == "test"]
        if not motion_effect.empty:
            print("Motion effect:")
            for _, row in motion_effect.iterrows():
                print(
                    f"  - {row['target']}: {row['effect_direction']} "
                    f"({row['adjusted_effect']:.4f}, CI {row['effect_ci_lower']:.4f} to {row['effect_ci_upper']:.4f})"
                )

    if defensive_reaction_path.exists():
        defensive_reaction = pd.read_csv(defensive_reaction_path)
        defensive_reaction = defensive_reaction.loc[
            defensive_reaction["dataset_split"] == "test"
        ]
        if not defensive_reaction.empty:
            sparse_mask = (
                defensive_reaction["tracking_is_sparse"].astype(bool)
                if "tracking_is_sparse" in defensive_reaction.columns
                else pd.Series(False, index=defensive_reaction.index)
            )
            reportable = defensive_reaction.loc[~sparse_mask]
            if reportable.empty:
                print("Defensive response highlights:")
                print("  - test tracking coverage is sparse, so defensive reaction outputs are directional only")
                return
            print("Defensive response highlights:")
            top_rows = reportable.reindex(
                reportable["adjusted_effect"].abs().sort_values(ascending=False).index
            ).head(3)
            for _, row in top_rows.iterrows():
                sparse_suffix = " (directional)" if row.get("tracking_is_sparse") else ""
                print(
                    f"  - {row['response_column']}: {row['adjusted_effect']:.4f} "
                    f"CI {row['effect_ci_lower']:.4f} to {row['effect_ci_upper']:.4f}{sparse_suffix}"
                )


def compare_configs(config_paths: list[Path]) -> dict[str, Path]:
    rows: list[dict[str, object]] = []
    markdown_lines = ["# Experiment Comparison", ""]

    for config_path in config_paths:
        config = load_config(config_path)
        metrics_dir = project_artifacts_dir(config) / "metrics"
        selected_models_path = metrics_dir / "selected_models.csv"
        motion_effect_path = metrics_dir / "motion_effect_overall.csv"
        defensive_reaction_path = metrics_dir / "defensive_reaction_overall.csv"
        if not selected_models_path.exists() or not motion_effect_path.exists():
            continue

        selected_models = pd.read_csv(selected_models_path)
        motion_effect = pd.read_csv(motion_effect_path)
        defensive_reaction = pd.read_csv(defensive_reaction_path) if defensive_reaction_path.exists() else pd.DataFrame()
        motion_effect = motion_effect.loc[motion_effect["dataset_split"] == "test"]
        if not defensive_reaction.empty and "dataset_split" in defensive_reaction.columns:
            defensive_reaction = defensive_reaction.loc[
                defensive_reaction["dataset_split"] == "test"
            ]

        row: dict[str, object] = {
            "project_name": config.project_name,
            "config_path": str(config_path),
            "selected_models": int(len(selected_models)),
        }
        for target_name in ["success", "explosive", "completion", "epa"]:
            target_motion = motion_effect.loc[motion_effect["target"] == target_name]
            if not target_motion.empty:
                effect_row = target_motion.iloc[0]
                row[f"{target_name}_effect"] = effect_row["adjusted_effect"]
                row[f"{target_name}_direction"] = effect_row["effect_direction"]
        if not defensive_reaction.empty:
            sparse_mask = (
                defensive_reaction["tracking_is_sparse"].astype(bool)
                if "tracking_is_sparse" in defensive_reaction.columns
                else pd.Series(False, index=defensive_reaction.index)
            )
            reportable = defensive_reaction.loc[~sparse_mask]
            row["reportable_defensive_reactions"] = int(len(reportable))
            if not reportable.empty:
                top_row = reportable.reindex(
                    reportable["adjusted_effect"].abs().sort_values(ascending=False).index
                ).iloc[0]
                row["top_defensive_response"] = top_row["response_column"]
                row["top_defensive_response_effect"] = top_row["adjusted_effect"]
        rows.append(row)

        markdown_lines.append(f"## {config.project_name}")
        markdown_lines.append(f"- Config: `{config_path}`")
        markdown_lines.append(f"- Selected holdout models: {len(selected_models)}")
        for target_name in ["success", "explosive", "completion", "epa"]:
            target_motion = motion_effect.loc[motion_effect["target"] == target_name]
            if target_motion.empty:
                continue
            effect_row = target_motion.iloc[0]
            markdown_lines.append(
                f"- {target_name}: {effect_row['effect_direction']} "
                f"({effect_row['adjusted_effect']:.4f}, CI {effect_row['effect_ci_lower']:.4f} to {effect_row['effect_ci_upper']:.4f})"
            )
        if not defensive_reaction.empty:
            reportable = defensive_reaction.loc[
                ~defensive_reaction.get("tracking_is_sparse", False).astype(bool)
            ]
            if reportable.empty:
                markdown_lines.append(
                    "- Defensive reaction: directional only because tracking coverage is sparse."
                )
            else:
                top_row = reportable.reindex(
                    reportable["adjusted_effect"].abs().sort_values(ascending=False).index
                ).iloc[0]
                markdown_lines.append(
                    f"- Top reportable defensive response: {top_row['response_column']} "
                    f"({top_row['adjusted_effect']:.4f})"
                )
        markdown_lines.append("")

    comparison_dir = Path("artifacts") / "experiment_comparisons"
    csv_path = write_frame(pd.DataFrame(rows), comparison_dir / "experiment_comparison.csv")
    md_path = write_text("\n".join(markdown_lines).rstrip() + "\n", comparison_dir / "experiment_comparison.md")
    return {
        "experiment_comparison_csv": csv_path,
        "experiment_comparison_md": md_path,
    }


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
    if command == "compare":
        outputs = compare_configs([config_path])
        _print_outputs(outputs)
        return

    outputs = COMMAND_HANDLERS[command](config)
    _print_outputs(outputs)
    if command in {"train", "run"} and not skip_summary:
        _print_metric_summary(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_paths = _config_paths_from_args(args)
    if args.command == "compare":
        outputs = compare_configs(config_paths)
        _print_outputs(outputs)
        return

    for config_path in config_paths:
        run_config(
            config_path=config_path,
            command=args.command,
            skip_summary=args.skip_summary,
        )


if __name__ == "__main__":
    main()
