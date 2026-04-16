from __future__ import annotations

import argparse
import json
from math import inf
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.errors import EmptyDataError

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


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


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
        best_models = _read_csv_if_present(best_models_path)
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
        selected_models = _read_csv_if_present(selected_models_path)
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
        motion_effect = _read_csv_if_present(motion_effect_path)
        if "dataset_split" in motion_effect.columns:
            motion_effect = motion_effect.loc[motion_effect["dataset_split"] == "test"]
        if not motion_effect.empty:
            print("Motion effect:")
            for _, row in motion_effect.iterrows():
                print(
                    f"  - {row['target']}: {row['effect_direction']} "
                    f"({row['adjusted_effect']:.4f}, CI {row['effect_ci_lower']:.4f} to {row['effect_ci_upper']:.4f})"
                )

    if defensive_reaction_path.exists():
        defensive_reaction = _read_csv_if_present(defensive_reaction_path)
        if "dataset_split" in defensive_reaction.columns:
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


def _preferred_selected_row(
    selected_models: pd.DataFrame,
    target_name: str,
) -> pd.Series | None:
    target_rows = selected_models.loc[selected_models["target"] == target_name]
    if target_rows.empty:
        return None
    if "evaluation_slice" in target_rows.columns:
        all_rows = target_rows.loc[target_rows["evaluation_slice"] == "all"]
        if not all_rows.empty:
            return all_rows.iloc[0]
    return target_rows.iloc[0]


def _comparison_metric_name(config: ProjectConfig, task: str) -> str:
    if task == "classification":
        return config.comparison.classification_metric
    return config.comparison.regression_metric


def _comparison_metric_value(
    selected_models: pd.DataFrame,
    config: ProjectConfig,
    target_name: str,
) -> tuple[str | None, float | None]:
    selected_row = _preferred_selected_row(selected_models, target_name)
    if selected_row is None:
        return None, None
    metric_name = _comparison_metric_name(config, selected_row["task"])
    metric_value = selected_row.get(metric_name)
    if pd.isna(metric_value):
        return metric_name, None
    return metric_name, float(metric_value)


def _metric_sort_value(
    metric_name: str | None,
    metric_value: float | None,
) -> float:
    if metric_name is None or metric_value is None:
        return -inf
    if metric_name in {"rmse", "mae", "log_loss", "brier_score", "expected_calibration_error"}:
        return -metric_value
    return metric_value


def _effect_counts(motion_effect: pd.DataFrame, rank_targets: list[str]) -> dict[str, int]:
    if "target" not in motion_effect.columns or "effect_direction" not in motion_effect.columns:
        return {"helps": 0, "hurts": 0, "unclear": 0}
    target_effects = motion_effect.loc[motion_effect["target"].isin(rank_targets)]
    if target_effects.empty:
        return {"helps": 0, "hurts": 0, "unclear": 0}
    direction_counts = target_effects["effect_direction"].value_counts()
    return {
        "helps": int(direction_counts.get("helps", 0)),
        "hurts": int(direction_counts.get("hurts", 0)),
        "unclear": int(direction_counts.get("unclear", 0)),
    }


def _comparison_row(
    config_path: Path,
    config: ProjectConfig,
    selected_models: pd.DataFrame,
    motion_effect: pd.DataFrame,
    defensive_reaction: pd.DataFrame,
    dataset_summary: dict[str, object],
) -> dict[str, object]:
    primary_metric_name, primary_metric_value = _comparison_metric_value(
        selected_models,
        config,
        config.comparison.primary_target,
    )
    effect_counts = _effect_counts(motion_effect, config.comparison.rank_targets)
    row: dict[str, object] = {
        "project_name": config.project_name,
        "config_path": str(config_path),
        "primary_target": config.comparison.primary_target,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "motion_help_targets": effect_counts["helps"],
        "motion_hurt_targets": effect_counts["hurts"],
        "motion_unclear_targets": effect_counts["unclear"],
        "selected_models": int(len(selected_models)),
    }
    if "test_tracking_coverage_rate" in dataset_summary:
        row["test_tracking_coverage_rate"] = dataset_summary["test_tracking_coverage_rate"]

    for target_name in config.comparison.rank_targets:
        metric_name, metric_value = _comparison_metric_value(
            selected_models,
            config,
            target_name,
        )
        if metric_name is not None:
            row[f"{target_name}_metric"] = metric_name
        if metric_value is not None:
            row[f"{target_name}_value"] = metric_value
        if "target" not in motion_effect.columns:
            target_motion = pd.DataFrame()
        else:
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
    return row


def _rank_comparison_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return rows
    ranked = sorted(
        rows,
        key=lambda row: (
            -_metric_sort_value(
                row.get("primary_metric_name"),
                row.get("primary_metric_value"),
            ),
            -int(row.get("motion_help_targets", 0)),
            int(row.get("motion_hurt_targets", 0)),
            int(row.get("motion_unclear_targets", 0)),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _comparison_markdown(ranked_rows: list[dict[str, object]]) -> str:
    markdown_lines = ["# Experiment Comparison", ""]
    if not ranked_rows:
        markdown_lines.append("No completed experiment outputs were available to compare.")
        return "\n".join(markdown_lines).rstrip() + "\n"

    top_row = ranked_rows[0]
    headline = f"`{top_row['project_name']}` ranks first"
    if top_row.get("primary_metric_name") and top_row.get("primary_metric_value") is not None:
        headline += (
            f" on `{top_row['primary_target']}` via "
            f"{top_row['primary_metric_name']}={top_row['primary_metric_value']:.4f}"
        )
    markdown_lines.append(headline + ".")
    markdown_lines.append("")

    for row in ranked_rows:
        markdown_lines.append(f"## {int(row['rank'])}. {row['project_name']}")
        markdown_lines.append(f"- Config: `{row['config_path']}`")
        if row.get("primary_metric_name") and row.get("primary_metric_value") is not None:
            markdown_lines.append(
                f"- Primary target `{row['primary_target']}`: "
                f"{row['primary_metric_name']}={row['primary_metric_value']:.4f}"
            )
        markdown_lines.append(
            "- Motion directions across ranked targets: "
            f"{int(row.get('motion_help_targets', 0))} help, "
            f"{int(row.get('motion_hurt_targets', 0))} hurt, "
            f"{int(row.get('motion_unclear_targets', 0))} unclear"
        )
        if "test_tracking_coverage_rate" in row:
            markdown_lines.append(
                f"- Test tracking coverage: {float(row['test_tracking_coverage_rate']):.1%}"
            )
        if "reportable_defensive_reactions" in row:
            reportable_count = int(row["reportable_defensive_reactions"])
            if reportable_count == 0:
                markdown_lines.append(
                    "- Defensive response: directional only because reportable tracking coverage is sparse."
                )
            else:
                markdown_lines.append(
                    f"- Reportable defensive reactions: {reportable_count}"
                )
                markdown_lines.append(
                    f"- Top defensive response: {row['top_defensive_response']} "
                    f"({float(row['top_defensive_response_effect']):.4f})"
                )
        markdown_lines.append("- Ranked targets:")
        for target_name in [
            "completion",
            "explosive",
            "success",
            "epa",
        ]:
            metric_name = row.get(f"{target_name}_metric")
            metric_value = row.get(f"{target_name}_value")
            effect_direction = row.get(f"{target_name}_direction")
            effect_value = row.get(f"{target_name}_effect")
            if metric_name is None and effect_direction is None:
                continue
            target_line = f"  - {target_name}:"
            details: list[str] = []
            if metric_name is not None and metric_value is not None:
                details.append(f"{metric_name}={float(metric_value):.4f}")
            if effect_direction is not None and effect_value is not None:
                details.append(f"motion {effect_direction} ({float(effect_value):.4f})")
            markdown_lines.append(target_line + " " + ", ".join(details))
        markdown_lines.append("")
    return "\n".join(markdown_lines).rstrip() + "\n"


def compare_configs(config_paths: list[Path]) -> dict[str, Path]:
    rows: list[dict[str, object]] = []

    for config_path in config_paths:
        config = load_config(config_path)
        metrics_dir = project_artifacts_dir(config) / "metrics"
        selected_models_path = metrics_dir / "selected_models.csv"
        motion_effect_path = metrics_dir / "motion_effect_overall.csv"
        defensive_reaction_path = metrics_dir / "defensive_reaction_overall.csv"
        dataset_summary_path = metrics_dir / "dataset_summary.json"
        if not selected_models_path.exists() or not motion_effect_path.exists():
            continue

        selected_models = _read_csv_if_present(selected_models_path)
        motion_effect = _read_csv_if_present(motion_effect_path)
        defensive_reaction = _read_csv_if_present(defensive_reaction_path)
        dataset_summary = (
            json.loads(dataset_summary_path.read_text(encoding="utf-8"))
            if dataset_summary_path.exists()
            else {}
        )
        if selected_models.empty:
            continue
        if "dataset_split" in motion_effect.columns:
            motion_effect = motion_effect.loc[motion_effect["dataset_split"] == "test"]
        if not defensive_reaction.empty and "dataset_split" in defensive_reaction.columns:
            defensive_reaction = defensive_reaction.loc[
                defensive_reaction["dataset_split"] == "test"
            ]
        rows.append(
            _comparison_row(
                config_path,
                config,
                selected_models,
                motion_effect,
                defensive_reaction,
                dataset_summary,
            )
        )

    ranked_rows = _rank_comparison_rows(rows)
    comparison_dir = Path("artifacts") / "experiment_comparisons"
    csv_path = write_frame(
        pd.DataFrame(ranked_rows), comparison_dir / "experiment_comparison.csv"
    )
    md_path = write_text(
        _comparison_markdown(ranked_rows),
        comparison_dir / "experiment_comparison.md",
    )
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
