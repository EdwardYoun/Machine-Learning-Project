from pathlib import Path

import pandas as pd

from pre_snap_motion.experiment_runner import (
    _read_csv_if_present,
    available_config_paths,
    compare_configs,
)


def test_available_config_paths_finds_yaml_configs() -> None:
    config_names = [path.name for path in available_config_paths()]

    assert "default.yaml" in config_names
    assert "quickstart.yaml" in config_names
    assert "tracking_experiment.yaml" in config_names
    assert "motion_value_v2.yaml" in config_names
    assert "motion_value_v2_inference.yaml" in config_names
    assert "motion_value_v2_no_calibration.yaml" in config_names
    assert "motion_value_v2_offense_only.yaml" in config_names


def test_compare_configs_writes_summary_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        "\n".join(
            [
                'project_name: "compare-project"',
                "paths:",
                f'  artifacts_dir: "{(tmp_path / "artifacts").as_posix()}"',
                "comparison:",
                '  primary_target: "completion"',
            ]
        ),
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "artifacts" / "compare-project" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "completion",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "balanced_accuracy": 0.62,
            },
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "success",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "balanced_accuracy": 0.58,
            },
        ]
    ).to_csv(metrics_dir / "selected_models.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_split": "test",
                "target": "completion",
                "adjusted_effect": 0.08,
                "effect_direction": "helps",
                "effect_ci_lower": 0.01,
                "effect_ci_upper": 0.15,
            },
            {
                "dataset_split": "test",
                "target": "success",
                "adjusted_effect": -0.02,
                "effect_direction": "hurts",
                "effect_ci_lower": -0.04,
                "effect_ci_upper": 0.0,
            },
        ]
    ).to_csv(metrics_dir / "motion_effect_overall.csv", index=False)
    pd.DataFrame(
        [{"dataset_split": "test", "response_column": "tracking_skill_separation_gain", "adjusted_effect": 0.05, "tracking_is_sparse": False}]
    ).to_csv(metrics_dir / "defensive_reaction_overall.csv", index=False)
    (metrics_dir / "dataset_summary.json").write_text(
        '{"test_tracking_coverage_rate": 0.4}',
        encoding="utf-8",
    )

    outputs = compare_configs([config_path])
    comparison = pd.read_csv(outputs["experiment_comparison_csv"])

    assert outputs["experiment_comparison_csv"].exists()
    assert outputs["experiment_comparison_md"].exists()
    assert comparison.loc[0, "rank"] == 1
    assert comparison.loc[0, "primary_target"] == "completion"
    assert comparison.loc[0, "motion_help_targets"] == 1


def test_read_csv_if_present_handles_empty_csv(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    frame = _read_csv_if_present(empty_csv)

    assert frame.empty


def test_compare_configs_keeps_rows_when_motion_effect_is_empty(tmp_path: Path) -> None:
    config_path = tmp_path / "compare_empty_motion.yaml"
    config_path.write_text(
        "\n".join(
            [
                'project_name: "compare-empty-motion"',
                "paths:",
                f'  artifacts_dir: "{(tmp_path / "artifacts").as_posix()}"',
                "comparison:",
                '  primary_target: "completion"',
            ]
        ),
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "artifacts" / "compare-empty-motion" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "completion",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "balanced_accuracy": 0.61,
            }
        ]
    ).to_csv(metrics_dir / "selected_models.csv", index=False)
    (metrics_dir / "motion_effect_overall.csv").write_text("", encoding="utf-8")

    outputs = compare_configs([config_path])
    comparison = pd.read_csv(outputs["experiment_comparison_csv"])

    assert len(comparison) == 1
    assert comparison.loc[0, "project_name"] == "compare-empty-motion"
