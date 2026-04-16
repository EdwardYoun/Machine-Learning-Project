from pathlib import Path

import pandas as pd

from pre_snap_motion.experiment_runner import available_config_paths, compare_configs


def test_available_config_paths_finds_yaml_configs() -> None:
    config_names = [path.name for path in available_config_paths()]

    assert "default.yaml" in config_names
    assert "quickstart.yaml" in config_names
    assert "tracking_experiment.yaml" in config_names
    assert "motion_value_v2.yaml" in config_names


def test_compare_configs_writes_summary_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        "\n".join(
            [
                'project_name: "compare-project"',
                "paths:",
                f'  artifacts_dir: "{(tmp_path / "artifacts").as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "artifacts" / "compare-project" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"evaluation_slice": "all", "task": "classification", "target": "success", "model_name": "logistic_regression", "feature_set": "full"}]
    ).to_csv(metrics_dir / "selected_models.csv", index=False)
    pd.DataFrame(
        [{"dataset_split": "test", "target": "success", "adjusted_effect": 0.1, "effect_direction": "helps", "effect_ci_lower": 0.01, "effect_ci_upper": 0.2}]
    ).to_csv(metrics_dir / "motion_effect_overall.csv", index=False)
    pd.DataFrame(
        [{"dataset_split": "test", "response_column": "tracking_skill_separation_gain", "adjusted_effect": 0.05, "tracking_is_sparse": False}]
    ).to_csv(metrics_dir / "defensive_reaction_overall.csv", index=False)

    outputs = compare_configs([config_path])

    assert outputs["experiment_comparison_csv"].exists()
    assert outputs["experiment_comparison_md"].exists()
