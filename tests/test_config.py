from pathlib import Path

from pre_snap_motion.config import load_config
from pre_snap_motion.data.features import processed_dataset_path
from pre_snap_motion.io import project_artifacts_dir


def test_load_config_reads_quickstart_values() -> None:
    config = load_config(Path("configs/quickstart.yaml"))

    assert config.project_name == "reading-the-defense-quickstart"
    assert config.data.pbp_seasons == [2024, 2025]
    assert config.split.test_seasons == [2025]
    assert config.targets.classification_targets == ["success", "explosive", "completion"]
    assert config.targets.regression_targets == ["epa"]
    assert config.targets.explosive_threshold == 20
    assert (
        processed_dataset_path(config).name
        == "reading-the-defense-quickstart_passing_motion_modeling_dataset.parquet"
    )
    assert project_artifacts_dir(config).as_posix().endswith(
        "artifacts/reading-the-defense-quickstart"
    )


def test_load_config_reads_v2_experiment_values() -> None:
    config = load_config(Path("configs/motion_value_v2.yaml"))

    assert config.experiment.mode == "balanced_research"
    assert config.split.strategy == "rolling_origin"
    assert config.split.rolling_min_train_seasons == 2
    assert config.experiment.feature_sets == [
        "context_only",
        "context_plus_motion",
        "full",
    ]
