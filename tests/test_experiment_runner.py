from pathlib import Path

from pre_snap_motion.experiment_runner import available_config_paths


def test_available_config_paths_finds_yaml_configs() -> None:
    config_names = [path.name for path in available_config_paths()]

    assert "default.yaml" in config_names
    assert "quickstart.yaml" in config_names
    assert "tracking_experiment.yaml" in config_names
