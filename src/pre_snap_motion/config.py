from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _value(source: dict[str, Any], key: str, default: Any) -> Any:
    return source[key] if key in source else default


def _legacy_or_default(
    source: dict[str, Any],
    plural_key: str,
    singular_key: str,
    default: list[str],
) -> list[str]:
    if plural_key in source:
        return source[plural_key]
    if singular_key in source:
        return [source[singular_key]]
    return default


@dataclass(slots=True)
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"


@dataclass(slots=True)
class DataConfig:
    pbp_seasons: list[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    ftn_seasons: list[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])
    use_ftn_charting: bool = True
    season_type: str = "REG"


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool = False
    input_globs: list[str] = field(default_factory=lambda: ["train/input_*.csv"])
    input_files: list[str] = field(default_factory=lambda: ["test_input.csv"])
    cache_file: str = "tracking_play_features_v2.parquet"


@dataclass(slots=True)
class FeaturesConfig:
    base_numeric: list[str] = field(
        default_factory=lambda: [
            "down",
            "ydstogo",
            "yardline_100",
            "game_seconds_remaining",
            "half_seconds_remaining",
            "score_differential",
            "posteam_timeouts_remaining",
            "defteam_timeouts_remaining",
        ]
    )
    base_categorical: list[str] = field(
        default_factory=lambda: [
            "posteam",
            "defteam",
            "posteam_type",
            "shotgun",
            "no_huddle",
            "goal_to_go",
            "game_half",
            "roof",
            "surface",
            "div_game",
            "season",
            "qtr",
        ]
    )
    ftn_numeric: list[str] = field(
        default_factory=lambda: [
            "n_blitzers",
            "n_pass_rushers",
            "n_offense_backfield",
            "n_defense_box",
        ]
    )
    ftn_categorical: list[str] = field(
        default_factory=lambda: [
            "is_motion",
            "is_no_huddle",
            "is_play_action",
            "is_screen_pass",
            "is_rpo",
            "is_trick_play",
            "qb_location",
            "starting_hash",
        ]
    )
    tracking_numeric: list[str] = field(
        default_factory=lambda: [
            "tracking_frames",
            "tracking_players",
            "tracking_offense_players",
            "tracking_defense_players",
            "tracking_wr_players",
            "tracking_te_players",
            "tracking_rb_players",
            "tracking_db_players",
            "tracking_start_offense_x_span",
            "tracking_start_offense_y_span",
            "tracking_end_offense_x_span",
            "tracking_end_offense_y_span",
            "tracking_start_defense_x_span",
            "tracking_start_defense_y_span",
            "tracking_end_defense_x_span",
            "tracking_end_defense_y_span",
            "tracking_offense_centroid_shift",
            "tracking_defense_centroid_shift",
            "tracking_offense_mean_displacement",
            "tracking_defense_mean_displacement",
            "tracking_offense_max_displacement",
            "tracking_defense_max_displacement",
            "tracking_start_skill_mean_separation",
            "tracking_end_skill_mean_separation",
            "tracking_skill_separation_gain",
            "tracking_start_qb_nearest_defender_distance",
            "tracking_end_qb_nearest_defender_distance",
            "tracking_qb_pressure_distance_delta",
        ]
    )
    tracking_categorical: list[str] = field(
        default_factory=lambda: [
            "tracking_play_direction",
        ]
    )
    derived_numeric: list[str] = field(default_factory=lambda: ["score_margin_abs"])
    derived_categorical: list[str] = field(
        default_factory=lambda: [
            "down_bucket",
            "distance_bucket",
            "field_zone",
            "score_state",
            "pressure_bucket",
            "box_bucket",
            "backfield_bucket",
            "clock_bucket",
        ]
    )
    motion_related_columns: list[str] = field(
        default_factory=lambda: [
            "is_motion",
            "is_no_huddle",
            "is_play_action",
            "is_screen_pass",
            "is_rpo",
            "is_trick_play",
            "n_blitzers",
            "n_pass_rushers",
            "n_defense_box",
            "n_offense_backfield",
            "qb_location",
            "starting_hash",
            "pressure_bucket",
            "box_bucket",
            "backfield_bucket",
        ]
    )
    tracking_response_columns: list[str] = field(
        default_factory=lambda: [
            "tracking_end_offense_x_span",
            "tracking_end_offense_y_span",
            "tracking_end_defense_x_span",
            "tracking_end_defense_y_span",
            "tracking_offense_centroid_shift",
            "tracking_defense_centroid_shift",
            "tracking_offense_mean_displacement",
            "tracking_defense_mean_displacement",
            "tracking_offense_max_displacement",
            "tracking_defense_max_displacement",
            "tracking_end_skill_mean_separation",
            "tracking_skill_separation_gain",
            "tracking_end_qb_nearest_defender_distance",
            "tracking_qb_pressure_distance_delta",
        ]
    )


@dataclass(slots=True)
class TargetsConfig:
    classification_targets: list[str] = field(
        default_factory=lambda: ["success", "explosive", "completion"]
    )
    regression_targets: list[str] = field(default_factory=lambda: ["epa"])
    explosive_threshold: int = 20


@dataclass(slots=True)
class SplitConfig:
    train_seasons: list[int] = field(default_factory=lambda: [2022, 2023, 2024])
    validation_seasons: list[int] = field(default_factory=list)
    test_seasons: list[int] = field(default_factory=lambda: [2025])


@dataclass(slots=True)
class ModelsConfig:
    classification_models: list[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        ]
    )
    regression_models: list[str] = field(
        default_factory=lambda: [
            "ridge_regression",
            "random_forest",
            "gradient_boosting",
        ]
    )
    random_state: int = 42


@dataclass(slots=True)
class EvaluationConfig:
    classification_threshold: float = 0.5
    calibration_bins: int = 10
    subgroup_columns: list[str] = field(
        default_factory=lambda: ["down_bucket", "field_zone", "pressure_bucket"]
    )
    minimum_subgroup_size: int = 50


@dataclass(slots=True)
class ProjectConfig:
    project_name: str = "reading-the-defense"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        valid_classification_targets = {"success", "explosive", "completion"}
        valid_regression_targets = {"epa"}

        overlap = set(self.split.train_seasons) & set(self.split.test_seasons)
        if overlap:
            raise ValueError(f"Train and test seasons overlap: {sorted(overlap)}")
        overlap = set(self.split.validation_seasons) & set(self.split.test_seasons)
        if overlap:
            raise ValueError(f"Validation and test seasons overlap: {sorted(overlap)}")
        if self.data.use_ftn_charting and not self.data.ftn_seasons:
            raise ValueError("FTN charting is enabled but no FTN seasons were supplied.")
        unknown_classification = set(self.targets.classification_targets) - valid_classification_targets
        if unknown_classification:
            raise ValueError(
                f"Unknown classification targets: {sorted(unknown_classification)}"
            )
        unknown_regression = set(self.targets.regression_targets) - valid_regression_targets
        if unknown_regression:
            raise ValueError(f"Unknown regression targets: {sorted(unknown_regression)}")


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    paths_data = payload.get("paths", {})
    data_data = payload.get("data", {})
    tracking_data = payload.get("tracking", {})
    features_data = payload.get("features", {})
    targets_data = payload.get("targets", {})
    split_data = payload.get("split", {})
    models_data = payload.get("models", {})
    evaluation_data = payload.get("evaluation", {})

    config = ProjectConfig(
        project_name=_value(payload, "project_name", "reading-the-defense"),
        paths=PathsConfig(
            raw_dir=_value(paths_data, "raw_dir", "data/raw"),
            processed_dir=_value(paths_data, "processed_dir", "data/processed"),
            artifacts_dir=_value(paths_data, "artifacts_dir", "artifacts"),
        ),
        data=DataConfig(
            pbp_seasons=_value(data_data, "pbp_seasons", [2022, 2023, 2024, 2025]),
            ftn_seasons=_value(data_data, "ftn_seasons", [2022, 2023, 2024, 2025]),
            use_ftn_charting=_value(data_data, "use_ftn_charting", True),
            season_type=_value(data_data, "season_type", "REG"),
        ),
        tracking=TrackingConfig(
            enabled=_value(tracking_data, "enabled", False),
            input_globs=_value(tracking_data, "input_globs", ["train/input_*.csv"]),
            input_files=_value(tracking_data, "input_files", ["test_input.csv"]),
            cache_file=_value(tracking_data, "cache_file", "tracking_play_features_v2.parquet"),
        ),
        features=FeaturesConfig(
            base_numeric=_value(features_data, "base_numeric", FeaturesConfig().base_numeric),
            base_categorical=_value(
                features_data, "base_categorical", FeaturesConfig().base_categorical
            ),
            ftn_numeric=_value(features_data, "ftn_numeric", FeaturesConfig().ftn_numeric),
            ftn_categorical=_value(
                features_data, "ftn_categorical", FeaturesConfig().ftn_categorical
            ),
            tracking_numeric=_value(
                features_data, "tracking_numeric", FeaturesConfig().tracking_numeric
            ),
            tracking_categorical=_value(
                features_data,
                "tracking_categorical",
                FeaturesConfig().tracking_categorical,
            ),
            derived_numeric=_value(
                features_data, "derived_numeric", FeaturesConfig().derived_numeric
            ),
            derived_categorical=_value(
                features_data,
                "derived_categorical",
                FeaturesConfig().derived_categorical,
            ),
            motion_related_columns=_value(
                features_data,
                "motion_related_columns",
                FeaturesConfig().motion_related_columns,
            ),
            tracking_response_columns=_value(
                features_data,
                "tracking_response_columns",
                FeaturesConfig().tracking_response_columns,
            ),
        ),
        targets=TargetsConfig(
            classification_targets=_legacy_or_default(
                targets_data,
                "classification_targets",
                "classification_target",
                TargetsConfig().classification_targets,
            ),
            regression_targets=_legacy_or_default(
                targets_data,
                "regression_targets",
                "regression_target",
                TargetsConfig().regression_targets,
            ),
            explosive_threshold=_value(targets_data, "explosive_threshold", 20),
        ),
        split=SplitConfig(
            train_seasons=_value(split_data, "train_seasons", [2022, 2023, 2024]),
            validation_seasons=_value(split_data, "validation_seasons", []),
            test_seasons=_value(split_data, "test_seasons", [2025]),
        ),
        models=ModelsConfig(
            classification_models=_value(
                models_data,
                "classification_models",
                ModelsConfig().classification_models,
            ),
            regression_models=_value(
                models_data, "regression_models", ModelsConfig().regression_models
            ),
            random_state=_value(models_data, "random_state", 42),
        ),
        evaluation=EvaluationConfig(
            classification_threshold=_value(
                evaluation_data, "classification_threshold", 0.5
            ),
            calibration_bins=_value(evaluation_data, "calibration_bins", 10),
            subgroup_columns=_value(
                evaluation_data,
                "subgroup_columns",
                EvaluationConfig().subgroup_columns,
            ),
            minimum_subgroup_size=_value(
                evaluation_data, "minimum_subgroup_size", 50
            ),
        ),
    )
    config.validate()
    return config
