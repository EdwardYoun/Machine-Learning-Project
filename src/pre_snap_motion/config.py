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
            "tracking_start_offense_speed_mean",
            "tracking_start_defense_speed_mean",
            "tracking_end_offense_speed_mean",
            "tracking_end_defense_speed_mean",
            "tracking_start_offense_accel_mean",
            "tracking_start_defense_accel_mean",
            "tracking_end_offense_accel_mean",
            "tracking_end_defense_accel_mean",
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
            "tracking_speed_advantage_gain",
            "tracking_accel_advantage_gain",
            "tracking_defense_y_span_change",
            "tracking_offense_y_span_change",
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
    motion_numeric: list[str] = field(
        default_factory=lambda: [
            "is_motion_flag",
            "play_action_motion_flag",
            "late_down_motion_flag",
            "red_zone_motion_flag",
            "under_center_motion_flag",
        ]
    )
    motion_related_columns: list[str] = field(
        default_factory=lambda: [
            "is_motion",
            "is_motion_flag",
            "is_no_huddle",
            "is_play_action",
            "play_action_motion_flag",
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
            "late_down_motion_flag",
            "red_zone_motion_flag",
            "under_center_motion_flag",
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
            "tracking_end_offense_speed_mean",
            "tracking_end_defense_speed_mean",
            "tracking_end_offense_accel_mean",
            "tracking_end_defense_accel_mean",
            "tracking_speed_advantage_gain",
            "tracking_accel_advantage_gain",
            "tracking_defense_y_span_change",
            "tracking_offense_y_span_change",
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
    strategy: str = "explicit"
    train_seasons: list[int] = field(default_factory=lambda: [2022, 2023, 2024])
    validation_seasons: list[int] = field(default_factory=list)
    test_seasons: list[int] = field(default_factory=lambda: [2025])
    rolling_min_train_seasons: int = 2
    rolling_min_train_weeks: int = 8
    rolling_validation_window_weeks: int = 1


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
    target_classification_models: dict[str, list[str]] = field(default_factory=dict)
    target_regression_models: dict[str, list[str]] = field(default_factory=dict)
    classification_calibration_method: str = "none"
    random_state: int = 42


@dataclass(slots=True)
class EvaluationConfig:
    classification_threshold: float = 0.5
    classification_threshold_grid: list[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7]
    )
    threshold_selection_metric: str = "balanced_accuracy"
    classification_selection_metric: str = "auroc"
    regression_selection_metric: str = "rmse"
    calibration_bins: int = 10
    motion_effect_control_columns: list[str] = field(
        default_factory=lambda: ["down_bucket", "distance_bucket", "field_zone", "score_state"]
    )
    subgroup_columns: list[str] = field(
        default_factory=lambda: ["down_bucket", "field_zone", "pressure_bucket"]
    )
    minimum_subgroup_size: int = 50
    motion_effect_minimum_size: int = 100
    defensive_response_minimum_size: int = 75
    sparse_tracking_threshold: float = 0.25
    effect_confidence_level: float = 0.95
    effect_bootstrap_samples: int = 200
    effect_random_state: int = 42


@dataclass(slots=True)
class ComparisonConfig:
    primary_target: str = "completion"
    rank_targets: list[str] = field(
        default_factory=lambda: ["completion", "explosive", "success", "epa"]
    )
    classification_metric: str = "balanced_accuracy"
    regression_metric: str = "rmse"


@dataclass(slots=True)
class ExperimentConfig:
    mode: str = "balanced_research"
    enable_motion_effect_analysis: bool = True
    enable_defensive_response_analysis: bool = True
    feature_sets: list[str] = field(
        default_factory=lambda: ["context_only", "context_plus_motion", "full"]
    )


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
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def validate(self) -> None:
        valid_classification_targets = {"success", "explosive", "completion"}
        valid_regression_targets = {"epa"}
        valid_split_strategies = {"explicit", "rolling_origin", "rolling_origin_weeks"}
        valid_experiment_modes = {"balanced_research", "inference_first", "prediction_first"}
        valid_feature_sets = {"context_only", "context_plus_motion", "full"}
        valid_calibration_methods = {"none", "sigmoid", "isotonic"}
        valid_threshold_metrics = {"balanced_accuracy", "f1"}
        valid_classification_selection_metrics = {
            "auroc",
            "balanced_accuracy",
            "f1",
            "log_loss",
            "brier_score",
            "expected_calibration_error",
        }
        valid_regression_selection_metrics = {"rmse", "mae"}

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
        if self.split.strategy not in valid_split_strategies:
            raise ValueError(f"Unknown split strategy: {self.split.strategy}")
        if self.split.rolling_min_train_seasons < 1:
            raise ValueError("rolling_min_train_seasons must be at least 1.")
        if self.split.rolling_min_train_weeks < 1:
            raise ValueError("rolling_min_train_weeks must be at least 1.")
        if self.split.rolling_validation_window_weeks < 1:
            raise ValueError("rolling_validation_window_weeks must be at least 1.")
        if self.experiment.mode not in valid_experiment_modes:
            raise ValueError(f"Unknown experiment mode: {self.experiment.mode}")
        unknown_feature_sets = set(self.experiment.feature_sets) - valid_feature_sets
        if unknown_feature_sets:
            raise ValueError(f"Unknown experiment feature sets: {sorted(unknown_feature_sets)}")
        if "full" not in self.experiment.feature_sets:
            raise ValueError("experiment.feature_sets must include 'full'.")
        if "context_only" not in self.experiment.feature_sets:
            raise ValueError("experiment.feature_sets must include 'context_only'.")
        if (
            self.experiment.enable_defensive_response_analysis
            and "context_plus_motion" not in self.experiment.feature_sets
        ):
            raise ValueError(
                "experiment.feature_sets must include 'context_plus_motion' when defensive response analysis is enabled."
            )
        if self.models.classification_calibration_method not in valid_calibration_methods:
            raise ValueError(
                f"Unknown classification calibration method: {self.models.classification_calibration_method}"
            )
        if self.evaluation.threshold_selection_metric not in valid_threshold_metrics:
            raise ValueError(
                f"Unknown threshold selection metric: {self.evaluation.threshold_selection_metric}"
            )
        if (
            self.evaluation.classification_selection_metric
            not in valid_classification_selection_metrics
        ):
            raise ValueError(
                "Unknown classification selection metric: "
                f"{self.evaluation.classification_selection_metric}"
            )
        if self.evaluation.regression_selection_metric not in valid_regression_selection_metrics:
            raise ValueError(
                f"Unknown regression selection metric: {self.evaluation.regression_selection_metric}"
            )
        valid_comparison_targets = (
            self.targets.classification_targets + self.targets.regression_targets
        )
        if self.comparison.primary_target not in valid_comparison_targets:
            raise ValueError(
                f"Unknown comparison primary target: {self.comparison.primary_target}"
            )
        unknown_rank_targets = set(self.comparison.rank_targets) - set(
            valid_comparison_targets
        )
        if unknown_rank_targets:
            raise ValueError(
                f"Unknown comparison rank targets: {sorted(unknown_rank_targets)}"
            )
        if self.comparison.classification_metric not in valid_classification_selection_metrics:
            raise ValueError(
                "Unknown comparison classification metric: "
                f"{self.comparison.classification_metric}"
            )
        if self.comparison.regression_metric not in valid_regression_selection_metrics:
            raise ValueError(
                f"Unknown comparison regression metric: {self.comparison.regression_metric}"
            )
        if not self.evaluation.classification_threshold_grid:
            raise ValueError("classification_threshold_grid must not be empty.")
        if any(
            threshold <= 0 or threshold >= 1
            for threshold in self.evaluation.classification_threshold_grid
        ):
            raise ValueError("classification_threshold_grid values must be between 0 and 1.")
        if not 0 < self.evaluation.effect_confidence_level < 1:
            raise ValueError("effect_confidence_level must be between 0 and 1.")
        if self.evaluation.effect_bootstrap_samples < 10:
            raise ValueError("effect_bootstrap_samples must be at least 10.")


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
    comparison_data = payload.get("comparison", {})
    experiment_data = payload.get("experiment", {})

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
            motion_numeric=_value(
                features_data,
                "motion_numeric",
                FeaturesConfig().motion_numeric,
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
            strategy=_value(split_data, "strategy", "explicit"),
            train_seasons=_value(split_data, "train_seasons", [2022, 2023, 2024]),
            validation_seasons=_value(split_data, "validation_seasons", []),
            test_seasons=_value(split_data, "test_seasons", [2025]),
            rolling_min_train_seasons=_value(split_data, "rolling_min_train_seasons", 2),
            rolling_min_train_weeks=_value(split_data, "rolling_min_train_weeks", 8),
            rolling_validation_window_weeks=_value(
                split_data,
                "rolling_validation_window_weeks",
                1,
            ),
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
            target_classification_models=_value(
                models_data, "target_classification_models", {}
            ),
            target_regression_models=_value(
                models_data, "target_regression_models", {}
            ),
            classification_calibration_method=_value(
                models_data, "classification_calibration_method", "none"
            ),
            random_state=_value(models_data, "random_state", 42),
        ),
        evaluation=EvaluationConfig(
            classification_threshold=_value(
                evaluation_data, "classification_threshold", 0.5
            ),
            classification_threshold_grid=_value(
                evaluation_data,
                "classification_threshold_grid",
                EvaluationConfig().classification_threshold_grid,
            ),
            threshold_selection_metric=_value(
                evaluation_data, "threshold_selection_metric", "balanced_accuracy"
            ),
            classification_selection_metric=_value(
                evaluation_data, "classification_selection_metric", "auroc"
            ),
            regression_selection_metric=_value(
                evaluation_data, "regression_selection_metric", "rmse"
            ),
            calibration_bins=_value(evaluation_data, "calibration_bins", 10),
            motion_effect_control_columns=_value(
                evaluation_data,
                "motion_effect_control_columns",
                EvaluationConfig().motion_effect_control_columns,
            ),
            subgroup_columns=_value(
                evaluation_data,
                "subgroup_columns",
                EvaluationConfig().subgroup_columns,
            ),
            minimum_subgroup_size=_value(
                evaluation_data, "minimum_subgroup_size", 50
            ),
            motion_effect_minimum_size=_value(
                evaluation_data, "motion_effect_minimum_size", 100
            ),
            defensive_response_minimum_size=_value(
                evaluation_data, "defensive_response_minimum_size", 75
            ),
            sparse_tracking_threshold=_value(
                evaluation_data, "sparse_tracking_threshold", 0.25
            ),
            effect_confidence_level=_value(
                evaluation_data, "effect_confidence_level", 0.95
            ),
            effect_bootstrap_samples=_value(
                evaluation_data, "effect_bootstrap_samples", 200
            ),
            effect_random_state=_value(
                evaluation_data, "effect_random_state", 42
            ),
        ),
        comparison=ComparisonConfig(
            primary_target=_value(comparison_data, "primary_target", "completion"),
            rank_targets=_value(
                comparison_data,
                "rank_targets",
                ComparisonConfig().rank_targets,
            ),
            classification_metric=_value(
                comparison_data, "classification_metric", "balanced_accuracy"
            ),
            regression_metric=_value(comparison_data, "regression_metric", "rmse"),
        ),
        experiment=ExperimentConfig(
            mode=_value(experiment_data, "mode", "balanced_research"),
            enable_motion_effect_analysis=_value(
                experiment_data, "enable_motion_effect_analysis", True
            ),
            enable_defensive_response_analysis=_value(
                experiment_data, "enable_defensive_response_analysis", True
            ),
            feature_sets=_value(
                experiment_data,
                "feature_sets",
                ExperimentConfig().feature_sets,
            ),
        ),
    )
    config.validate()
    return config
