import pandas as pd
import polars as pl

from pre_snap_motion.config import ProjectConfig
from pre_snap_motion.modeling.train import _model_names_for_target, _select_threshold, train_models


def test_select_threshold_uses_requested_metric() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "prediction": [0.2, 0.45, 0.55, 0.9],
        }
    )

    threshold = _select_threshold(
        validation_predictions=predictions,
        thresholds=[0.4, 0.5, 0.6],
        metric_name="balanced_accuracy",
    )

    assert threshold == 0.5


def test_model_names_for_target_respects_target_overrides() -> None:
    config = ProjectConfig()
    config.models.target_classification_models = {"success": ["logistic_regression"]}
    config.models.target_regression_models = {"epa": ["ridge_regression"]}

    assert _model_names_for_target("classification", "success", config) == [
        "logistic_regression"
    ]
    assert _model_names_for_target("classification", "completion", config) == config.models.classification_models
    assert _model_names_for_target("regression", "epa", config) == ["ridge_regression"]


def test_train_models_runs_end_to_end_on_small_dataset(tmp_path) -> None:
    config = ProjectConfig()
    config.project_name = "tiny"
    config.paths.artifacts_dir = str(tmp_path / "artifacts")
    config.models.classification_models = ["logistic_regression"]
    config.models.regression_models = ["ridge_regression"]
    config.models.classification_calibration_method = "none"
    config.split.train_seasons = [2022]
    config.split.validation_seasons = [2023]
    config.split.test_seasons = [2024]
    config.split.strategy = "explicit"
    config.evaluation.classification_selection_metric = "balanced_accuracy"

    rows = []
    for season in [2022, 2023, 2024]:
        for idx in range(6):
            label = idx % 2
            rows.append(
                {
                    "game_id": f"{season}{idx:06d}",
                    "play_id": idx,
                    "season": season,
                    "week": 1,
                    "posteam": "A",
                    "defteam": "B",
                    "down": 1 + (idx % 4),
                    "ydstogo": 3 + idx,
                    "yardline_100": 20 + idx,
                    "game_seconds_remaining": 1200 - idx * 10,
                    "half_seconds_remaining": 600 - idx * 5,
                    "score_differential": idx - 2,
                    "posteam_timeouts_remaining": 3,
                    "defteam_timeouts_remaining": 3,
                    "is_motion": label,
                    "is_no_huddle": 0,
                    "is_play_action": label,
                    "is_screen_pass": 0,
                    "is_rpo": 0,
                    "is_trick_play": 0,
                    "n_blitzers": 4 + label,
                    "n_pass_rushers": 4,
                    "n_offense_backfield": 1 + label,
                    "n_defense_box": 6,
                    "qb_location": "pocket",
                    "starting_hash": "middle",
                    "shotgun": 1,
                    "goal_to_go": 0,
                    "game_half": "Half1",
                    "roof": "outdoors",
                    "surface": "grass",
                    "div_game": 0,
                    "qtr": 1,
                    "down_bucket": "early_down",
                    "distance_bucket": "short",
                    "field_zone": "red_zone",
                    "score_state": "one_score",
                    "pressure_bucket": "standard_rush",
                    "box_bucket": "standard_box",
                    "backfield_bucket": "spread_backfield",
                    "clock_bucket": "standard_clock",
                    "is_motion_flag": label,
                    "play_action_motion_flag": label,
                    "late_down_motion_flag": 0,
                    "red_zone_motion_flag": label,
                    "under_center_motion_flag": 0,
                    "has_ftn_charting": 1,
                    "has_tracking_data": 0,
                    "target_success": label,
                    "target_explosive": label,
                    "target_completion": label,
                    "target_epa": float(label),
                }
            )

    outputs = train_models(pl.DataFrame(rows), config)

    assert outputs["overall_metrics"].exists()
    assert outputs["validation_metrics"].exists()
    assert outputs["selected_models"].exists()
