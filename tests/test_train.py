import pandas as pd

from pre_snap_motion.config import ProjectConfig
from pre_snap_motion.modeling.train import _model_names_for_target, _select_threshold


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
