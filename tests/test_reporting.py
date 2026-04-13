import pandas as pd

from pre_snap_motion.evaluation.reporting import best_models, dataset_summary


def test_dataset_summary_reports_split_tracking_coverage() -> None:
    frame = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "has_ftn_charting": [1, 1, 1, 1],
            "has_tracking_data": [1, 0, 1, 1],
            "target_success": [1, 0, 1, 0],
        }
    )

    summary = dataset_summary(
        frame=frame,
        train_frame=frame.iloc[:2].copy(),
        test_frame=frame.iloc[2:].copy(),
        target_columns={"success": "target_success"},
    )

    assert summary["train_rows"] == 2
    assert summary["test_rows"] == 2
    assert summary["train_tracking_coverage_rate"] == 0.5
    assert summary["test_tracking_coverage_rate"] == 1.0
    assert summary["test_tracking_rows"] == 2


def test_best_models_selects_within_each_evaluation_slice() -> None:
    metrics = pd.DataFrame(
        [
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "success",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "auroc": 0.60,
            },
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "success",
                "model_name": "random_forest",
                "feature_set": "full",
                "auroc": 0.65,
            },
            {
                "evaluation_slice": "tracking_only",
                "task": "classification",
                "target": "success",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "auroc": 0.70,
            },
            {
                "evaluation_slice": "tracking_only",
                "task": "classification",
                "target": "success",
                "model_name": "random_forest",
                "feature_set": "full",
                "auroc": 0.68,
            },
        ]
    )

    best = best_models(metrics)

    assert set(best["evaluation_slice"]) == {"all", "tracking_only"}
    assert (
        best.loc[best["evaluation_slice"] == "all", "model_name"].iloc[0]
        == "random_forest"
    )
    assert (
        best.loc[best["evaluation_slice"] == "tracking_only", "model_name"].iloc[0]
        == "logistic_regression"
    )
