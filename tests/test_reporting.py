import pandas as pd

from pre_snap_motion.evaluation.reporting import (
    best_models,
    dataset_summary,
    defensive_reaction_overall,
    motion_effect_overall,
)


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
    selection_metrics = pd.DataFrame(
        [
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "success",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "dataset_split": "validation",
                "auroc": 0.60,
            },
            {
                "evaluation_slice": "all",
                "task": "classification",
                "target": "success",
                "model_name": "random_forest",
                "feature_set": "full",
                "dataset_split": "validation",
                "auroc": 0.65,
            },
            {
                "evaluation_slice": "tracking_only",
                "task": "classification",
                "target": "success",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "dataset_split": "validation",
                "auroc": 0.70,
            },
            {
                "evaluation_slice": "tracking_only",
                "task": "classification",
                "target": "success",
                "model_name": "random_forest",
                "feature_set": "full",
                "dataset_split": "validation",
                "auroc": 0.68,
            },
        ]
    )
    overall_metrics = pd.DataFrame(
        [
            {
                "evaluation_slice": "all",
                "dataset_split": "test",
                "task": "classification",
                "target": "success",
                "model_name": "random_forest",
                "feature_set": "full",
                "auroc": 0.64,
            },
            {
                "evaluation_slice": "tracking_only",
                "dataset_split": "test",
                "task": "classification",
                "target": "success",
                "model_name": "logistic_regression",
                "feature_set": "full",
                "auroc": 0.69,
            },
        ]
    )

    best = best_models(selection_metrics, overall_metrics=overall_metrics)

    assert set(best["evaluation_slice"]) == {"all", "tracking_only"}
    assert (
        best.loc[best["evaluation_slice"] == "all", "model_name"].iloc[0]
        == "random_forest"
    )
    assert (
        best.loc[best["evaluation_slice"] == "tracking_only", "model_name"].iloc[0]
        == "logistic_regression"
    )
    assert "test_auroc" in best.columns


def test_motion_effect_overall_computes_adjusted_difference() -> None:
    frame = pd.DataFrame(
        {
            "is_motion_flag": [1, 0, 1, 0],
            "down_bucket": ["early", "early", "late", "late"],
            "distance_bucket": ["short", "short", "long", "long"],
            "target_success": [1, 0, 1, 0],
        }
    )

    effects = motion_effect_overall(
        frame,
        target_columns={"success": "target_success"},
        control_columns=["down_bucket", "distance_bucket"],
        minimum_size=2,
    )

    assert effects.loc[0, "target"] == "success"
    assert effects.loc[0, "adjusted_effect"] == 1.0


def test_defensive_reaction_overall_filters_to_tracking_rows() -> None:
    frame = pd.DataFrame(
        {
            "has_tracking_data": [1, 1, 0, 1],
            "is_motion_flag": [1, 0, 1, 0],
            "down_bucket": ["early", "early", "late", "late"],
            "distance_bucket": ["short", "short", "long", "long"],
            "tracking_skill_separation_gain": [0.2, 0.1, 0.9, 0.0],
        }
    )

    effects = defensive_reaction_overall(
        frame,
        response_columns=["tracking_skill_separation_gain"],
        control_columns=["down_bucket", "distance_bucket"],
        minimum_size=2,
    )

    assert effects.loc[0, "response_column"] == "tracking_skill_separation_gain"
    assert effects.loc[0, "n_obs"] == 2
