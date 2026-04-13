from __future__ import annotations

from typing import Any

import pandas as pd

from pre_snap_motion.evaluation.metrics import classification_metrics, regression_metrics

CLASSIFICATION_METRICS = [
    "auroc",
    "log_loss",
    "brier_score",
    "expected_calibration_error",
]
REGRESSION_METRICS = ["rmse", "mae"]


def subgroup_metrics(
    predictions: pd.DataFrame,
    group_columns: list[str],
    task: str,
    threshold: float,
    bins: int,
    minimum_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in group_columns:
        if column not in predictions.columns:
            continue

        grouped = predictions.groupby(column, dropna=False)
        for subgroup, frame in grouped:
            if len(frame) < minimum_size:
                continue

            row: dict[str, Any] = {
                "group_column": column,
                "group_value": subgroup,
                "n_obs": len(frame),
            }

            if task == "classification":
                row.update(
                    classification_metrics(
                        frame["actual"].to_numpy(),
                        frame["prediction"].to_numpy(),
                        threshold=threshold,
                        bins=bins,
                    )
                )
            else:
                row.update(
                    regression_metrics(
                        frame["actual"].to_numpy(),
                        frame["prediction"].to_numpy(),
                    )
                )

            for metadata_column in ["model_name", "feature_set", "task", "target"]:
                if metadata_column in frame.columns:
                    row[metadata_column] = frame[metadata_column].iloc[0]

            rows.append(row)

    return pd.DataFrame(rows)


def season_summary(frame: pd.DataFrame, target_columns: dict[str, str]) -> pd.DataFrame:
    if "season" not in frame.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for season, season_frame in frame.groupby("season", dropna=False):
        row: dict[str, Any] = {
            "season": season,
            "rows": len(season_frame),
        }
        if "has_ftn_charting" in season_frame.columns:
            row["ftn_coverage_rate"] = float(season_frame["has_ftn_charting"].mean())
        if "has_tracking_data" in season_frame.columns:
            row["tracking_coverage_rate"] = float(season_frame["has_tracking_data"].mean())
        if "is_motion" in season_frame.columns:
            row["motion_rate"] = float(pd.to_numeric(season_frame["is_motion"], errors="coerce").fillna(0).mean())
        for target_name, target_column in target_columns.items():
            if target_column not in season_frame.columns:
                continue
            values = pd.to_numeric(season_frame[target_column], errors="coerce")
            row[f"{target_name}_mean"] = float(values.mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("season")


def dataset_summary(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_columns: dict[str, str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "seasons": sorted(frame["season"].dropna().unique().tolist()) if "season" in frame.columns else [],
    }
    if "has_ftn_charting" in frame.columns:
        summary["ftn_coverage_rate"] = float(frame["has_ftn_charting"].mean())
    if "has_tracking_data" in frame.columns:
        summary["tracking_coverage_rate"] = float(frame["has_tracking_data"].mean())
    for split_name, split_frame in [("train", train_frame), ("test", test_frame)]:
        if "has_ftn_charting" in split_frame.columns:
            summary[f"{split_name}_ftn_coverage_rate"] = float(
                split_frame["has_ftn_charting"].mean()
            )
        if "has_tracking_data" in split_frame.columns:
            tracking_values = (
                pd.to_numeric(split_frame["has_tracking_data"], errors="coerce")
                .fillna(0)
                .astype(float)
            )
            summary[f"{split_name}_tracking_coverage_rate"] = float(
                tracking_values.mean()
            )
            summary[f"{split_name}_tracking_rows"] = int(tracking_values.sum())
    if "is_motion" in frame.columns:
        motion_values = pd.to_numeric(frame["is_motion"], errors="coerce").fillna(0)
        summary["motion_rate"] = float(motion_values.mean())
    summary["target_rates"] = {}
    for target_name, target_column in target_columns.items():
        if target_column in frame.columns:
            values = pd.to_numeric(frame[target_column], errors="coerce")
            summary["target_rates"][target_name] = float(values.mean())
    return summary


def best_models(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    if overall_metrics.empty:
        return pd.DataFrame()

    group_columns = [
        column
        for column in ["evaluation_slice", "task", "target"]
        if column in overall_metrics.columns
    ]
    rows: list[dict[str, Any]] = []
    for group_keys, group in overall_metrics.groupby(group_columns, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        key_map = dict(zip(group_columns, group_keys))
        task = key_map["task"]
        metric = "auroc" if task == "classification" else "rmse"
        ascending = task == "regression"
        candidate_group = group.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        if candidate_group.empty:
            continue
        best = candidate_group.iloc[0].to_dict()
        best["selection_metric"] = metric
        rows.append(best)
    return pd.DataFrame(rows)


def _compare_feature_sets(
    metrics_frame: pd.DataFrame,
    id_columns: list[str],
    value_columns: list[str],
    candidate_feature_set: str,
    baseline_feature_set: str,
) -> pd.DataFrame:
    if metrics_frame.empty:
        return pd.DataFrame()

    candidate = metrics_frame.loc[
        metrics_frame["feature_set"] == candidate_feature_set, id_columns + value_columns
    ]
    baseline = metrics_frame.loc[
        metrics_frame["feature_set"] == baseline_feature_set, id_columns + value_columns
    ]
    merged = candidate.merge(
        baseline,
        on=id_columns,
        suffixes=(f"_{candidate_feature_set}", f"_{baseline_feature_set}"),
    )
    if merged.empty:
        return merged

    candidate_suffix = f"_{candidate_feature_set}"
    baseline_suffix = f"_{baseline_feature_set}"

    if {f"auroc{candidate_suffix}", f"auroc{baseline_suffix}"}.issubset(merged.columns):
        merged["improvement_auroc"] = (
            merged[f"auroc{candidate_suffix}"] - merged[f"auroc{baseline_suffix}"]
        )
    if {f"log_loss{candidate_suffix}", f"log_loss{baseline_suffix}"}.issubset(
        merged.columns
    ):
        merged["improvement_log_loss"] = (
            merged[f"log_loss{baseline_suffix}"] - merged[f"log_loss{candidate_suffix}"]
        )
    if {f"brier_score{candidate_suffix}", f"brier_score{baseline_suffix}"}.issubset(
        merged.columns
    ):
        merged["improvement_brier_score"] = (
            merged[f"brier_score{baseline_suffix}"]
            - merged[f"brier_score{candidate_suffix}"]
        )
    if {
        f"expected_calibration_error{candidate_suffix}",
        f"expected_calibration_error{baseline_suffix}",
    }.issubset(merged.columns):
        merged["improvement_expected_calibration_error"] = (
            merged[f"expected_calibration_error{baseline_suffix}"]
            - merged[f"expected_calibration_error{candidate_suffix}"]
        )
    if {f"rmse{candidate_suffix}", f"rmse{baseline_suffix}"}.issubset(merged.columns):
        merged["improvement_rmse"] = (
            merged[f"rmse{baseline_suffix}"] - merged[f"rmse{candidate_suffix}"]
        )
    if {f"mae{candidate_suffix}", f"mae{baseline_suffix}"}.issubset(merged.columns):
        merged["improvement_mae"] = (
            merged[f"mae{baseline_suffix}"] - merged[f"mae{candidate_suffix}"]
        )

    merged["candidate_feature_set"] = candidate_feature_set
    merged["baseline_feature_set"] = baseline_feature_set

    return merged.sort_values(id_columns)


def motion_lift_overall(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["task", "target", "model_name"]
    if "evaluation_slice" in overall_metrics.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        overall_metrics,
        id_columns=id_columns,
        value_columns=[
            "auroc",
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "rmse",
            "mae",
        ],
        candidate_feature_set="full",
        baseline_feature_set="no_motion",
    )


def motion_lift_subgroups(subgroup_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["task", "target", "model_name", "group_column", "group_value"]
    if "evaluation_slice" in subgroup_metrics_frame.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        subgroup_metrics_frame,
        id_columns=id_columns,
        value_columns=[
            "n_obs",
            "auroc",
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "rmse",
            "mae",
        ],
        candidate_feature_set="full",
        baseline_feature_set="no_motion",
    )


def tracking_response_lift_overall(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["task", "target", "model_name"]
    if "evaluation_slice" in overall_metrics.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        overall_metrics,
        id_columns=id_columns,
        value_columns=[
            "auroc",
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "rmse",
            "mae",
        ],
        candidate_feature_set="full",
        baseline_feature_set="no_tracking_response",
    )


def tracking_response_lift_subgroups(
    subgroup_metrics_frame: pd.DataFrame,
) -> pd.DataFrame:
    id_columns = ["task", "target", "model_name", "group_column", "group_value"]
    if "evaluation_slice" in subgroup_metrics_frame.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        subgroup_metrics_frame,
        id_columns=id_columns,
        value_columns=[
            "n_obs",
            "auroc",
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "rmse",
            "mae",
        ],
        candidate_feature_set="full",
        baseline_feature_set="no_tracking_response",
    )


def proposal_summary_markdown(
    dataset_summary_payload: dict[str, Any],
    best_models_frame: pd.DataFrame,
    motion_lift_frame: pd.DataFrame,
    tracking_response_lift_frame: pd.DataFrame | None = None,
) -> str:
    lines = ["# Proposal Summary", ""]
    lines.append("## Dataset")
    lines.append(f"- Total pass-play rows: {dataset_summary_payload.get('total_rows', 0):,}")
    lines.append(f"- Train rows: {dataset_summary_payload.get('train_rows', 0):,}")
    lines.append(f"- Test rows: {dataset_summary_payload.get('test_rows', 0):,}")
    seasons = dataset_summary_payload.get("seasons", [])
    if seasons:
        lines.append(f"- Seasons included: {', '.join(str(season) for season in seasons)}")
    if "ftn_coverage_rate" in dataset_summary_payload:
        lines.append(
            f"- FTN charting coverage: {dataset_summary_payload['ftn_coverage_rate']:.1%}"
        )
    if "tracking_coverage_rate" in dataset_summary_payload:
        lines.append(
            f"- Tracking coverage: {dataset_summary_payload['tracking_coverage_rate']:.1%}"
        )
    if {
        "train_tracking_coverage_rate",
        "test_tracking_coverage_rate",
    }.issubset(dataset_summary_payload):
        lines.append(
            "- Tracking coverage by split: "
            f"train {dataset_summary_payload['train_tracking_coverage_rate']:.1%}, "
            f"test {dataset_summary_payload['test_tracking_coverage_rate']:.1%}"
        )
        if dataset_summary_payload["test_tracking_coverage_rate"] < 0.25:
            lines.append(
                "- Tracking note: test-split coverage is sparse, so tracking-lift estimates "
                "should be treated as directional."
            )
    if "motion_rate" in dataset_summary_payload:
        lines.append(f"- Motion rate: {dataset_summary_payload['motion_rate']:.1%}")

    target_rates = dataset_summary_payload.get("target_rates", {})
    if target_rates:
        lines.append("")
        lines.append("## Target Rates")
        for target_name, value in target_rates.items():
            lines.append(f"- {target_name}: {value:.3f}")

    if not best_models_frame.empty:
        lines.append("")
        lines.append("## Best Models")
        for _, row in best_models_frame.iterrows():
            metric = row["selection_metric"]
            metric_value = row[metric]
            slice_prefix = ""
            if "evaluation_slice" in row and pd.notna(row["evaluation_slice"]):
                slice_prefix = f"[{row['evaluation_slice']}] "
            lines.append(
                f"- {slice_prefix}{row['task']} / {row['target']}: {row['model_name']} with `{row['feature_set']}` "
                f"({metric}={metric_value:.4f})"
            )

    if not motion_lift_frame.empty:
        lines.append("")
        lines.append("## Motion Lift")
        for _, row in motion_lift_frame.iterrows():
            improvement_columns = [
                column
                for column in row.index
                if column.startswith("improvement_") and pd.notna(row[column])
            ]
            if not improvement_columns:
                continue
            metric_name = improvement_columns[0].replace("improvement_", "")
            slice_prefix = ""
            if "evaluation_slice" in row and pd.notna(row["evaluation_slice"]):
                slice_prefix = f"[{row['evaluation_slice']}] "
            lines.append(
                f"- {slice_prefix}{row['task']} / {row['target']} / {row['model_name']}: "
                f"{metric_name} lift={row[improvement_columns[0]]:.4f}"
            )

    if tracking_response_lift_frame is not None and not tracking_response_lift_frame.empty:
        lines.append("")
        lines.append("## Tracking Response Lift")
        for _, row in tracking_response_lift_frame.iterrows():
            improvement_columns = [
                column
                for column in row.index
                if column.startswith("improvement_") and pd.notna(row[column])
            ]
            if not improvement_columns:
                continue
            metric_name = improvement_columns[0].replace("improvement_", "")
            slice_prefix = ""
            if "evaluation_slice" in row and pd.notna(row["evaluation_slice"]):
                slice_prefix = f"[{row['evaluation_slice']}] "
            lines.append(
                f"- {slice_prefix}{row['task']} / {row['target']} / {row['model_name']}: "
                f"{metric_name} lift={row[improvement_columns[0]]:.4f}"
            )

    return "\n".join(lines) + "\n"
