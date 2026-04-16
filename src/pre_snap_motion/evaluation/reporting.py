from __future__ import annotations

from typing import Any

import numpy as np
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

            for metadata_column in [
                "model_name",
                "feature_set",
                "task",
                "target",
                "dataset_split",
            ]:
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
        if "is_motion_flag" in season_frame.columns:
            row["motion_rate"] = float(
                pd.to_numeric(season_frame["is_motion_flag"], errors="coerce")
                .fillna(0)
                .mean()
            )
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
    validation_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "seasons": sorted(frame["season"].dropna().unique().tolist()) if "season" in frame.columns else [],
    }
    if validation_frame is not None:
        summary["validation_rows"] = int(len(validation_frame))
    if "has_ftn_charting" in frame.columns:
        summary["ftn_coverage_rate"] = float(frame["has_ftn_charting"].mean())
    if "has_tracking_data" in frame.columns:
        summary["tracking_coverage_rate"] = float(frame["has_tracking_data"].mean())
    for split_name, split_frame in [
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ]:
        if split_frame is None:
            continue
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
    if "is_motion_flag" in frame.columns:
        motion_values = pd.to_numeric(frame["is_motion_flag"], errors="coerce").fillna(0)
        summary["motion_rate"] = float(motion_values.mean())
    summary["target_rates"] = {}
    for target_name, target_column in target_columns.items():
        if target_column in frame.columns:
            values = pd.to_numeric(frame[target_column], errors="coerce")
            summary["target_rates"][target_name] = float(values.mean())
    return summary


def best_models(
    selection_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame | None = None,
    classification_metric: str = "auroc",
    regression_metric: str = "rmse",
) -> pd.DataFrame:
    if selection_metrics.empty:
        return pd.DataFrame()

    group_columns = [
        column
        for column in ["evaluation_slice", "task", "target"]
        if column in selection_metrics.columns
    ]
    rows: list[dict[str, Any]] = []
    for group_keys, group in selection_metrics.groupby(group_columns, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        key_map = dict(zip(group_columns, group_keys))
        task = key_map["task"]
        metric = classification_metric if task == "classification" else regression_metric
        ascending = metric in {"rmse", "mae", "log_loss", "brier_score", "expected_calibration_error"}
        candidate_group = group.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        if candidate_group.empty:
            continue
        best = candidate_group.iloc[0].to_dict()
        best["selection_metric"] = metric
        if overall_metrics is not None and not overall_metrics.empty:
            merge_columns = [column for column in group_columns + ["model_name", "feature_set"] if column in overall_metrics.columns]
            matching = overall_metrics
            for column in merge_columns:
                matching = matching.loc[matching[column] == best[column]]
            if not matching.empty:
                test_row = matching.iloc[0].to_dict()
                for column in CLASSIFICATION_METRICS + REGRESSION_METRICS:
                    if column in test_row:
                        best[f"test_{column}"] = test_row[column]
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
    id_columns = ["task", "target", "model_name", "dataset_split"]
    if "evaluation_slice" in overall_metrics.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        overall_metrics,
        id_columns=id_columns,
        value_columns=CLASSIFICATION_METRICS + REGRESSION_METRICS,
        candidate_feature_set="context_plus_motion",
        baseline_feature_set="context_only",
    )


def motion_lift_subgroups(subgroup_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "task",
        "target",
        "model_name",
        "group_column",
        "group_value",
        "dataset_split",
    ]
    if "evaluation_slice" in subgroup_metrics_frame.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        subgroup_metrics_frame,
        id_columns=id_columns,
        value_columns=["n_obs"] + CLASSIFICATION_METRICS + REGRESSION_METRICS,
        candidate_feature_set="context_plus_motion",
        baseline_feature_set="context_only",
    )


def tracking_response_lift_overall(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["task", "target", "model_name", "dataset_split"]
    if "evaluation_slice" in overall_metrics.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        overall_metrics,
        id_columns=id_columns,
        value_columns=CLASSIFICATION_METRICS + REGRESSION_METRICS,
        candidate_feature_set="full",
        baseline_feature_set="context_plus_motion",
    )


def tracking_response_lift_subgroups(
    subgroup_metrics_frame: pd.DataFrame,
) -> pd.DataFrame:
    id_columns = [
        "task",
        "target",
        "model_name",
        "group_column",
        "group_value",
        "dataset_split",
    ]
    if "evaluation_slice" in subgroup_metrics_frame.columns:
        id_columns.insert(0, "evaluation_slice")
    return _compare_feature_sets(
        subgroup_metrics_frame,
        id_columns=id_columns,
        value_columns=["n_obs"] + CLASSIFICATION_METRICS + REGRESSION_METRICS,
        candidate_feature_set="full",
        baseline_feature_set="context_plus_motion",
    )


def _adjusted_motion_effect_rows(
    frame: pd.DataFrame,
    value_column: str,
    control_columns: list[str],
    minimum_size: int,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int,
    group_column: str | None = None,
) -> list[dict[str, Any]]:
    if "is_motion_flag" not in frame.columns or value_column not in frame.columns:
        return []

    clean = frame.copy()
    clean["is_motion_flag"] = (
        pd.to_numeric(clean["is_motion_flag"], errors="coerce").fillna(0).astype(int)
    )
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean.dropna(subset=[value_column])
    if clean.empty:
        return []

    rows: list[dict[str, Any]] = []
    outer_groups = [(None, clean)] if group_column is None or group_column not in clean.columns else clean.groupby(group_column, dropna=False)
    for outer_value, outer_frame in outer_groups:
        subgroup_rows: list[dict[str, Any]] = []
        available_controls = [column for column in control_columns if column in outer_frame.columns]
        if available_controls:
            grouped = outer_frame.groupby(available_controls, dropna=False)
        else:
            grouped = [(("all",), outer_frame)]
        for _, control_frame in grouped:
            motion_frame = control_frame.loc[control_frame["is_motion_flag"] == 1]
            no_motion_frame = control_frame.loc[control_frame["is_motion_flag"] == 0]
            total_rows = len(control_frame)
            if (
                total_rows < minimum_size
                or motion_frame.empty
                or no_motion_frame.empty
            ):
                continue
            subgroup_rows.append(
                {
                    "n_obs": total_rows,
                    "motion_rows": len(motion_frame),
                    "no_motion_rows": len(no_motion_frame),
                    "motion_mean": float(motion_frame[value_column].mean()),
                    "no_motion_mean": float(no_motion_frame[value_column].mean()),
                }
            )
        if not subgroup_rows:
            continue

        subgroup_frame = pd.DataFrame(subgroup_rows)
        weights = subgroup_frame["n_obs"] / subgroup_frame["n_obs"].sum()
        adjusted_effect = float(
            ((subgroup_frame["motion_mean"] - subgroup_frame["no_motion_mean"]) * weights).sum()
        )
        ci_lower, ci_upper = _bootstrap_effect_interval(
            subgroup_frame=subgroup_frame,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        )
        row: dict[str, Any] = {
            "n_context_groups": int(len(subgroup_frame)),
            "n_obs": int(subgroup_frame["n_obs"].sum()),
            "motion_rows": int(subgroup_frame["motion_rows"].sum()),
            "no_motion_rows": int(subgroup_frame["no_motion_rows"].sum()),
            "adjusted_motion_mean": float((subgroup_frame["motion_mean"] * weights).sum()),
            "adjusted_no_motion_mean": float((subgroup_frame["no_motion_mean"] * weights).sum()),
            "adjusted_effect": adjusted_effect,
            "effect_ci_lower": ci_lower,
            "effect_ci_upper": ci_upper,
            "effect_direction": _effect_direction(adjusted_effect, ci_lower, ci_upper),
        }
        if group_column is not None and group_column in outer_frame.columns:
            row["group_column"] = group_column
            row["group_value"] = outer_value
        rows.append(row)
    return rows


def _bootstrap_effect_interval(
    subgroup_frame: pd.DataFrame,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int,
) -> tuple[float, float]:
    if subgroup_frame.empty:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    effects = subgroup_frame["motion_mean"].to_numpy() - subgroup_frame["no_motion_mean"].to_numpy()
    weights = subgroup_frame["n_obs"].to_numpy(dtype=float)
    normalized = weights / weights.sum()
    bootstrap_effects: list[float] = []
    for _ in range(bootstrap_samples):
        sample_idx = rng.choice(len(subgroup_frame), size=len(subgroup_frame), replace=True, p=normalized)
        sample_effects = effects[sample_idx]
        sample_weights = normalized[sample_idx]
        sample_weights = sample_weights / sample_weights.sum()
        bootstrap_effects.append(float((sample_effects * sample_weights).sum()))
    alpha = (1 - confidence_level) / 2
    return (
        float(np.quantile(bootstrap_effects, alpha)),
        float(np.quantile(bootstrap_effects, 1 - alpha)),
    )


def _effect_direction(effect: float, ci_lower: float, ci_upper: float) -> str:
    if effect > 0 and ci_lower > 0:
        return "helps"
    if effect < 0 and ci_upper < 0:
        return "hurts"
    return "unclear"


def motion_effect_overall(
    frame: pd.DataFrame,
    target_columns: dict[str, str],
    control_columns: list[str],
    minimum_size: int,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int,
    dataset_split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_name, column_name in target_columns.items():
        for row in _adjusted_motion_effect_rows(
            frame=frame,
            value_column=column_name,
            control_columns=control_columns,
            minimum_size=minimum_size,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        ):
            row["target"] = target_name
            row["target_column"] = column_name
            row["dataset_split"] = dataset_split
            rows.append(row)
    return pd.DataFrame(rows)


def motion_effect_subgroups(
    frame: pd.DataFrame,
    target_columns: dict[str, str],
    control_columns: list[str],
    subgroup_columns: list[str],
    minimum_size: int,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int,
    dataset_split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_name, column_name in target_columns.items():
        for subgroup_column in subgroup_columns:
            for row in _adjusted_motion_effect_rows(
                frame=frame,
                value_column=column_name,
                control_columns=[column for column in control_columns if column != subgroup_column],
                minimum_size=minimum_size,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                random_state=random_state,
                group_column=subgroup_column,
            ):
                row["target"] = target_name
                row["target_column"] = column_name
                row["dataset_split"] = dataset_split
                rows.append(row)
    return pd.DataFrame(rows)


def defensive_reaction_overall(
    frame: pd.DataFrame,
    response_columns: list[str],
    control_columns: list[str],
    minimum_size: int,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int,
    dataset_split: str,
    sparse_tracking_threshold: float,
) -> pd.DataFrame:
    if "has_tracking_data" not in frame.columns:
        return pd.DataFrame()
    tracking_frame = frame.loc[
        pd.to_numeric(frame["has_tracking_data"], errors="coerce").fillna(0).astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for column_name in response_columns:
        if column_name not in tracking_frame.columns:
            continue
        for row in _adjusted_motion_effect_rows(
            frame=tracking_frame,
            value_column=column_name,
            control_columns=control_columns,
            minimum_size=minimum_size,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        ):
            row["response_column"] = column_name
            row["dataset_split"] = dataset_split
            row["tracking_coverage_rate"] = float(
                pd.to_numeric(frame["has_tracking_data"], errors="coerce").fillna(0).mean()
            )
            row["tracking_is_sparse"] = row["tracking_coverage_rate"] < sparse_tracking_threshold
            row["reportable"] = not row["tracking_is_sparse"]
            rows.append(row)
    return pd.DataFrame(rows)


def defensive_reaction_subgroups(
    frame: pd.DataFrame,
    response_columns: list[str],
    control_columns: list[str],
    subgroup_columns: list[str],
    minimum_size: int,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int,
    dataset_split: str,
    sparse_tracking_threshold: float,
) -> pd.DataFrame:
    if "has_tracking_data" not in frame.columns:
        return pd.DataFrame()
    tracking_frame = frame.loc[
        pd.to_numeric(frame["has_tracking_data"], errors="coerce").fillna(0).astype(bool)
    ].copy()
    rows: list[dict[str, Any]] = []
    for column_name in response_columns:
        if column_name not in tracking_frame.columns:
            continue
        for subgroup_column in subgroup_columns:
            for row in _adjusted_motion_effect_rows(
                frame=tracking_frame,
                value_column=column_name,
                control_columns=[column for column in control_columns if column != subgroup_column],
                minimum_size=minimum_size,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                random_state=random_state,
                group_column=subgroup_column,
            ):
                row["response_column"] = column_name
                row["dataset_split"] = dataset_split
                row["tracking_coverage_rate"] = float(
                    pd.to_numeric(frame["has_tracking_data"], errors="coerce").fillna(0).mean()
                )
                row["tracking_is_sparse"] = row["tracking_coverage_rate"] < sparse_tracking_threshold
                row["reportable"] = not row["tracking_is_sparse"]
                rows.append(row)
    return pd.DataFrame(rows)


def _lift_lines(title: str, frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    lines = ["", title]
    for _, row in frame.iterrows():
        improvement_columns = [
            column for column in row.index if column.startswith("improvement_") and pd.notna(row[column])
        ]
        if not improvement_columns:
            continue
        metric_name = improvement_columns[0].replace("improvement_", "")
        slice_prefix = ""
        if "evaluation_slice" in row and pd.notna(row["evaluation_slice"]):
            slice_prefix = f"[{row['evaluation_slice']}] "
        split_prefix = f"{row['dataset_split']} / " if "dataset_split" in row and pd.notna(row["dataset_split"]) else ""
        lines.append(
            f"- {slice_prefix}{split_prefix}{row['task']} / {row['target']} / {row['model_name']}: "
            f"{metric_name} lift={row[improvement_columns[0]]:.4f}"
        )
    return lines


def proposal_summary_markdown(
    dataset_summary_payload: dict[str, Any],
    best_models_frame: pd.DataFrame,
    motion_lift_frame: pd.DataFrame,
    tracking_response_lift_frame: pd.DataFrame | None = None,
    motion_effect_frame: pd.DataFrame | None = None,
    defensive_reaction_frame: pd.DataFrame | None = None,
) -> str:
    lines = ["# Motion Value Summary", ""]
    lines.append("## Dataset")
    lines.append(f"- Total pass-play rows: {dataset_summary_payload.get('total_rows', 0):,}")
    lines.append(f"- Train rows: {dataset_summary_payload.get('train_rows', 0):,}")
    if "validation_rows" in dataset_summary_payload:
        lines.append(f"- Validation rows: {dataset_summary_payload.get('validation_rows', 0):,}")
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
        if dataset_summary_payload["test_tracking_coverage_rate"] < dataset_summary_payload.get(
            "sparse_tracking_threshold",
            0.25,
        ):
            lines.append(
                "- Tracking note: test-split coverage is sparse, so defensive-response conclusions "
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

    if motion_effect_frame is not None and not motion_effect_frame.empty:
        lines.append("")
        lines.append("## Overall Motion Effect")
        for _, row in motion_effect_frame.iterrows():
            lines.append(
                f"- {row['dataset_split']} / {row['target']}: motion {row['effect_direction']} by "
                f"{row['adjusted_effect']:.4f} "
                f"(CI {row['effect_ci_lower']:.4f} to {row['effect_ci_upper']:.4f}) after context controls "
                f"({int(row['n_obs']):,} rows across {int(row['n_context_groups'])} groups)."
            )

    if not best_models_frame.empty:
        lines.append("")
        lines.append("## Validation-Selected Models")
        for _, row in best_models_frame.iterrows():
            metric = row["selection_metric"]
            metric_value = row[metric]
            slice_prefix = ""
            if "evaluation_slice" in row and pd.notna(row["evaluation_slice"]):
                slice_prefix = f"[{row['evaluation_slice']}] "
            test_suffix = ""
            if f"test_{metric}" in row and pd.notna(row[f"test_{metric}"]):
                test_suffix = f", test_{metric}={row[f'test_{metric}']:.4f}"
            lines.append(
                f"- {slice_prefix}{row['task']} / {row['target']}: {row['model_name']} with `{row['feature_set']}` "
                f"(validation_{metric}={metric_value:.4f}{test_suffix})"
            )

    lines.extend(_lift_lines("## Motion Lift", motion_lift_frame))
    if tracking_response_lift_frame is not None:
        lines.extend(_lift_lines("## Defensive Response Contribution", tracking_response_lift_frame))

    if defensive_reaction_frame is not None and not defensive_reaction_frame.empty:
        lines.append("")
        lines.append("## Defensive Reaction Highlights")
        reportable = defensive_reaction_frame.loc[
            defensive_reaction_frame.get("reportable", True).astype(bool)
        ]
        highlight_source = reportable if not reportable.empty else defensive_reaction_frame
        if reportable.empty:
            lines.append("- Tracking coverage is sparse on the evaluated split, so defensive reaction results are directional only.")
        highlight = highlight_source.reindex(
            highlight_source["adjusted_effect"].abs().sort_values(ascending=False).index
        ).head(5)
        for _, row in highlight.iterrows():
            sparse_suffix = " (directional only)" if row.get("tracking_is_sparse") else ""
            lines.append(
                f"- {row['dataset_split']} / {row['response_column']}: adjusted motion effect={row['adjusted_effect']:.4f} "
                f"(CI {row['effect_ci_lower']:.4f} to {row['effect_ci_upper']:.4f}) across {int(row['n_obs']):,} tracking rows{sparse_suffix}."
            )

    return "\n".join(lines) + "\n"
