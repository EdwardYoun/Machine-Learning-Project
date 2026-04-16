from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from pre_snap_motion.config import ProjectConfig
from pre_snap_motion.data.splits import rolling_origin_validation_splits, split_frame
from pre_snap_motion.evaluation.metrics import classification_metrics, regression_metrics
from pre_snap_motion.evaluation.reporting import (
    best_models,
    dataset_summary,
    defensive_reaction_overall,
    defensive_reaction_subgroups,
    motion_effect_overall,
    motion_effect_subgroups,
    motion_lift_overall,
    motion_lift_subgroups,
    proposal_summary_markdown,
    season_summary,
    subgroup_metrics,
    tracking_response_lift_overall,
    tracking_response_lift_subgroups,
)
from pre_snap_motion.io import project_artifacts_dir, write_frame, write_json, write_text
from pre_snap_motion.modeling.registry import (
    build_classification_model,
    build_regression_model,
)


CLASSIFICATION_TARGET_MAP = {
    "success": "target_success",
    "explosive": "target_explosive",
    "completion": "target_completion",
}

REGRESSION_TARGET_MAP = {
    "epa": "target_epa",
}


@dataclass(slots=True)
class FeatureCatalog:
    numeric: list[str]
    categorical: list[str]

    @property
    def all_columns(self) -> list[str]:
        return self.numeric + self.categorical


def _existing(columns: list[str], candidates: list[str]) -> list[str]:
    available = set(columns)
    return [candidate for candidate in candidates if candidate in available]


def feature_catalog(config: ProjectConfig, columns: list[str]) -> FeatureCatalog:
    numeric = _existing(
        columns,
        config.features.base_numeric
        + config.features.ftn_numeric
        + config.features.tracking_numeric
        + config.features.derived_numeric
        + config.features.motion_numeric,
    )
    categorical = _existing(
        columns,
        config.features.base_categorical
        + config.features.ftn_categorical
        + config.features.tracking_categorical
        + config.features.derived_categorical,
    )
    return FeatureCatalog(numeric=numeric, categorical=categorical)


def feature_sets(config: ProjectConfig, catalog: FeatureCatalog) -> dict[str, FeatureCatalog]:
    motion_columns = set(config.features.motion_related_columns)
    tracking_response_columns = set(config.features.tracking_response_columns)

    context_only = FeatureCatalog(
        numeric=[column for column in catalog.numeric if column not in motion_columns],
        categorical=[column for column in catalog.categorical if column not in motion_columns],
    )
    context_plus_motion = FeatureCatalog(
        numeric=[column for column in catalog.numeric if column not in tracking_response_columns],
        categorical=[
            column for column in catalog.categorical if column not in tracking_response_columns
        ],
    )
    variants = {
        "context_only": context_only,
        "context_plus_motion": context_plus_motion,
        "full": catalog,
    }
    return {
        name: variants[name]
        for name in config.experiment.feature_sets
        if name in variants and variants[name].all_columns
    }


def _artifacts_dir(config: ProjectConfig, name: str) -> Path:
    return project_artifacts_dir(config) / name


def _save_model(model: Any, config: ProjectConfig, file_name: str) -> Path:
    output_path = _artifacts_dir(config, "models") / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)
    return output_path


def _target_column(task: str, target_name: str) -> str:
    if task == "classification":
        return CLASSIFICATION_TARGET_MAP[target_name]
    return REGRESSION_TARGET_MAP[target_name]


def _target_names(task: str, config: ProjectConfig) -> list[str]:
    if task == "classification":
        return config.targets.classification_targets
    return config.targets.regression_targets


def _prediction_frame(
    frame: pd.DataFrame,
    predictions: pd.Series,
    target_column: str,
    model_name: str,
    feature_set: str,
    task: str,
    target: str,
    dataset_split: str,
    config: ProjectConfig,
) -> pd.DataFrame:
    columns = ["season"] + [
        column
        for column in config.evaluation.subgroup_columns
        if column in frame.columns
    ]
    columns.extend(
        [
            column
            for column in ["has_ftn_charting", "has_tracking_data", "is_motion_flag"]
            if column in frame.columns and column not in columns
        ]
    )
    output = frame.loc[:, columns].copy()
    output["actual"] = frame[target_column].to_numpy()
    output["prediction"] = predictions.to_numpy()
    output["model_name"] = model_name
    output["feature_set"] = feature_set
    output["task"] = task
    output["target"] = target
    output["dataset_split"] = dataset_split
    return output


def _evaluation_slices(prediction_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slices = {"all": prediction_frame}
    if "has_tracking_data" not in prediction_frame.columns:
        return slices

    tracking_mask = (
        pd.to_numeric(prediction_frame["has_tracking_data"], errors="coerce")
        .fillna(0)
        .astype(bool)
    )
    tracking_only = prediction_frame.loc[tracking_mask].copy()
    if 0 < len(tracking_only) < len(prediction_frame):
        slices["tracking_only"] = tracking_only
    return slices


def _metric_row(
    evaluation_frame: pd.DataFrame,
    task: str,
    target_name: str,
    model_name: str,
    feature_set_name: str,
    dataset_split: str,
    feature_columns: FeatureCatalog,
    threshold: float,
    bins: int,
    train_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation_slice_name, slice_frame in _evaluation_slices(evaluation_frame).items():
        if task == "classification":
            metrics = classification_metrics(
                slice_frame["actual"].to_numpy(),
                slice_frame["prediction"].to_numpy(),
                threshold=threshold,
                bins=bins,
            )
        else:
            metrics = regression_metrics(
                slice_frame["actual"].to_numpy(),
                slice_frame["prediction"].to_numpy(),
            )

        row: dict[str, Any] = {
            "evaluation_slice": evaluation_slice_name,
            "dataset_split": dataset_split,
            "task": task,
            "target": target_name,
            "model_name": model_name,
            "feature_set": feature_set_name,
            "train_rows": train_rows,
            "evaluation_rows": len(slice_frame),
            "n_numeric_features": len(feature_columns.numeric),
            "n_categorical_features": len(feature_columns.categorical),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _fit_and_predict(
    task: str,
    model_name: str,
    feature_columns: FeatureCatalog,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[Any, pd.Series]:
    if task == "classification":
        model = build_classification_model(
            model_name,
            feature_columns.numeric,
            feature_columns.categorical,
            random_state,
        )
    else:
        model = build_regression_model(
            model_name,
            feature_columns.numeric,
            feature_columns.categorical,
            random_state,
        )

    X_train = train_frame.loc[:, feature_columns.all_columns]
    y_train = train_frame[target_column]
    X_eval = eval_frame.loc[:, feature_columns.all_columns]

    model.fit(X_train, y_train)
    if task == "classification":
        predictions = pd.Series(model.predict_proba(X_eval)[:, 1], index=X_eval.index)
    else:
        predictions = pd.Series(model.predict(X_eval), index=X_eval.index)
    return model, predictions


def _aggregate_selection_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in [
            "auroc",
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "predicted_positive_rate",
            "rmse",
            "mae",
        ]
        if column in frame.columns
    ]
    group_columns = [
        "evaluation_slice",
        "task",
        "target",
        "model_name",
        "feature_set",
    ]
    aggregated = (
        frame.groupby(group_columns, dropna=False)[metric_columns + ["evaluation_rows"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    aggregated["dataset_split"] = "validation"
    return aggregated


def train_models(dataset: pl.DataFrame, config: ProjectConfig) -> dict[str, Path]:
    frame = dataset.to_pandas()
    split = split_frame(frame, config.split)
    validation_folds = rolling_origin_validation_splits(frame, config.split)

    catalog = feature_catalog(config, frame.columns.tolist())
    feature_variants = feature_sets(config, catalog)

    overall_rows: list[dict[str, Any]] = []
    subgroup_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    model_manifest: dict[str, str] = {}

    test_train_frame = split.train.copy()
    if split.validation is not None:
        test_train_frame = pd.concat([test_train_frame, split.validation], ignore_index=True)

    for task, model_names in (
        ("classification", config.models.classification_models),
        ("regression", config.models.regression_models),
    ):
        for target_name in _target_names(task, config):
            target_column = _target_column(task, target_name)
            if target_column not in frame.columns:
                continue

            train_frame = split.train.dropna(subset=[target_column]).copy()
            test_frame = split.test.dropna(subset=[target_column]).copy()
            if train_frame.empty or test_frame.empty:
                continue

            for feature_set_name, feature_columns in feature_variants.items():
                if not feature_columns.all_columns:
                    continue

                for model_name in model_names:
                    for fold_index, fold in enumerate(validation_folds):
                        if fold.validation is None:
                            continue
                        fold_train = fold.train.dropna(subset=[target_column]).copy()
                        fold_validation = fold.validation.dropna(subset=[target_column]).copy()
                        if fold_train.empty or fold_validation.empty:
                            continue
                        _, validation_predictions = _fit_and_predict(
                            task=task,
                            model_name=model_name,
                            feature_columns=feature_columns,
                            train_frame=fold_train,
                            eval_frame=fold_validation,
                            target_column=target_column,
                            random_state=config.models.random_state,
                        )
                        validation_frame = _prediction_frame(
                            frame=fold_validation,
                            predictions=validation_predictions,
                            target_column=target_column,
                            model_name=model_name,
                            feature_set=feature_set_name,
                            task=task,
                            target=target_name,
                            dataset_split=f"validation_fold_{fold_index + 1}",
                            config=config,
                        )
                        selection_rows.extend(
                            _metric_row(
                                validation_frame,
                                task=task,
                                target_name=target_name,
                                model_name=model_name,
                                feature_set_name=feature_set_name,
                                dataset_split="validation",
                                feature_columns=feature_columns,
                                threshold=config.evaluation.classification_threshold,
                                bins=config.evaluation.calibration_bins,
                                train_rows=len(fold_train),
                            )
                        )

                    final_train = test_train_frame.dropna(subset=[target_column]).copy()
                    if final_train.empty:
                        continue
                    model, test_predictions = _fit_and_predict(
                        task=task,
                        model_name=model_name,
                        feature_columns=feature_columns,
                        train_frame=final_train,
                        eval_frame=test_frame,
                        target_column=target_column,
                        random_state=config.models.random_state,
                    )
                    prediction_frame = _prediction_frame(
                        frame=test_frame,
                        predictions=test_predictions,
                        target_column=target_column,
                        model_name=model_name,
                        feature_set=feature_set_name,
                        task=task,
                        target=target_name,
                        dataset_split="test",
                        config=config,
                    )

                    overall_rows.extend(
                        _metric_row(
                            prediction_frame,
                            task=task,
                            target_name=target_name,
                            model_name=model_name,
                            feature_set_name=feature_set_name,
                            dataset_split="test",
                            feature_columns=feature_columns,
                            threshold=config.evaluation.classification_threshold,
                            bins=config.evaluation.calibration_bins,
                            train_rows=len(final_train),
                        )
                    )

                    subgroup_frame = subgroup_metrics(
                        prediction_frame,
                        group_columns=config.evaluation.subgroup_columns,
                        task=task,
                        threshold=config.evaluation.classification_threshold,
                        bins=config.evaluation.calibration_bins,
                        minimum_size=config.evaluation.minimum_subgroup_size,
                    )
                    if not subgroup_frame.empty:
                        for evaluation_slice_name, slice_frame in _evaluation_slices(prediction_frame).items():
                            scoped = subgroup_metrics(
                                slice_frame,
                                group_columns=config.evaluation.subgroup_columns,
                                task=task,
                                threshold=config.evaluation.classification_threshold,
                                bins=config.evaluation.calibration_bins,
                                minimum_size=config.evaluation.minimum_subgroup_size,
                            )
                            if scoped.empty:
                                continue
                            scoped["evaluation_slice"] = evaluation_slice_name
                            subgroup_frames.append(scoped)

                    model_key = f"{task}:{target_name}:{feature_set_name}:{model_name}"
                    model_path = _save_model(
                        model,
                        config,
                        f"{task}_{target_name}_{feature_set_name}_{model_name}.pkl",
                    )
                    model_manifest[model_key] = str(model_path)

    overall_metrics = pd.DataFrame(overall_rows).sort_values(
        by=["dataset_split", "evaluation_slice", "task", "target", "feature_set", "model_name"]
    ) if overall_rows else pd.DataFrame()
    subgroup_metrics_frame = (
        pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
    )
    validation_metrics = _aggregate_selection_metrics(selection_rows)
    if validation_metrics.empty:
        validation_metrics = overall_metrics.loc[overall_metrics["dataset_split"] == "test"].copy()

    target_columns = {
        **{
            target_name: CLASSIFICATION_TARGET_MAP[target_name]
            for target_name in config.targets.classification_targets
            if CLASSIFICATION_TARGET_MAP[target_name] in frame.columns
        },
        **{
            target_name: REGRESSION_TARGET_MAP[target_name]
            for target_name in config.targets.regression_targets
            if REGRESSION_TARGET_MAP[target_name] in frame.columns
        },
    }
    season_summary_frame = season_summary(frame, target_columns)
    test_metrics = overall_metrics.loc[overall_metrics["dataset_split"] == "test"].copy()
    best_models_frame = best_models(validation_metrics, overall_metrics=test_metrics)
    motion_lift_frame = motion_lift_overall(test_metrics)
    motion_lift_subgroup_frame = motion_lift_subgroups(subgroup_metrics_frame)
    tracking_response_lift_frame = tracking_response_lift_overall(test_metrics)
    tracking_response_lift_subgroup_frame = tracking_response_lift_subgroups(
        subgroup_metrics_frame
    )
    motion_effect_frame = motion_effect_overall(
        frame=frame,
        target_columns=target_columns,
        control_columns=config.evaluation.motion_effect_control_columns,
        minimum_size=config.evaluation.motion_effect_minimum_size,
    )
    motion_effect_subgroup_frame = motion_effect_subgroups(
        frame=frame,
        target_columns=target_columns,
        control_columns=config.evaluation.motion_effect_control_columns,
        subgroup_columns=config.evaluation.subgroup_columns,
        minimum_size=config.evaluation.motion_effect_minimum_size,
    )
    defensive_reaction_frame = defensive_reaction_overall(
        frame=frame,
        response_columns=config.features.tracking_response_columns,
        control_columns=config.evaluation.motion_effect_control_columns,
        minimum_size=config.evaluation.defensive_response_minimum_size,
    )
    defensive_reaction_subgroup_frame = defensive_reaction_subgroups(
        frame=frame,
        response_columns=config.features.tracking_response_columns,
        control_columns=config.evaluation.motion_effect_control_columns,
        subgroup_columns=config.evaluation.subgroup_columns,
        minimum_size=config.evaluation.defensive_response_minimum_size,
    )
    dataset_summary_payload = dataset_summary(
        frame=frame,
        train_frame=split.train,
        validation_frame=split.validation,
        test_frame=split.test,
        target_columns=target_columns,
    )
    dataset_summary_payload["sparse_tracking_threshold"] = config.evaluation.sparse_tracking_threshold
    proposal_summary = proposal_summary_markdown(
        dataset_summary_payload,
        best_models_frame,
        motion_lift_frame,
        tracking_response_lift_frame,
        motion_effect_frame=motion_effect_frame,
        defensive_reaction_frame=defensive_reaction_frame,
    )

    overall_path = write_frame(
        overall_metrics,
        _artifacts_dir(config, "metrics") / "overall_metrics.csv",
    )
    validation_path = write_frame(
        validation_metrics,
        _artifacts_dir(config, "metrics") / "validation_metrics.csv",
    )
    subgroup_path = write_frame(
        subgroup_metrics_frame,
        _artifacts_dir(config, "metrics") / "subgroup_metrics.csv",
    )
    manifest_path = write_json(
        {"models": model_manifest},
        _artifacts_dir(config, "metrics") / "model_manifest.json",
    )
    season_summary_path = write_frame(
        season_summary_frame,
        _artifacts_dir(config, "metrics") / "season_summary.csv",
    )
    best_models_path = write_frame(
        best_models_frame,
        _artifacts_dir(config, "metrics") / "best_models.csv",
    )
    motion_lift_path = write_frame(
        motion_lift_frame,
        _artifacts_dir(config, "metrics") / "motion_lift_overall.csv",
    )
    motion_lift_subgroup_path = write_frame(
        motion_lift_subgroup_frame,
        _artifacts_dir(config, "metrics") / "motion_lift_subgroups.csv",
    )
    tracking_response_lift_path = write_frame(
        tracking_response_lift_frame,
        _artifacts_dir(config, "metrics") / "tracking_response_lift_overall.csv",
    )
    tracking_response_lift_subgroup_path = write_frame(
        tracking_response_lift_subgroup_frame,
        _artifacts_dir(config, "metrics") / "tracking_response_lift_subgroups.csv",
    )
    motion_effect_path = write_frame(
        motion_effect_frame,
        _artifacts_dir(config, "metrics") / "motion_effect_overall.csv",
    )
    motion_effect_subgroup_path = write_frame(
        motion_effect_subgroup_frame,
        _artifacts_dir(config, "metrics") / "motion_effect_subgroups.csv",
    )
    defensive_reaction_path = write_frame(
        defensive_reaction_frame,
        _artifacts_dir(config, "metrics") / "defensive_reaction_overall.csv",
    )
    defensive_reaction_subgroup_path = write_frame(
        defensive_reaction_subgroup_frame,
        _artifacts_dir(config, "metrics") / "defensive_reaction_subgroups.csv",
    )
    dataset_summary_path = write_json(
        dataset_summary_payload,
        _artifacts_dir(config, "metrics") / "dataset_summary.json",
    )
    proposal_summary_path = write_text(
        proposal_summary,
        _artifacts_dir(config, "metrics") / "proposal_summary.md",
    )

    return {
        "overall_metrics": overall_path,
        "validation_metrics": validation_path,
        "subgroup_metrics": subgroup_path,
        "model_manifest": manifest_path,
        "season_summary": season_summary_path,
        "best_models": best_models_path,
        "motion_lift_overall": motion_lift_path,
        "motion_lift_subgroups": motion_lift_subgroup_path,
        "tracking_response_lift_overall": tracking_response_lift_path,
        "tracking_response_lift_subgroups": tracking_response_lift_subgroup_path,
        "motion_effect_overall": motion_effect_path,
        "motion_effect_subgroups": motion_effect_subgroup_path,
        "defensive_reaction_overall": defensive_reaction_path,
        "defensive_reaction_subgroups": defensive_reaction_subgroup_path,
        "dataset_summary": dataset_summary_path,
        "proposal_summary": proposal_summary_path,
    }
