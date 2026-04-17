from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class PlotInputs:
    selected_models: pd.DataFrame
    motion_effect_overall: pd.DataFrame
    motion_lift_overall: pd.DataFrame
    tracking_response_lift_overall: pd.DataFrame
    season_summary: pd.DataFrame
    dataset_summary: dict[str, object]
    validation_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    motion_effect_subgroups: pd.DataFrame
    motion_lift_subgroups: pd.DataFrame
    tracking_response_lift_subgroups: pd.DataFrame


def metrics_dir_from(path: str | Path) -> Path:
    metrics_dir = Path(path)
    if metrics_dir.is_file():
        return metrics_dir.parent
    return metrics_dir


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_plot_inputs(metrics_dir: str | Path) -> PlotInputs:
    metrics_path = metrics_dir_from(metrics_dir)
    dataset_summary_path = metrics_path / "dataset_summary.json"
    dataset_summary = (
        json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        if dataset_summary_path.exists()
        else {}
    )
    return PlotInputs(
        selected_models=_read_csv(metrics_path / "selected_models.csv"),
        motion_effect_overall=_read_csv(metrics_path / "motion_effect_overall.csv"),
        motion_lift_overall=_read_csv(metrics_path / "motion_lift_overall.csv"),
        tracking_response_lift_overall=_read_csv(
            metrics_path / "tracking_response_lift_overall.csv"
        ),
        season_summary=_read_csv(metrics_path / "season_summary.csv"),
        dataset_summary=dataset_summary,
        validation_metrics=_read_csv(metrics_path / "validation_metrics.csv"),
        overall_metrics=_read_csv(metrics_path / "overall_metrics.csv"),
        subgroup_metrics=_read_csv(metrics_path / "subgroup_metrics.csv"),
        motion_effect_subgroups=_read_csv(metrics_path / "motion_effect_subgroups.csv"),
        motion_lift_subgroups=_read_csv(metrics_path / "motion_lift_subgroups.csv"),
        tracking_response_lift_subgroups=_read_csv(
            metrics_path / "tracking_response_lift_subgroups.csv"
        ),
    )
