from __future__ import annotations

from pathlib import Path

from pre_snap_motion.plotting.charts import (
    plot_dataset_snapshot,
    plot_motion_effect,
    plot_motion_lift,
    plot_overall_model_leaderboard,
    plot_selected_models,
    plot_subgroup_motion_effect,
    plot_subgroup_motion_lift,
    plot_target_rates,
    plot_tracking_coverage,
    plot_tracking_response_lift,
    plot_validation_vs_test,
)
from pre_snap_motion.plotting.io import ensure_output_dir, load_plot_inputs

CHART_FILENAMES = [
    "01_selected_models.png",
    "02_motion_effect_overall.png",
    "03_motion_lift_classification.png",
    "04_tracking_response_lift.png",
    "05_tracking_coverage_by_season.png",
    "06_dataset_snapshot.png",
    "07_validation_vs_test_selected_models.png",
    "08_completion_motion_effect_subgroups.png",
    "09_completion_motion_lift_subgroups.png",
    "10_target_rates_by_season.png",
    "11_classification_leaderboard.png",
]


def _clear_previous_chart_outputs(output_dir: Path) -> None:
    for file_name in CHART_FILENAMES + ["chart_manifest.md"]:
        output_path = output_dir / file_name
        if output_path.exists():
            output_path.unlink()


def _write_chart_manifest(
    chart_paths: list[Path],
    output_dir: Path,
    notes: list[str] | None = None,
) -> Path:
    manifest_path = output_dir / "chart_manifest.md"
    lines = [
        "# Chart Manifest",
        "",
        "Generated from `artifacts/metrics` using `scripts/generate_report_graphs.py`.",
        "",
        "## Charts",
        "",
    ]
    for chart_path in chart_paths:
        lines.append(f"- `{chart_path.name}`")
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def generate_all_graphs(
    metrics_dir: str | Path = "artifacts/metrics",
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    inputs = load_plot_inputs(metrics_dir)
    output_path = ensure_output_dir(output_dir)
    _clear_previous_chart_outputs(output_path)

    validation_vs_test_path = plot_validation_vs_test(inputs.selected_models, output_path)
    chart_paths = [
        plot_selected_models(inputs.selected_models, output_path),
        plot_motion_effect(inputs.motion_effect_overall, output_path),
        plot_motion_lift(inputs.motion_lift_overall, output_path),
        plot_tracking_response_lift(
            inputs.tracking_response_lift_overall,
            inputs.dataset_summary,
            output_path,
        ),
        plot_tracking_coverage(inputs.season_summary, output_path),
        plot_dataset_snapshot(inputs.dataset_summary, output_path),
        validation_vs_test_path,
        plot_subgroup_motion_effect(inputs.motion_effect_subgroups, output_path),
        plot_subgroup_motion_lift(inputs.motion_lift_subgroups, output_path),
        plot_target_rates(inputs.season_summary, output_path),
        plot_overall_model_leaderboard(inputs.overall_metrics, output_path),
    ]
    rendered = [path for path in chart_paths if path is not None]
    notes: list[str] = []
    if validation_vs_test_path is None:
        notes.append(
            "`07_validation_vs_test_selected_models.png` omitted because the current metrics "
            "do not include a separate validation selection split."
        )
    rendered.append(_write_chart_manifest(rendered, output_path, notes))
    return rendered
