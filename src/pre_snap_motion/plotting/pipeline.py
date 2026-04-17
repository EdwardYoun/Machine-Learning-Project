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


def _write_chart_manifest(chart_paths: list[Path], output_dir: Path) -> Path:
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
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def generate_all_graphs(
    metrics_dir: str | Path = "artifacts/metrics",
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    inputs = load_plot_inputs(metrics_dir)
    output_path = ensure_output_dir(output_dir)

    chart_paths = [
        plot_selected_models(inputs.selected_models, output_path),
        plot_motion_effect(inputs.motion_effect_overall, output_path),
        plot_motion_lift(inputs.motion_lift_overall, output_path),
        plot_tracking_response_lift(inputs.tracking_response_lift_overall, output_path),
        plot_tracking_coverage(inputs.season_summary, output_path),
        plot_dataset_snapshot(inputs.dataset_summary, output_path),
        plot_validation_vs_test(inputs.selected_models, output_path),
        plot_subgroup_motion_effect(inputs.motion_effect_subgroups, output_path),
        plot_subgroup_motion_lift(inputs.motion_lift_subgroups, output_path),
        plot_target_rates(inputs.season_summary, output_path),
        plot_overall_model_leaderboard(inputs.overall_metrics, output_path),
    ]
    rendered = [path for path in chart_paths if path is not None]
    rendered.append(_write_chart_manifest(rendered, output_path))
    return rendered
