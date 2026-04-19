from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.color": "#D7E0E7",
        "grid.linewidth": 0.8,
        "font.size": 11,
    }
)


PALETTE = {
    "navy": "#12355B",
    "teal": "#2A7F9E",
    "gold": "#F4B942",
    "coral": "#EE6C4D",
    "sage": "#7D9D78",
    "gray": "#617073",
    "light": "#E9F1F7",
}

GROUP_LABELS = {
    "box_bucket": "Box Bucket",
    "distance_bucket": "Distance Bucket",
    "down_bucket": "Down Bucket",
    "field_zone": "Field Zone",
    "pressure_bucket": "Pressure Bucket",
    "qb_location": "QB Location",
    "score_state": "Score State",
    "starting_hash": "Starting Hash",
}

GROUP_VALUE_LABELS = {
    "qb_location": {
        "0": "Unknown",
        "P": "Pocket",
        "S": "Shotgun",
        "U": "Under Center",
    },
    "starting_hash": {
        "0": "Unknown",
        "L": "Left",
        "M": "Middle",
        "R": "Right",
    },
}

MODEL_SHORT_NAMES = {
    "gradient_boosting": "GB",
    "logistic_regression": "LR",
    "ridge_regression": "Ridge",
}

FEATURE_SET_SHORT_NAMES = {
    "context_only": "Context",
    "context_plus_motion": "Motion",
    "full": "Full",
}


def _humanize(value: str) -> str:
    return value.replace("_", " ").title()


def _group_label(column_name: str, group_value: object) -> str:
    column_label = GROUP_LABELS.get(column_name, _humanize(column_name))
    value_key = str(group_value)
    value_label = GROUP_VALUE_LABELS.get(column_name, {}).get(value_key, _humanize(value_key))
    return f"{column_label}: {value_label}"


def _metric_label(metric_name: str) -> str:
    labels = {
        "auroc": "AUROC",
        "balanced_accuracy": "Balanced Accuracy",
        "expected_calibration_error": "Expected Calibration Error",
        "f1": "F1",
        "log_loss": "Log Loss",
        "mae": "MAE",
        "rmse": "RMSE",
    }
    return labels.get(metric_name, _humanize(metric_name))


def _short_model_name(model_name: str) -> str:
    return MODEL_SHORT_NAMES.get(model_name, _humanize(model_name))


def _short_feature_set(feature_set: str) -> str:
    return FEATURE_SET_SHORT_NAMES.get(feature_set, _humanize(feature_set))


def _style_axis(ax: plt.Axes, *, y_grid: bool = True, x_grid: bool = False) -> None:
    ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", alpha=0.9)
    if x_grid:
        ax.grid(axis="x", alpha=0.9)
    ax.spines["left"].set_color(PALETTE["gray"])
    ax.spines["bottom"].set_color(PALETTE["gray"])


def _add_subtitle(fig: plt.Figure, text: str, *, y: float = 0.935) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=9.5, color=PALETTE["gray"])


def _save(fig: plt.Figure, output_path: Path, *, layout_top: float = 0.88) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, layout_top))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _target_order(frame: pd.DataFrame) -> list[str]:
    preferred = ["completion", "explosive", "success", "epa"]
    available = frame["target"].dropna().unique().tolist() if "target" in frame.columns else []
    ordered = [target for target in preferred if target in available]
    return ordered + [target for target in available if target not in ordered]


def _selection_metric_value(row: pd.Series) -> float | None:
    for column in ["selection_value", "validation_selection_value"]:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    metric_name = row.get("selection_metric")
    if metric_name in row.index and pd.notna(row[metric_name]):
        return float(row[metric_name])
    return None


def plot_selected_models(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None

    rows = frame.loc[frame["evaluation_slice"] == "all"].copy()
    if rows.empty:
        rows = frame.copy()
    rows["display_value"] = rows["balanced_accuracy"].fillna(rows["rmse"])
    rows["metric_label"] = rows["balanced_accuracy"].apply(
        lambda value: "Balanced Accuracy" if pd.notna(value) else "RMSE"
    )
    rows["label"] = rows["target"].str.title() + "\n" + rows["feature_set"].str.replace("_", " ")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

    classification = rows.loc[rows["task"] == "classification"].copy()
    classification["target"] = pd.Categorical(
        classification["target"],
        categories=[target for target in _target_order(classification) if target != "epa"],
        ordered=True,
    )
    classification = classification.sort_values("target")
    axes[0].bar(
        classification["target"].str.title(),
        classification["balanced_accuracy"],
        color=[PALETTE["navy"], PALETTE["gold"], PALETTE["teal"]],
    )
    _style_axis(axes[0])
    axes[0].set_title("Best Classification Models")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_ylim(0.45, max(0.58, classification["balanced_accuracy"].max() + 0.03))
    for idx, (_, row) in enumerate(classification.iterrows()):
        axes[0].text(
            idx,
            row["balanced_accuracy"] + 0.005,
            f"{row['model_name']}\n{row['feature_set']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    regression = rows.loc[rows["task"] == "regression"].copy()
    axes[1].bar(
        regression["target"].str.upper(),
        regression["rmse"],
        color=PALETTE["coral"],
    )
    _style_axis(axes[1])
    axes[1].set_title("Best Regression Model")
    axes[1].set_ylabel("RMSE")
    if not regression.empty:
        axes[1].set_ylim(0, regression["rmse"].max() + 0.4)
        for idx, (_, row) in enumerate(regression.iterrows()):
            axes[1].text(
                idx,
                row["rmse"] + 0.03,
                f"{row['model_name']}\n{row['feature_set']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Final Selected Holdout Winners", fontsize=16, fontweight="bold", y=0.99)
    _add_subtitle(fig, "Best-performing models on the held-out 2024 test split")
    return _save(fig, output_dir / "01_selected_models.png", layout_top=0.8)


def plot_motion_effect(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[frame["dataset_split"] == "test"].copy()
    if rows.empty:
        rows = frame.copy()
    rows["target"] = pd.Categorical(rows["target"], categories=_target_order(rows), ordered=True)
    rows = rows.sort_values("target")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        PALETTE["sage"] if direction == "helps" else PALETTE["gray"]
        for direction in rows["effect_direction"]
    ]
    ci_lower = rows[["effect_ci_lower", "adjusted_effect"]].min(axis=1)
    ci_upper = rows[["effect_ci_upper", "adjusted_effect"]].max(axis=1)
    lower_errors = (rows["adjusted_effect"] - ci_lower).abs()
    upper_errors = (ci_upper - rows["adjusted_effect"]).abs()
    errors = [lower_errors, upper_errors]
    ax.errorbar(
        rows["target"].str.title(),
        rows["adjusted_effect"],
        yerr=errors,
        fmt="none",
        ecolor=PALETTE["gray"],
        elinewidth=2,
        capsize=6,
    )
    ax.scatter(rows["target"].str.title(), rows["adjusted_effect"], s=120, c=colors, zorder=3)
    ax.axhline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax)
    ax.set_title("Overall Motion Effect by Target")
    ax.set_ylabel("Adjusted Effect")
    for idx, (_, row) in enumerate(rows.iterrows()):
        ax.text(
            idx,
            row["adjusted_effect"] + (0.003 if row["adjusted_effect"] >= 0 else -0.006),
            f"{row['effect_direction']}\n{row['adjusted_effect']:+.3f}",
            ha="center",
            va="bottom" if row["adjusted_effect"] >= 0 else "top",
            fontsize=9,
        )
    fig.suptitle("Overall Motion Effect by Target", fontsize=16, fontweight="bold", y=0.99)
    _add_subtitle(fig, "Point estimates with 95% confidence intervals after context controls")
    ax.set_title("")
    return _save(fig, output_dir / "02_motion_effect_overall.png", layout_top=0.8)


def _lift_value_column(frame: pd.DataFrame) -> str:
    if "improvement_auroc" in frame.columns and frame["improvement_auroc"].notna().any():
        return "improvement_auroc"
    return "improvement_rmse"


def plot_motion_lift(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[frame["evaluation_slice"] == "all"].copy()
    if rows.empty:
        rows = frame.copy()
    classification = rows.loc[rows["task"] == "classification"].copy()
    if classification.empty:
        return None
    classification["target"] = pd.Categorical(
        classification["target"],
        categories=[target for target in _target_order(classification) if target != "epa"],
        ordered=True,
    )
    classification = classification.sort_values("target")

    fig, ax = plt.subplots(figsize=(9, 5))
    model_names = classification["model_name"].unique().tolist()
    width = 0.35 if len(model_names) > 1 else 0.55
    x_positions = range(len(classification["target"].cat.categories))

    for index, model_name in enumerate(model_names):
        model_rows = classification.loc[classification["model_name"] == model_name]
        offset = (index - (len(model_names) - 1) / 2) * width
        positions = [x + offset for x in x_positions]
        ax.bar(
            positions,
            model_rows["improvement_auroc"],
            width=width,
            label=model_name.replace("_", " ").title(),
            color=PALETTE["navy"] if index == 0 else PALETTE["gold"],
        )

    ax.axhline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([target.title() for target in classification["target"].cat.categories])
    ax.set_ylabel("AUROC Improvement")
    ax.set_title("Motion Feature Lift: Context Plus Motion vs Context Only")
    ax.legend(frameon=False, title="Model")
    return _save(fig, output_dir / "03_motion_lift_classification.png")


def plot_tracking_response_lift(
    frame: pd.DataFrame,
    dataset_summary: dict[str, object],
    output_dir: Path,
) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[frame["evaluation_slice"] == "all"].copy()
    if rows.empty:
        rows = frame.copy()
    classification = rows.loc[rows["task"] == "classification"].copy()
    if classification.empty:
        return None
    classification["target"] = pd.Categorical(
        classification["target"],
        categories=[target for target in _target_order(classification) if target != "epa"],
        ordered=True,
    )
    classification = classification.sort_values("target")

    fig, ax = plt.subplots(figsize=(9, 5))
    model_names = classification["model_name"].unique().tolist()
    width = 0.35 if len(model_names) > 1 else 0.55
    x_positions = range(len(classification["target"].cat.categories))

    for index, model_name in enumerate(model_names):
        model_rows = classification.loc[classification["model_name"] == model_name]
        offset = (index - (len(model_names) - 1) / 2) * width
        positions = [x + offset for x in x_positions]
        ax.bar(
            positions,
            model_rows["improvement_auroc"],
            width=width,
            label=model_name.replace("_", " ").title(),
            color=PALETTE["teal"] if index == 0 else PALETTE["coral"],
        )

    ax.axhline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([target.title() for target in classification["target"].cat.categories])
    ax.set_ylabel("AUROC Improvement")
    subtitle = "Full vs. context-plus-motion lift on the evaluated test slice"
    test_tracking_coverage = float(dataset_summary.get("test_tracking_coverage_rate", 0.0))
    sparse_threshold = float(dataset_summary.get("sparse_tracking_threshold", 0.25))
    test_tracking_rows = int(dataset_summary.get("test_tracking_rows", 0))
    if test_tracking_coverage and test_tracking_coverage < sparse_threshold:
        subtitle = (
            f"Directional only: just {test_tracking_rows:,} tracked test plays "
            f"({test_tracking_coverage:.1%} coverage)"
        )
    fig.suptitle("Tracking Response Lift", fontsize=16, fontweight="bold", y=0.99)
    _add_subtitle(fig, subtitle)
    ax.set_title("")
    ax.legend(frameon=False, title="Model")
    return _save(fig, output_dir / "04_tracking_response_lift.png", layout_top=0.8)


def plot_tracking_coverage(season_summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if season_summary.empty:
        return None
    rows = season_summary.copy().sort_values("season")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(rows["season"].astype(str), rows["tracking_coverage_rate"], color=[PALETTE["navy"], PALETTE["gold"]])
    _style_axis(ax)
    ax.set_title("Tracking Coverage by Season")
    ax.set_ylabel("Coverage Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for idx, (_, row) in enumerate(rows.iterrows()):
        ax.text(
            idx,
            row["tracking_coverage_rate"] + 0.015,
            f"{row['tracking_coverage_rate']:.1%}\n{int(row['rows']):,} plays",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _save(fig, output_dir / "05_tracking_coverage_by_season.png")


def plot_dataset_snapshot(dataset_summary: dict[str, object], output_dir: Path) -> Path | None:
    if not dataset_summary:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    split_labels = ["Train", "Test"]
    tracking_values = [
        float(dataset_summary.get("train_tracking_coverage_rate", 0.0)),
        float(dataset_summary.get("test_tracking_coverage_rate", 0.0)),
    ]
    axes[0].bar(split_labels, tracking_values, color=[PALETTE["teal"], PALETTE["coral"]])
    _style_axis(axes[0])
    axes[0].set_title("Tracking Coverage by Split")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    tracking_upper = max(0.05, max(tracking_values) + 0.05)
    axes[0].set_ylim(0, tracking_upper)
    tracking_label_offset = max(0.004, tracking_upper * 0.02)
    for idx, value in enumerate(tracking_values):
        axes[0].text(
            idx,
            min(value + tracking_label_offset, tracking_upper - tracking_label_offset),
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )

    totals = [
        int(dataset_summary.get("train_rows", 0)),
        int(dataset_summary.get("test_rows", 0)),
    ]
    axes[1].bar(split_labels, totals, color=[PALETTE["navy"], PALETTE["gold"]])
    _style_axis(axes[1])
    axes[1].set_title("Rows by Split")
    axes[1].set_ylabel("Plays")
    for idx, value in enumerate(totals):
        axes[1].text(idx, value + max(totals) * 0.02, f"{value:,}", ha="center", va="bottom")

    fig.suptitle("Dataset Snapshot", fontsize=16, fontweight="bold", y=0.99)
    _add_subtitle(fig, "Train/test sample sizes and the tracking coverage gap")
    return _save(fig, output_dir / "06_dataset_snapshot.png", layout_top=0.84)


def plot_validation_vs_test(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[
        (frame["evaluation_slice"] == "all")
        & (frame["task"] == "classification")
    ].copy()
    if rows.empty:
        rows = frame.copy()
    if rows.empty:
        return None

    if "selection_metric" not in rows.columns or "selection_split" not in rows.columns:
        return None
    rows["selection_split"] = rows["selection_split"].astype(str).str.lower()
    rows = rows.loc[rows["selection_split"] == "validation"].copy()
    if rows.empty:
        return None
    rows["selection_value_resolved"] = rows.apply(
        lambda row: _selection_metric_value(row),
        axis=1,
    )
    rows = rows.loc[rows["selection_value_resolved"].notna()].copy()
    if rows.empty:
        return None
    rows["target"] = pd.Categorical(
        rows["target"],
        categories=[target for target in _target_order(rows) if target != "epa"],
        ordered=True,
    )
    rows = rows.sort_values("target")
    metric_name = (
        str(rows["selection_metric"].iloc[0])
        if rows["selection_metric"].nunique() == 1
        else "selection_metric"
    )
    metric_label = (
        _metric_label(metric_name)
        if metric_name != "selection_metric"
        else "Selection Score"
    )
    rows["test_metric_value"] = rows.apply(
        lambda row: row[row["selection_metric"]] if row["selection_metric"] in row.index else pd.NA,
        axis=1,
    )
    if rows["test_metric_value"].isna().any():
        fallback_column = "balanced_accuracy" if "balanced_accuracy" in rows.columns else None
        if fallback_column is None:
            return None
        rows["test_metric_value"] = rows[fallback_column]

    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = list(range(len(rows)))

    ax.plot(
        x_positions,
        rows["selection_value_resolved"],
        marker="o",
        linewidth=2,
        color=PALETTE["navy"],
        label=f"Selection {metric_label}",
        zorder=3,
    )
    ax.plot(
        x_positions,
        rows["test_metric_value"],
        marker="o",
        linewidth=2,
        color=PALETTE["gold"],
        label=f"Test {metric_label}",
        zorder=3,
    )
    _style_axis(ax)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(rows["target"].str.title())
    ax.set_xlim(min(x_positions) - 0.3, max(x_positions) + 0.3)
    y_min = min(float(rows["selection_value_resolved"].min()), float(rows["test_metric_value"].min()))
    y_max = max(float(rows["selection_value_resolved"].max()), float(rows["test_metric_value"].max()))
    ax.set_ylim(max(0.48, y_min - 0.01), max(0.56, y_max + 0.015))
    ax.set_ylabel(metric_label)
    for idx, (_, row) in enumerate(rows.iterrows()):
        top_value = max(float(row["selection_value_resolved"]), float(row["test_metric_value"]))
        ax.text(
            x_positions[idx],
            top_value + 0.003,
            f"{_short_model_name(str(row['model_name']))} | {_short_feature_set(str(row['feature_set']))}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle("Validation Selection Metric vs Test Performance", fontsize=16, fontweight="bold", y=0.99)
    _add_subtitle(fig, "Blue shows the selection metric used to choose each winner; gold shows the held-out test score")
    ax.set_title("")
    ax.legend(frameon=False)
    return _save(fig, output_dir / "07_validation_vs_test_selected_models.png", layout_top=0.8)


def plot_subgroup_motion_effect(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[
        (frame["dataset_split"] == "test")
        & (frame["target"] == "completion")
    ].copy()
    if rows.empty:
        return None
    rows = rows.sort_values("adjusted_effect", ascending=False).head(8)
    rows["label"] = rows.apply(
        lambda row: _group_label(str(row["group_column"]), row["group_value"]),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rows["label"], rows["adjusted_effect"], color=PALETTE["sage"])
    ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax, y_grid=False, x_grid=True)
    ax.set_title("Top Completion Motion-Effect Subgroups")
    ax.set_xlabel("Adjusted Effect")
    x_min = min(0.0, float(rows["adjusted_effect"].min())) - 0.01
    x_max = float(rows["adjusted_effect"].max()) + 0.02
    ax.set_xlim(x_min, x_max)
    for patch, (_, row) in zip(ax.patches, rows.iterrows()):
        value = float(row["adjusted_effect"])
        ax.text(
            value + 0.001,
            patch.get_y() + patch.get_height() / 2,
            f"{value:+.3f} | n={int(row['n_obs']):,}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.invert_yaxis()
    return _save(fig, output_dir / "08_completion_motion_effect_subgroups.png")


def plot_subgroup_motion_lift(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[
        (frame["evaluation_slice"] == "all")
        & (frame["task"] == "classification")
        & (frame["target"] == "completion")
        & (frame["model_name"] == "logistic_regression")
    ].copy()
    if rows.empty:
        rows = frame.loc[
            (frame["task"] == "classification") & (frame["target"] == "completion")
        ].copy()
    if rows.empty:
        return None
    rows = rows.sort_values("improvement_auroc", ascending=False).head(10)
    rows["label"] = rows.apply(
        lambda row: _group_label(str(row["group_column"]), row["group_value"]),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rows["label"], rows["improvement_auroc"], color=PALETTE["navy"])
    ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax, y_grid=False, x_grid=True)
    ax.set_title("Top Completion Lift Subgroups")
    ax.set_xlabel("AUROC Improvement")
    x_min = min(0.0, float(rows["improvement_auroc"].min())) - 0.01
    x_max = float(rows["improvement_auroc"].max()) + 0.02
    ax.set_xlim(x_min, x_max)
    for patch, (_, row) in zip(ax.patches, rows.iterrows()):
        value = float(row["improvement_auroc"])
        sample_size = int(row.get("n_obs_context_plus_motion", 0) or 0)
        ax.text(
            value + 0.001,
            patch.get_y() + patch.get_height() / 2,
            f"{value:+.3f} | n={sample_size:,}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.invert_yaxis()
    return _save(fig, output_dir / "09_completion_motion_lift_subgroups.png")


def plot_target_rates(season_summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if season_summary.empty:
        return None
    rows = season_summary.copy().sort_values("season")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1.2]})
    ax = axes[0]
    classification_offsets = {
        "completion_mean": 0.010,
        "success_mean": 0.008,
        "explosive_mean": 0.006,
    }
    classification_max = 0.0
    classification_min = 1.0
    for column, color, label in [
        ("completion_mean", PALETTE["navy"], "Completion"),
        ("success_mean", PALETTE["gold"], "Success"),
        ("explosive_mean", PALETTE["teal"], "Explosive"),
    ]:
        if column in rows.columns:
            values = rows[column].astype(float)
            classification_max = max(classification_max, float(values.max()))
            classification_min = min(classification_min, float(values.min()))
            ax.plot(
                rows["season"].astype(str),
                values,
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )
            for idx, value in enumerate(values):
                ax.text(
                    idx,
                    value + classification_offsets.get(column, 0.007),
                    f"{value:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=color,
                )
    _style_axis(ax)
    ax.set_title("Classification Target Rates")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    if classification_max > 0:
        ax.set_ylim(max(0, classification_min - 0.02), min(1, classification_max + 0.05))
    ax.legend(frameon=False)
    epa_ax = axes[1]
    if "epa_mean" in rows.columns:
        epa_ax.bar(
            rows["season"].astype(str),
            rows["epa_mean"],
            color=[PALETTE["navy"], PALETTE["gold"]],
        )
        epa_ax.axhline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
        _style_axis(epa_ax)
        epa_ax.set_title("EPA Mean")
        epa_ax.set_ylabel("EPA / play")
        for idx, (_, row) in enumerate(rows.iterrows()):
            value = float(row["epa_mean"])
            offset = 0.005 if value >= 0 else -0.008
            epa_ax.text(
                idx,
                value + offset,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    else:
        epa_ax.set_visible(False)
    fig.suptitle("Target Patterns by Season", fontsize=16, fontweight="bold", y=0.99)
    _add_subtitle(fig, "Classification rates stay stable while average EPA shifts upward in 2024")
    return _save(fig, output_dir / "10_target_rates_by_season.png", layout_top=0.8)


def plot_overall_model_leaderboard(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[
        (frame["dataset_split"] == "test")
        & (frame["evaluation_slice"] == "all")
        & (frame["task"] == "classification")
    ].copy()
    if rows.empty:
        return None
    rows["score"] = rows["balanced_accuracy"]
    rows["label"] = (
        rows["target"].str.title()
        + " | "
        + rows["model_name"].map(_short_model_name)
        + " | "
        + rows["feature_set"].map(_short_feature_set)
    )
    rows["target"] = pd.Categorical(
        rows["target"],
        categories=[target for target in _target_order(rows) if target != "epa"],
        ordered=True,
    )
    rows["rank_within_target"] = (
        rows.groupby("target", observed=False)["score"].rank(method="first", ascending=False)
    )
    rows = rows.loc[rows["rank_within_target"] <= 2].copy()
    rows = rows.sort_values(["target", "score"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [
        PALETTE["navy"] if rank == 1 else PALETTE["light"]
        for rank in rows["rank_within_target"]
    ]
    ax.barh(rows["label"], rows["score"], color=colors, edgecolor=PALETTE["teal"])
    _style_axis(ax, y_grid=False, x_grid=True)
    ax.axvline(0.5, color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax.set_title("Top Two Variants per Classification Target")
    ax.set_xlabel("Balanced Accuracy")
    ax.set_xlim(0, max(0.58, float(rows["score"].max()) + 0.02))
    for patch, (_, row) in zip(ax.patches, rows.iterrows()):
        ax.text(
            float(row["score"]) + 0.003,
            patch.get_y() + patch.get_height() / 2,
            f"{float(row['score']):.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    return _save(fig, output_dir / "11_classification_leaderboard.png")
