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


def _humanize(value: str) -> str:
    return value.replace("_", " ").title()


def _style_axis(ax: plt.Axes, *, y_grid: bool = True, x_grid: bool = False) -> None:
    ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", alpha=0.9)
    if x_grid:
        ax.grid(axis="x", alpha=0.9)
    ax.spines["left"].set_color(PALETTE["gray"])
    ax.spines["bottom"].set_color(PALETTE["gray"])


def _add_subtitle(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.955, text, ha="center", va="top", fontsize=10, color=PALETTE["gray"])


def _save(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _target_order(frame: pd.DataFrame) -> list[str]:
    preferred = ["completion", "explosive", "success", "epa"]
    available = frame["target"].dropna().unique().tolist() if "target" in frame.columns else []
    ordered = [target for target in preferred if target in available]
    return ordered + [target for target in available if target not in ordered]


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

    fig.suptitle("Final Selected Holdout Winners", fontsize=16, fontweight="bold", y=0.985)
    _add_subtitle(fig, "Best-performing models on the held-out 2024 test split")
    return _save(fig, output_dir / "01_selected_models.png")


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
    lower_errors = (rows["adjusted_effect"] - rows["effect_ci_lower"]).abs()
    upper_errors = (rows["effect_ci_upper"] - rows["adjusted_effect"]).abs()
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
    return _save(fig, output_dir / "02_motion_effect_overall.png")


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


def plot_tracking_response_lift(frame: pd.DataFrame, output_dir: Path) -> Path | None:
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
    ax.set_title("Tracking Response Lift: Full vs Context Plus Motion")
    ax.legend(frameon=False, title="Model")
    return _save(fig, output_dir / "04_tracking_response_lift.png")


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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    split_labels = ["Train", "Test"]
    tracking_values = [
        float(dataset_summary.get("train_tracking_coverage_rate", 0.0)),
        float(dataset_summary.get("test_tracking_coverage_rate", 0.0)),
    ]
    axes[0].bar(split_labels, tracking_values, color=[PALETTE["teal"], PALETTE["coral"]])
    _style_axis(axes[0])
    axes[0].set_title("Tracking Coverage by Split")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for idx, value in enumerate(tracking_values):
        axes[0].text(idx, value + 0.015, f"{value:.1%}", ha="center", va="bottom")

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

    fig.suptitle("Dataset Snapshot", fontsize=16, fontweight="bold", y=0.985)
    _add_subtitle(fig, "Final experiment sample sizes and the train-vs-test tracking coverage gap")
    return _save(fig, output_dir / "06_dataset_snapshot.png")


def plot_validation_vs_test(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    rows = frame.loc[frame["evaluation_slice"] == "all"].copy()
    if rows.empty:
        rows = frame.copy()
    rows = rows.loc[rows["task"] == "classification"].copy()
    if rows.empty or "validation_selection_value" not in rows.columns:
        return None
    rows["target"] = pd.Categorical(
        rows["target"],
        categories=[target for target in _target_order(rows) if target != "epa"],
        ordered=True,
    )
    rows = rows.sort_values("target")

    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = range(len(rows))
    ax.plot(x_positions, rows["validation_selection_value"], marker="o", linewidth=2, color=PALETTE["navy"], label="Validation selection metric")
    ax.plot(x_positions, rows["balanced_accuracy"], marker="o", linewidth=2, color=PALETTE["gold"], label="Test balanced accuracy")
    _style_axis(ax)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(rows["target"].str.title())
    ax.set_ylabel("Score")
    ax.set_title("Validation-Selected Winners on Test Data")
    ax.legend(frameon=False)
    return _save(fig, output_dir / "07_validation_vs_test_selected_models.png")


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
    rows["label"] = rows["group_column"].astype(str).map(_humanize) + ": " + rows["group_value"].astype(str).map(_humanize)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rows["label"], rows["adjusted_effect"], color=PALETTE["sage"])
    ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax, y_grid=False, x_grid=True)
    ax.set_title("Top Completion Motion-Effect Subgroups")
    ax.set_xlabel("Adjusted Effect")
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
    rows["label"] = rows["group_column"].astype(str).map(_humanize) + ": " + rows["group_value"].astype(str).map(_humanize)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rows["label"], rows["improvement_auroc"], color=PALETTE["navy"])
    ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=1)
    _style_axis(ax, y_grid=False, x_grid=True)
    ax.set_title("Top Completion Lift Subgroups")
    ax.set_xlabel("AUROC Improvement")
    ax.invert_yaxis()
    return _save(fig, output_dir / "09_completion_motion_lift_subgroups.png")


def plot_target_rates(season_summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if season_summary.empty:
        return None
    rows = season_summary.copy().sort_values("season")
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, color, label in [
        ("completion_mean", PALETTE["navy"], "Completion"),
        ("success_mean", PALETTE["gold"], "Success"),
        ("explosive_mean", PALETTE["teal"], "Explosive"),
    ]:
        if column in rows.columns:
            ax.plot(rows["season"].astype(str), rows[column], marker="o", linewidth=2, color=color, label=label)
    _style_axis(ax)
    ax.set_title("Target Rates by Season")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(frameon=False)
    return _save(fig, output_dir / "10_target_rates_by_season.png")


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
        + rows["model_name"].str.replace("_", " ")
        + " | "
        + rows["feature_set"].str.replace("_", " ")
    )
    rows = rows.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(rows["label"], rows["score"], color=PALETTE["teal"])
    _style_axis(ax, y_grid=False, x_grid=True)
    ax.set_title("Classification Leaderboard on the Full Test Slice")
    ax.set_xlabel("Balanced Accuracy")
    return _save(fig, output_dir / "11_classification_leaderboard.png")
