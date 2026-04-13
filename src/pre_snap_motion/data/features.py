from __future__ import annotations

from pathlib import Path

import polars as pl

from pre_snap_motion.config import ProjectConfig


TARGET_COLUMNS = [
    "target_success",
    "target_epa",
    "target_explosive",
    "target_completion",
]

IDENTIFIER_COLUMNS = ["game_id", "play_id", "season", "week", "posteam", "defteam"]


def _existing(columns: list[str], candidates: list[str]) -> list[str]:
    available = set(columns)
    return [candidate for candidate in candidates if candidate in available]


def _unique_ordered(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def filter_pass_plays(pbp: pl.DataFrame, config: ProjectConfig) -> pl.DataFrame:
    filters = []

    if "posteam" in pbp.columns:
        filters.append(pl.col("posteam").is_not_null())
    if "epa" in pbp.columns:
        filters.append(pl.col("epa").is_not_null())
    if "two_point_attempt" in pbp.columns:
        filters.append(pl.col("two_point_attempt").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        filters.append(pl.col("qb_spike").fill_null(0) == 0)
    if "qb_kneel" in pbp.columns:
        filters.append(pl.col("qb_kneel").fill_null(0) == 0)

    if "season_type" in pbp.columns:
        filters.append(pl.col("season_type") == config.data.season_type)

    if "qb_dropback" in pbp.columns:
        pass_filter = pl.col("qb_dropback").fill_null(0) == 1
    else:
        pass_filter = pl.col("play_type").fill_null("").is_in(["pass", "no_play"])

    filtered = pbp.filter(pass_filter)
    for condition in filters:
        filtered = filtered.filter(condition)
    return filtered


def join_ftn_charting(pbp: pl.DataFrame, ftn: pl.DataFrame | None) -> pl.DataFrame:
    if ftn is None:
        return pbp

    pbp_normalized = pbp.with_columns(
        [
            pl.col("game_id").cast(pl.Utf8, strict=False),
            pl.col("play_id").cast(pl.Int64, strict=False),
        ]
    )
    rename_map = {}
    if "nflverse_game_id" in ftn.columns:
        rename_map["nflverse_game_id"] = "game_id"
    if "nflverse_play_id" in ftn.columns:
        rename_map["nflverse_play_id"] = "play_id"
    renamed = ftn.rename(rename_map) if rename_map else ftn
    normalized_ftn = renamed.with_columns(
        [
            pl.col("game_id").cast(pl.Utf8, strict=False),
            pl.col("play_id").cast(pl.Int64, strict=False),
        ]
    )
    deduped = normalized_ftn.unique(["game_id", "play_id"], keep="first")
    return pbp_normalized.join(deduped, on=["game_id", "play_id"], how="left")


def join_tracking_features(
    frame: pl.DataFrame, tracking: pl.DataFrame | None
) -> pl.DataFrame:
    if tracking is None:
        return frame

    frame_casts = [
        pl.col("game_id").cast(pl.Utf8, strict=False),
        pl.col("play_id").cast(pl.Int64, strict=False),
    ]
    if "old_game_id" in frame.columns:
        frame_casts.append(pl.col("old_game_id").cast(pl.Utf8, strict=False))
    normalized_frame = frame.with_columns(frame_casts)
    normalized_tracking = tracking.with_columns(
        [
            pl.col("game_id").cast(pl.Utf8, strict=False),
            pl.col("play_id").cast(pl.Int64, strict=False),
        ]
    )
    overlapping_columns = [
        column
        for column in normalized_tracking.columns
        if column in normalized_frame.columns and column not in {"game_id", "play_id"}
    ]
    if overlapping_columns:
        normalized_tracking = normalized_tracking.drop(overlapping_columns)
    deduped = normalized_tracking.unique(["game_id", "play_id"], keep="first")
    if "old_game_id" in normalized_frame.columns:
        return normalized_frame.join(
            deduped,
            left_on=["old_game_id", "play_id"],
            right_on=["game_id", "play_id"],
            how="left",
        )
    return normalized_frame.join(deduped, on=["game_id", "play_id"], how="left")


def engineer_features(joined: pl.DataFrame, config: ProjectConfig) -> pl.DataFrame:
    expressions: list[pl.Expr] = []

    if "score_differential" in joined.columns:
        expressions.extend(
            [
                pl.col("score_differential").abs().alias("score_margin_abs"),
                pl.when(pl.col("score_differential") >= 8)
                .then(pl.lit("leading_big"))
                .when(pl.col("score_differential") <= -8)
                .then(pl.lit("trailing_big"))
                .otherwise(pl.lit("one_score"))
                .alias("score_state"),
            ]
        )

    if "down" in joined.columns:
        expressions.append(
            pl.when(pl.col("down") <= 2)
            .then(pl.lit("early_down"))
            .otherwise(pl.lit("late_down"))
            .alias("down_bucket")
        )

    if "ydstogo" in joined.columns:
        expressions.append(
            pl.when(pl.col("ydstogo") <= 3)
            .then(pl.lit("short"))
            .when(pl.col("ydstogo") <= 7)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("long"))
            .alias("distance_bucket")
        )

    if "yardline_100" in joined.columns:
        expressions.append(
            pl.when(pl.col("yardline_100") <= 20)
            .then(pl.lit("red_zone"))
            .when(pl.col("yardline_100") <= 50)
            .then(pl.lit("scoring_range"))
            .otherwise(pl.lit("backed_up"))
            .alias("field_zone")
        )

    if "n_blitzers" in joined.columns:
        expressions.append(
            pl.when(pl.col("n_blitzers").fill_null(0) >= 6)
            .then(pl.lit("heavy_blitz"))
            .when(pl.col("n_blitzers").fill_null(0) >= 5)
            .then(pl.lit("likely_pressure"))
            .otherwise(pl.lit("standard_rush"))
                .alias("pressure_bucket")
        )

    if "n_defense_box" in joined.columns:
        expressions.append(
            pl.when(pl.col("n_defense_box").fill_null(0) >= 8)
            .then(pl.lit("loaded_box"))
            .when(pl.col("n_defense_box").fill_null(0) >= 6)
            .then(pl.lit("standard_box"))
            .otherwise(pl.lit("light_box"))
            .alias("box_bucket")
        )

    if "n_offense_backfield" in joined.columns:
        expressions.append(
            pl.when(pl.col("n_offense_backfield").fill_null(0) >= 3)
            .then(pl.lit("heavy_backfield"))
            .when(pl.col("n_offense_backfield").fill_null(0) == 2)
            .then(pl.lit("balanced_backfield"))
            .otherwise(pl.lit("spread_backfield"))
            .alias("backfield_bucket")
        )

    if "game_seconds_remaining" in joined.columns:
        expressions.append(
            pl.when(pl.col("game_seconds_remaining") <= 120)
            .then(pl.lit("two_minute"))
            .when(pl.col("game_seconds_remaining") <= 900)
            .then(pl.lit("late_game"))
            .otherwise(pl.lit("standard_clock"))
            .alias("clock_bucket")
        )

    if "success" in joined.columns:
        expressions.append(pl.col("success").cast(pl.Int8).alias("target_success"))

    if "epa" in joined.columns:
        expressions.append(pl.col("epa").cast(pl.Float64).alias("target_epa"))

    if "yards_gained" in joined.columns:
        expressions.append(
            (pl.col("yards_gained") >= config.targets.explosive_threshold)
            .cast(pl.Int8)
            .alias("target_explosive")
        )

    if "complete_pass" in joined.columns:
        expressions.append(
            pl.col("complete_pass").fill_null(0).cast(pl.Int8).alias("target_completion")
        )

    if "is_motion" in joined.columns:
        expressions.append(pl.col("is_motion").is_not_null().alias("has_ftn_charting"))
    if "tracking_frames" in joined.columns:
        expressions.append(pl.col("tracking_frames").is_not_null().alias("has_tracking_data"))

    engineered = joined.with_columns(expressions) if expressions else joined
    feature_columns = _feature_columns(engineered, config)
    keep_columns = _existing(
        engineered.columns,
        _unique_ordered(IDENTIFIER_COLUMNS + feature_columns + TARGET_COLUMNS + ["has_ftn_charting", "has_tracking_data"]),
    )
    return engineered.select(keep_columns)


def _feature_columns(frame: pl.DataFrame, config: ProjectConfig) -> list[str]:
    desired = (
        config.features.base_numeric
        + config.features.base_categorical
        + config.features.ftn_numeric
        + config.features.ftn_categorical
        + config.features.tracking_numeric
        + config.features.tracking_categorical
        + config.features.derived_numeric
        + config.features.derived_categorical
    )
    return _existing(frame.columns, desired)


def build_modeling_dataset(
    pbp: pl.DataFrame,
    ftn: pl.DataFrame | None,
    tracking: pl.DataFrame | None,
    config: ProjectConfig,
) -> pl.DataFrame:
    joined = join_ftn_charting(filter_pass_plays(pbp, config), ftn)
    joined = join_tracking_features(joined, tracking)
    return engineer_features(joined, config)


def processed_dataset_path(config: ProjectConfig) -> Path:
    return (
        Path(config.paths.processed_dir)
        / f"{config.project_name}_passing_motion_modeling_dataset.parquet"
    )
