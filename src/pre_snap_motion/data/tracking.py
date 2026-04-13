from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pre_snap_motion.config import ProjectConfig

TRACKING_USECOLS = [
    "game_id",
    "play_id",
    "nfl_id",
    "frame_id",
    "play_direction",
    "absolute_yardline_number",
    "player_position",
    "player_side",
    "x",
    "y",
    "s",
    "a",
]

PLAY_KEYS = ["game_id", "play_id"]
SKILL_POSITIONS = {"WR", "TE", "RB", "FB"}
DEFENSIVE_BACK_POSITIONS = {"CB", "FS", "SS", "S", "DB"}


def tracking_features_path(config: ProjectConfig) -> Path:
    return Path(config.paths.raw_dir) / config.tracking.cache_file


def infer_nfl_season_from_game_id(game_id: str | int) -> int:
    digits = "".join(character for character in str(game_id) if character.isdigit())
    if len(digits) < 8:
        return int(digits[:4])

    year = int(digits[:4])
    month = int(digits[4:6])
    # January and February games belong to the previous NFL season.
    return year - 1 if month <= 2 else year


def resolve_tracking_input_paths(config: ProjectConfig) -> list[Path]:
    paths: list[Path] = []
    for pattern in config.tracking.input_globs:
        paths.extend(sorted(Path().glob(pattern)))
    for file_name in config.tracking.input_files:
        path = Path(file_name)
        if path.exists():
            paths.append(path)

    deduped: dict[str, Path] = {}
    for path in paths:
        deduped[str(path.resolve())] = path
    return sorted(deduped.values())


def tracking_cache_is_stale(config: ProjectConfig) -> bool:
    cache_path = tracking_features_path(config)
    input_paths = resolve_tracking_input_paths(config)
    if not input_paths:
        return False
    if not cache_path.exists():
        return True

    cache_mtime = cache_path.stat().st_mtime
    return any(path.stat().st_mtime > cache_mtime for path in input_paths)


def build_tracking_play_features(config: ProjectConfig) -> Path:
    input_paths = resolve_tracking_input_paths(config)
    if not input_paths:
        raise FileNotFoundError(
            "Tracking is enabled but no local Big Data Bowl input files were found."
        )

    aggregated_frames = [_aggregate_tracking_file(path) for path in input_paths]
    combined = pd.concat(aggregated_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=PLAY_KEYS, keep="last")

    output_path = tracking_features_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(combined).write_parquet(output_path)
    return output_path


def load_tracking_play_features(config: ProjectConfig) -> pl.DataFrame | None:
    if not config.tracking.enabled:
        return None

    cache_path = tracking_features_path(config)
    if tracking_cache_is_stale(config):
        build_tracking_play_features(config)
    return pl.read_parquet(cache_path)


def summarize_tracking_play_features(tracking: pl.DataFrame) -> pd.DataFrame:
    if tracking.is_empty():
        return pd.DataFrame()

    summary = (
        tracking.select(["season", "game_id", "play_id"])
        .unique()
        .group_by("season")
        .agg(
            [
                pl.col("game_id").n_unique().alias("unique_games"),
                pl.len().alias("unique_plays"),
            ]
        )
        .sort("season")
    )
    return summary.to_pandas()


def _aggregate_tracking_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=TRACKING_USECOLS,
        dtype={
            "game_id": "string",
            "play_id": "int64",
            "nfl_id": "int64",
            "frame_id": "int32",
            "play_direction": "string",
            "absolute_yardline_number": "float64",
            "player_position": "string",
            "player_side": "string",
            "x": "float64",
            "y": "float64",
            "s": "float64",
            "a": "float64",
        },
    )
    return _aggregate_tracking_frame(frame)


def _aggregate_tracking_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["season"] = frame["game_id"].map(infer_nfl_season_from_game_id).astype(int)
    frame["is_skill"] = (frame["player_side"] == "Offense") & (
        frame["player_position"].isin(SKILL_POSITIONS)
    )
    frame["is_qb"] = (frame["player_side"] == "Offense") & (
        frame["player_position"] == "QB"
    )
    frame["is_db"] = (frame["player_side"] == "Defense") & (
        frame["player_position"].isin(DEFENSIVE_BACK_POSITIONS)
    )
    frame = frame.sort_values(PLAY_KEYS + ["nfl_id", "frame_id"])

    play_basic = (
        frame.groupby(PLAY_KEYS, sort=False)
        .agg(
            season=("season", "first"),
            tracking_rows=("nfl_id", "size"),
            tracking_players=("nfl_id", "nunique"),
            tracking_frames=("frame_id", "nunique"),
            tracking_play_direction=("play_direction", "first"),
            tracking_absolute_yardline_number=("absolute_yardline_number", "first"),
        )
        .reset_index()
    )

    play_basic = _merge_unique_count(
        play_basic,
        frame[frame["player_side"] == "Offense"],
        "tracking_offense_players",
    )
    play_basic = _merge_unique_count(
        play_basic,
        frame[frame["player_side"] == "Defense"],
        "tracking_defense_players",
    )
    play_basic = _merge_unique_count(
        play_basic,
        frame[frame["player_position"] == "WR"],
        "tracking_wr_players",
    )
    play_basic = _merge_unique_count(
        play_basic,
        frame[frame["player_position"] == "TE"],
        "tracking_te_players",
    )
    play_basic = _merge_unique_count(
        play_basic,
        frame[frame["player_position"].isin({"RB", "FB"})],
        "tracking_rb_players",
    )
    play_basic = _merge_unique_count(
        play_basic,
        frame[frame["is_db"]],
        "tracking_db_players",
    )

    frame_bounds = (
        frame.groupby(PLAY_KEYS, sort=False)
        .agg(
            tracking_input_start_frame=("frame_id", "min"),
            tracking_input_end_frame=("frame_id", "max"),
        )
        .reset_index()
    )
    frame = frame.merge(frame_bounds, on=PLAY_KEYS, how="left")
    start_frame = frame.loc[
        frame["frame_id"] == frame["tracking_input_start_frame"]
    ].copy()
    end_frame = frame.loc[frame["frame_id"] == frame["tracking_input_end_frame"]].copy()

    start_summary = _frame_side_summary(start_frame, prefix="tracking_start")
    end_summary = _frame_side_summary(end_frame, prefix="tracking_end")
    start_separation = _frame_separation_summary(start_frame, prefix="tracking_start")
    end_separation = _frame_separation_summary(end_frame, prefix="tracking_end")

    player_displacement = _player_displacement_summary(frame)

    aggregated = play_basic.merge(frame_bounds, on=PLAY_KEYS, how="left")
    for extra in [
        start_summary,
        end_summary,
        start_separation,
        end_separation,
        player_displacement,
    ]:
        aggregated = aggregated.merge(extra, on=PLAY_KEYS, how="left")

    aggregated["tracking_offense_centroid_shift"] = _centroid_shift(
        aggregated,
        "tracking_start_offense_x_mean",
        "tracking_start_offense_y_mean",
        "tracking_end_offense_x_mean",
        "tracking_end_offense_y_mean",
    )
    aggregated["tracking_defense_centroid_shift"] = _centroid_shift(
        aggregated,
        "tracking_start_defense_x_mean",
        "tracking_start_defense_y_mean",
        "tracking_end_defense_x_mean",
        "tracking_end_defense_y_mean",
    )
    aggregated["tracking_skill_separation_gain"] = (
        aggregated["tracking_end_skill_mean_separation"]
        - aggregated["tracking_start_skill_mean_separation"]
    )
    aggregated["tracking_qb_pressure_distance_delta"] = (
        aggregated["tracking_end_qb_nearest_defender_distance"]
        - aggregated["tracking_start_qb_nearest_defender_distance"]
    )

    return aggregated


def _merge_unique_count(
    base: pd.DataFrame, subset: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    if subset.empty:
        base[column_name] = 0
        return base

    counts = (
        subset.groupby(PLAY_KEYS, sort=False)["nfl_id"]
        .nunique()
        .rename(column_name)
        .reset_index()
    )
    merged = base.merge(counts, on=PLAY_KEYS, how="left")
    merged[column_name] = merged[column_name].fillna(0)
    return merged


def _frame_side_summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    summary = frame.loc[:, PLAY_KEYS].drop_duplicates().reset_index(drop=True)

    for side_name, side_value in [("offense", "Offense"), ("defense", "Defense")]:
        subset = frame.loc[frame["player_side"] == side_value].copy()
        if subset.empty:
            continue

        aggregated = (
            subset.groupby(PLAY_KEYS, sort=False)
            .agg(
                x_mean=("x", "mean"),
                y_mean=("y", "mean"),
                x_min=("x", "min"),
                x_max=("x", "max"),
                y_min=("y", "min"),
                y_max=("y", "max"),
                speed_mean=("s", "mean"),
                accel_mean=("a", "mean"),
            )
            .reset_index()
        )
        aggregated[f"{prefix}_{side_name}_x_span"] = (
            aggregated["x_max"] - aggregated["x_min"]
        )
        aggregated[f"{prefix}_{side_name}_y_span"] = (
            aggregated["y_max"] - aggregated["y_min"]
        )
        aggregated = aggregated.rename(
            columns={
                "x_mean": f"{prefix}_{side_name}_x_mean",
                "y_mean": f"{prefix}_{side_name}_y_mean",
                "speed_mean": f"{prefix}_{side_name}_speed_mean",
                "accel_mean": f"{prefix}_{side_name}_accel_mean",
            }
        )
        aggregated = aggregated.drop(columns=["x_min", "x_max", "y_min", "y_max"])
        summary = summary.merge(aggregated, on=PLAY_KEYS, how="left")

    return summary


def _frame_separation_summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | None]] = []

    for (game_id, play_id), play_frame in frame.groupby(PLAY_KEYS, sort=False):
        defenders = play_frame.loc[
            play_frame["player_side"] == "Defense", ["x", "y"]
        ].to_numpy()
        skill_players = play_frame.loc[play_frame["is_skill"], ["x", "y"]].to_numpy()
        qb = play_frame.loc[play_frame["is_qb"], ["x", "y"]].to_numpy()

        row: dict[str, float | str | int | None] = {
            "game_id": game_id,
            "play_id": play_id,
            f"{prefix}_skill_mean_separation": np.nan,
            f"{prefix}_qb_nearest_defender_distance": np.nan,
        }

        if defenders.size > 0 and skill_players.size > 0:
            row[f"{prefix}_skill_mean_separation"] = _mean_min_distance(
                skill_players, defenders
            )
        if defenders.size > 0 and qb.size > 0:
            row[f"{prefix}_qb_nearest_defender_distance"] = _mean_min_distance(qb, defenders)

        rows.append(row)

    return pd.DataFrame(rows)


def _player_displacement_summary(frame: pd.DataFrame) -> pd.DataFrame:
    player_summary = (
        frame.groupby(PLAY_KEYS + ["nfl_id"], sort=False)
        .agg(
            player_side=("player_side", "first"),
            player_position=("player_position", "first"),
            start_x=("x", "first"),
            start_y=("y", "first"),
            end_x=("x", "last"),
            end_y=("y", "last"),
        )
        .reset_index()
    )
    player_summary["displacement"] = np.hypot(
        player_summary["end_x"] - player_summary["start_x"],
        player_summary["end_y"] - player_summary["start_y"],
    )

    result = player_summary.loc[:, PLAY_KEYS].drop_duplicates().reset_index(drop=True)
    for side_name, mask in [
        ("offense", player_summary["player_side"] == "Offense"),
        ("defense", player_summary["player_side"] == "Defense"),
    ]:
        subset = player_summary.loc[mask].copy()
        if subset.empty:
            continue
        aggregated = (
            subset.groupby(PLAY_KEYS, sort=False)
            .agg(
                mean_displacement=("displacement", "mean"),
                max_displacement=("displacement", "max"),
            )
            .reset_index()
            .rename(
                columns={
                    "mean_displacement": f"tracking_{side_name}_mean_displacement",
                    "max_displacement": f"tracking_{side_name}_max_displacement",
                }
            )
        )
        result = result.merge(aggregated, on=PLAY_KEYS, how="left")

    return result


def _mean_min_distance(points: np.ndarray, reference_points: np.ndarray) -> float:
    distances = np.sqrt(
        ((points[:, None, :] - reference_points[None, :, :]) ** 2).sum(axis=2)
    )
    return float(distances.min(axis=1).mean())


def _centroid_shift(
    frame: pd.DataFrame,
    start_x: str,
    start_y: str,
    end_x: str,
    end_y: str,
) -> pd.Series:
    return np.hypot(frame[end_x] - frame[start_x], frame[end_y] - frame[start_y])
