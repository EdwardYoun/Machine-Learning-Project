import math

import pandas as pd

from pre_snap_motion.config import PathsConfig, ProjectConfig, TrackingConfig
from pre_snap_motion.data.tracking import (
    _aggregate_tracking_frame,
    build_tracking_play_features,
    infer_nfl_season_from_game_id,
    tracking_cache_is_stale,
    tracking_manifest_path,
)


def test_aggregate_tracking_frame_builds_safe_play_level_features() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["2023090700"] * 8,
            "play_id": [101] * 8,
            "nfl_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "frame_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "play_direction": ["right"] * 8,
            "absolute_yardline_number": [42.0] * 8,
            "player_position": ["QB", "QB", "WR", "WR", "CB", "CB", "FS", "FS"],
            "player_side": [
                "Offense",
                "Offense",
                "Offense",
                "Offense",
                "Defense",
                "Defense",
                "Defense",
                "Defense",
            ],
            "x": [50.0, 50.5, 52.0, 54.0, 53.0, 53.5, 55.0, 54.0],
            "y": [26.0, 26.0, 20.0, 21.0, 20.5, 20.5, 29.0, 28.0],
            "s": [0.3, 0.4, 0.9, 1.3, 0.4, 0.7, 0.6, 1.1],
            "a": [0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1, 0.3],
        }
    )

    aggregated = _aggregate_tracking_frame(frame)
    row = aggregated.iloc[0]

    assert row["season"] == 2023
    assert row["tracking_frames"] == 2
    assert row["tracking_players"] == 4
    assert row["tracking_offense_players"] == 2
    assert row["tracking_defense_players"] == 2
    assert row["tracking_wr_players"] == 1
    assert row["tracking_db_players"] == 2
    assert math.isclose(
        row["tracking_offense_mean_displacement"], 1.368033988749895, rel_tol=1e-6
    )
    assert math.isclose(
        row["tracking_start_skill_mean_separation"], 1.118033988749895, rel_tol=1e-6
    )
    assert math.isclose(
        row["tracking_end_qb_nearest_defender_distance"],
        math.hypot(3.5, 2.0),
        rel_tol=1e-6,
    )


def test_infer_nfl_season_from_game_id_handles_january_games() -> None:
    assert infer_nfl_season_from_game_id("2023090700") == 2023
    assert infer_nfl_season_from_game_id("2024010705") == 2023
    assert infer_nfl_season_from_game_id("2024120805") == 2024


def test_tracking_cache_becomes_stale_when_input_set_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    train_dir = tmp_path / "train"
    raw_dir = tmp_path / "raw"
    train_dir.mkdir()
    raw_dir.mkdir()

    def write_tracking_input(path, game_id: str, play_id: int) -> None:
        frame = pd.DataFrame(
            {
                "game_id": [game_id] * 8,
                "play_id": [play_id] * 8,
                "nfl_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "frame_id": [1, 2, 1, 2, 1, 2, 1, 2],
                "play_direction": ["right"] * 8,
                "absolute_yardline_number": [42.0] * 8,
                "player_position": ["QB", "QB", "WR", "WR", "CB", "CB", "FS", "FS"],
                "player_side": [
                    "Offense",
                    "Offense",
                    "Offense",
                    "Offense",
                    "Defense",
                    "Defense",
                    "Defense",
                    "Defense",
                ],
                "x": [50.0, 50.5, 52.0, 54.0, 53.0, 53.5, 55.0, 54.0],
                "y": [26.0, 26.0, 20.0, 21.0, 20.5, 20.5, 29.0, 28.0],
                "s": [0.3, 0.4, 0.9, 1.3, 0.4, 0.7, 0.6, 1.1],
                "a": [0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.1, 0.3],
            }
        )
        frame.to_csv(path, index=False)

    config = ProjectConfig(
        paths=PathsConfig(raw_dir="raw"),
        tracking=TrackingConfig(
            enabled=True,
            input_globs=["train/input_*.csv"],
            input_files=[],
            cache_file="tracking.parquet",
        ),
    )

    write_tracking_input(train_dir / "input_2023_w01.csv", "2023090700", 101)
    assert tracking_cache_is_stale(config) is True

    build_tracking_play_features(config)
    assert tracking_manifest_path(config).exists()
    assert tracking_cache_is_stale(config) is False

    write_tracking_input(train_dir / "input_2023_w02.csv", "2023091400", 102)
    assert tracking_cache_is_stale(config) is True
