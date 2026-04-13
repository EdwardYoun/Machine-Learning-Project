import math

import pandas as pd

from pre_snap_motion.data.tracking import (
    _aggregate_tracking_frame,
    infer_nfl_season_from_game_id,
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
