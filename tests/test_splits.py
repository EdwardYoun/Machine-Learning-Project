import pandas as pd

from pre_snap_motion.config import SplitConfig
from pre_snap_motion.data.splits import rolling_origin_validation_splits, split_by_season, split_frame


def test_split_by_season_respects_configured_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2025],
            "target_success": [1, 0, 1, 0],
        }
    )

    splits = split_by_season(
        frame,
        SplitConfig(
            train_seasons=[2022, 2023],
            validation_seasons=[2024],
            test_seasons=[2025],
        ),
    )

    assert splits.train["season"].tolist() == [2022, 2023]
    assert splits.validation is not None
    assert splits.validation["season"].tolist() == [2024]
    assert splits.test["season"].tolist() == [2025]


def test_split_frame_builds_rolling_origin_validation() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2025],
            "target_success": [1, 0, 1, 0],
        }
    )

    split = split_frame(
        frame,
        SplitConfig(
            strategy="rolling_origin",
            train_seasons=[2022, 2023, 2024],
            validation_seasons=[],
            test_seasons=[2025],
            rolling_min_train_seasons=2,
        ),
    )

    assert split.train["season"].tolist() == [2022, 2023]
    assert split.validation is not None
    assert split.validation["season"].tolist() == [2024]
    assert split.test["season"].tolist() == [2025]

    folds = rolling_origin_validation_splits(
        frame,
        SplitConfig(
            strategy="rolling_origin",
            train_seasons=[2022, 2023, 2024],
            validation_seasons=[],
            test_seasons=[2025],
            rolling_min_train_seasons=1,
        ),
    )
    assert len(folds) == 2
