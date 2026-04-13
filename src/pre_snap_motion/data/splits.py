from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pre_snap_motion.config import SplitConfig


@dataclass(slots=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame | None
    test: pd.DataFrame


def split_by_season(frame: pd.DataFrame, split_config: SplitConfig) -> SplitFrames:
    train = frame.loc[frame["season"].isin(split_config.train_seasons)].copy()
    validation = None
    if split_config.validation_seasons:
        validation = frame.loc[
            frame["season"].isin(split_config.validation_seasons)
        ].copy()
    test = frame.loc[frame["season"].isin(split_config.test_seasons)].copy()

    if train.empty:
        raise ValueError("Train split is empty. Check the configured train seasons.")
    if test.empty:
        raise ValueError("Test split is empty. Check the configured test seasons.")

    return SplitFrames(train=train, validation=validation, test=test)
