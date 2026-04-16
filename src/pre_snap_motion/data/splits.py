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


def split_frame(frame: pd.DataFrame, split_config: SplitConfig) -> SplitFrames:
    if split_config.strategy == "explicit":
        return split_by_season(frame, split_config)

    if split_config.strategy == "rolling_origin":
        ordered_train_seasons = sorted(set(split_config.train_seasons))
        if len(ordered_train_seasons) <= split_config.rolling_min_train_seasons:
            raise ValueError(
                "rolling_origin strategy requires more train seasons than rolling_min_train_seasons."
            )
        train_seasons = ordered_train_seasons[:-1]
        validation_seasons = [ordered_train_seasons[-1]]
        return split_by_season(
            frame,
            SplitConfig(
                strategy="explicit",
                train_seasons=train_seasons,
                validation_seasons=validation_seasons,
                test_seasons=split_config.test_seasons,
            ),
        )

    raise ValueError(f"Unsupported split strategy: {split_config.strategy}")


def rolling_origin_validation_splits(
    frame: pd.DataFrame,
    split_config: SplitConfig,
) -> list[SplitFrames]:
    if split_config.strategy != "rolling_origin":
        base_split = split_frame(frame, split_config)
        return [base_split] if base_split.validation is not None else []

    ordered_train_seasons = sorted(set(split_config.train_seasons))
    folds: list[SplitFrames] = []
    for index in range(split_config.rolling_min_train_seasons, len(ordered_train_seasons)):
        train_seasons = ordered_train_seasons[:index]
        validation_seasons = [ordered_train_seasons[index]]
        folds.append(
            split_by_season(
                frame,
                SplitConfig(
                    strategy="explicit",
                    train_seasons=train_seasons,
                    validation_seasons=validation_seasons,
                    test_seasons=split_config.test_seasons,
                ),
            )
        )
    return folds
