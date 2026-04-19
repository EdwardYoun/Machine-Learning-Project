from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pre_snap_motion.config import SplitConfig


@dataclass(slots=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame | None
    test: pd.DataFrame


def _season_train_test_split(frame: pd.DataFrame, split_config: SplitConfig) -> SplitFrames:
    train = frame.loc[frame["season"].isin(split_config.train_seasons)].copy()
    test = frame.loc[frame["season"].isin(split_config.test_seasons)].copy()

    if train.empty:
        raise ValueError("Train split is empty. Check the configured train seasons.")
    if test.empty:
        raise ValueError("Test split is empty. Check the configured test seasons.")

    return SplitFrames(train=train, validation=None, test=test)


def split_by_season(frame: pd.DataFrame, split_config: SplitConfig) -> SplitFrames:
    base = _season_train_test_split(frame, split_config)
    train = base.train
    validation = None
    if split_config.validation_seasons:
        validation = frame.loc[
            frame["season"].isin(split_config.validation_seasons)
        ].copy()
    return SplitFrames(train=train, validation=validation, test=base.test)


def _ordered_train_weeks(
    frame: pd.DataFrame,
    split_config: SplitConfig,
) -> list[tuple[int, int]]:
    if "week" not in frame.columns:
        raise ValueError("rolling_origin_weeks strategy requires a 'week' column.")

    train = frame.loc[frame["season"].isin(split_config.train_seasons)].copy()
    if train.empty:
        raise ValueError("Train split is empty. Check the configured train seasons.")

    keys = sorted(
        {
            (int(season), int(week))
            for season, week in zip(train["season"], train["week"])
            if pd.notna(season) and pd.notna(week)
        }
    )
    if not keys:
        raise ValueError("rolling_origin_weeks strategy requires non-null season/week values.")
    return keys


def _mask_for_week_keys(frame: pd.DataFrame, week_keys: set[tuple[int, int]]) -> pd.Series:
    season_values = pd.to_numeric(frame["season"], errors="coerce")
    week_values = pd.to_numeric(frame["week"], errors="coerce")
    return pd.Series(
        list(zip(season_values.fillna(-1).astype(int), week_values.fillna(-1).astype(int))),
        index=frame.index,
    ).isin(week_keys)


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

    if split_config.strategy == "rolling_origin_weeks":
        return _season_train_test_split(frame, split_config)

    raise ValueError(f"Unsupported split strategy: {split_config.strategy}")


def rolling_origin_validation_splits(
    frame: pd.DataFrame,
    split_config: SplitConfig,
) -> list[SplitFrames]:
    if split_config.strategy == "rolling_origin_weeks":
        ordered_weeks = _ordered_train_weeks(frame, split_config)
        min_train_weeks = split_config.rolling_min_train_weeks
        validation_window = split_config.rolling_validation_window_weeks
        if len(ordered_weeks) < min_train_weeks + validation_window:
            raise ValueError(
                "rolling_origin_weeks strategy requires more observed train weeks than "
                "rolling_min_train_weeks + rolling_validation_window_weeks."
            )

        base = _season_train_test_split(frame, split_config)
        folds: list[SplitFrames] = []
        for index in range(
            min_train_weeks,
            len(ordered_weeks) - validation_window + 1,
            validation_window,
        ):
            train_keys = set(ordered_weeks[:index])
            validation_keys = set(ordered_weeks[index : index + validation_window])
            train_mask = _mask_for_week_keys(base.train, train_keys)
            validation_mask = _mask_for_week_keys(base.train, validation_keys)
            fold_train = base.train.loc[train_mask].copy()
            fold_validation = base.train.loc[validation_mask].copy()
            if fold_train.empty or fold_validation.empty:
                continue
            folds.append(
                SplitFrames(
                    train=fold_train,
                    validation=fold_validation,
                    test=base.test.copy(),
                )
            )
        return folds

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
