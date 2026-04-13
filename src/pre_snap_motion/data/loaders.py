from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import polars as pl

from pre_snap_motion.config import ProjectConfig
from pre_snap_motion.io import ensure_directories
from pre_snap_motion.data.tracking import (
    build_tracking_play_features,
    load_tracking_play_features,
    tracking_features_path,
)


def _season_slug(seasons: list[int]) -> str:
    return "_".join(str(season) for season in seasons)


def raw_pbp_path(config: ProjectConfig) -> Path:
    return Path(config.paths.raw_dir) / f"pbp_{_season_slug(config.data.pbp_seasons)}.parquet"


def raw_ftn_path(config: ProjectConfig) -> Path:
    return (
        Path(config.paths.raw_dir)
        / f"ftn_charting_{_season_slug(config.data.ftn_seasons)}.parquet"
    )


def fetch_raw_datasets(config: ProjectConfig) -> dict[str, Path]:
    ensure_directories(config)

    pbp = nfl.load_pbp(config.data.pbp_seasons)
    pbp_path = raw_pbp_path(config)
    pbp.write_parquet(pbp_path)

    outputs: dict[str, Path] = {"pbp": pbp_path}
    if config.data.use_ftn_charting:
        ftn = nfl.load_ftn_charting(config.data.ftn_seasons)
        ftn_path = raw_ftn_path(config)
        ftn.write_parquet(ftn_path)
        outputs["ftn"] = ftn_path
    if config.tracking.enabled:
        outputs["tracking"] = build_tracking_play_features(config)

    return outputs


def load_raw_datasets(
    config: ProjectConfig,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
    pbp_path = raw_pbp_path(config)
    if not pbp_path.exists():
        raise FileNotFoundError(
            f"Missing raw play-by-play data at {pbp_path}. Run the fetch command first."
        )

    pbp = pl.read_parquet(pbp_path)
    ftn: pl.DataFrame | None = None
    if config.data.use_ftn_charting:
        ftn_path = raw_ftn_path(config)
        if not ftn_path.exists():
            raise FileNotFoundError(
                f"Missing raw FTN charting data at {ftn_path}. Run the fetch command first."
            )
        ftn = pl.read_parquet(ftn_path)

    tracking: pl.DataFrame | None = None
    if config.tracking.enabled:
        tracking_path = tracking_features_path(config)
        if not tracking_path.exists():
            build_tracking_play_features(config)
        tracking = load_tracking_play_features(config)

    return pbp, ftn, tracking
