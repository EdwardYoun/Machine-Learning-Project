from __future__ import annotations

from pathlib import Path

import polars as pl

from pre_snap_motion.config import ProjectConfig
from pre_snap_motion.data.features import build_modeling_dataset, processed_dataset_path
from pre_snap_motion.data.loaders import fetch_raw_datasets, load_raw_datasets
from pre_snap_motion.io import ensure_directories
from pre_snap_motion.modeling.train import train_models


def fetch(config: ProjectConfig) -> dict[str, Path]:
    ensure_directories(config)
    return fetch_raw_datasets(config)


def prepare(config: ProjectConfig) -> Path:
    ensure_directories(config)
    pbp, ftn, tracking = load_raw_datasets(config)
    dataset = build_modeling_dataset(pbp, ftn, tracking, config)
    output_path = processed_dataset_path(config)
    dataset.write_parquet(output_path)
    return output_path


def train(config: ProjectConfig) -> dict[str, Path]:
    ensure_directories(config)
    dataset_path = processed_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset at {dataset_path}. Run the prepare command first."
        )
    dataset = pl.read_parquet(dataset_path)
    return train_models(dataset, config)


def run(config: ProjectConfig) -> dict[str, Path]:
    outputs = fetch(config)
    dataset_path = prepare(config)
    train_outputs = train(config)
    outputs["processed_dataset"] = dataset_path
    outputs.update(train_outputs)
    return outputs
