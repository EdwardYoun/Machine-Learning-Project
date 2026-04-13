from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pre_snap_motion.config import ProjectConfig


def project_artifacts_dir(config: ProjectConfig) -> Path:
    return Path(config.paths.artifacts_dir) / config.project_name


def ensure_directories(config: ProjectConfig) -> None:
    artifacts_root = project_artifacts_dir(config)
    directories = [
        Path(config.paths.raw_dir),
        Path(config.paths.processed_dir),
        artifacts_root,
        artifacts_root / "models",
        artifacts_root / "metrics",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_text(content: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path
