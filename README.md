# Reading the Defense Before the Snap

Project framework for the proposal: **Quantifying the Value of Pre-Snap Motion in the NFL**.

This scaffold turns the proposal into a reproducible Python pipeline that:

- pulls play-by-play data and FTN charting data through `nflreadpy`
- filters to pass plays and engineers pre-snap motion/context features
- trains interpretable baseline models and nonlinear models across multiple proposal targets
- compares full-feature models against motion-ablated variants
- reports season-aware test metrics, subgroup analysis, and motion-lift summaries

## Why Python here?

The proposal cites `nflreadr`, but the current nflverse docs point Python users to `nflreadpy`, which mirrors the `load_*` API and supports both `load_pbp()` and `load_ftn_charting()`. That makes it the cleanest fit for this environment while staying aligned with the proposal's data sources.

## Repository Layout

```text
configs/                  YAML experiment configs
docs/                     Project notes and collaboration docs
scripts/                  Experiment entrypoints
src/pre_snap_motion/      Project package
tests/                    Lightweight regression tests
ML_Project_Proposal.pdf   Original proposal
```

## Quick Start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
pre-snap-motion run --config configs/quickstart.yaml
```

## Pipeline Commands

```powershell
pre-snap-motion fetch --config configs/default.yaml
pre-snap-motion prepare --config configs/default.yaml
pre-snap-motion train --config configs/default.yaml
pre-snap-motion run --config configs/default.yaml
pre-snap-motion run --config configs/tracking_experiment.yaml
python scripts/run_experiment.py --command run --config configs/tracking_experiment.yaml
python scripts/run_experiment.py --command inspect --config configs/tracking_experiment.yaml
```

Artifacts are written to:

- `data/raw/` for downloaded play-level tables
- `data/processed/<project_name>_passing_motion_modeling_dataset.parquet` for the modeling dataset
- `artifacts/<project_name>/models/` for trained pipelines
- `artifacts/<project_name>/metrics/` for CSV and JSON summaries

The metrics directory now includes:

- `overall_metrics.csv` for all target/model/feature-set combinations
- an `evaluation_slice` column so tracking experiments can report both `all` plays and `tracking_only` holdout rows
- `motion_lift_overall.csv` for `full` vs `no_motion` improvement deltas
- `subgroup_metrics.csv` and `motion_lift_subgroups.csv` for where motion helps most
- `season_summary.csv` and `dataset_summary.json` for dataset coverage and target rates
- `proposal_summary.md` for a readable experiment summary

The centralized runner can also execute multiple configs or inspect tracking availability:

```powershell
python scripts/run_experiment.py --command inspect --config configs/tracking_experiment.yaml
python scripts/run_experiment.py --command run --all-configs
```

## Tracking Integration

The repo now supports a local Big Data Bowl tracking branch through
`configs/tracking_experiment.yaml`.

- It reads the competition `train/input_*.csv` and `test_input.csv` files locally.
- It aggregates them into play-level tracking features under `data/raw/tracking_play_features_*.parquet`.
- It joins those features to the nflverse + FTN play table on legacy `old_game_id` plus `play_id`.
- It adds a `no_tracking_response` feature-set ablation so we can compare alignment-only features against response-aware tracking features.
- The tracking cache now auto-refreshes when new local input files are added, so if `train/input_2024_w*.csv` files are dropped in later they will be picked up automatically by the same config.

Important caveat:

- The local 2026 competition files are reduced competition inputs, not the full raw tracking feed.
- They also include post-play labels that would leak outcome information, so the integration intentionally excludes fields such as `player_to_predict`, target-role labels, and ball landing metadata.
- The current local 2024 tracking coverage is sparse because only `test_input.csv` is available, and most of its January 2024 games belong to the 2023 NFL season. Fuller 2024 ingestion requires additional local `2024` weekly input files.

## How The Framework Maps To The Proposal

1. **Problem and data**
   The pipeline joins nflverse play-by-play data with FTN charting data on `game_id` and `play_id`, then focuses on pass plays where motion, pressure, coverage context, and game state can be analyzed together.
2. **Baseline vs nonlinear models**
   The registry includes logistic/ridge baselines and nonlinear tree-based models.
3. **Motion-value hypothesis**
   The training loop automatically runs both `full` and `no_motion` feature sets so the ablation question from the proposal is built into the workflow.
4. **Evaluation**
   Classification metrics include AUROC, log loss, Brier score, and expected calibration error across success, explosive-play, and completion targets. Regression metrics include RMSE and MAE for EPA. Splits are season-aware by config.
5. **Subgroup analysis**
   The reporting step slices performance by down, distance, field zone, score state, pressure, QB location, and hash-based structure.

## Basic Implementation Steps

1. Run `pre-snap-motion fetch` to cache raw nflverse and FTN data locally.
2. Run `pre-snap-motion prepare` to build the pass-play modeling table.
3. Run `pre-snap-motion train` to fit baseline and nonlinear models on explicit train/test seasons.
4. Inspect `artifacts/metrics/overall_metrics.csv` and `best_models.csv` for model comparisons.
5. Inspect `artifacts/metrics/motion_lift_overall.csv` and `motion_lift_subgroups.csv` to see where motion adds the most value.
6. Read `artifacts/metrics/proposal_summary.md` for a concise write-up of the current experiment.
7. Extend `configs/default.yaml` or add a new config for tracking-data experiments later.
