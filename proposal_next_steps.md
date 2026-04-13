# project Completion Next Steps

This repo now covers most of the project skeleton and a meaningful share of the project analysis workflow.

## What Is Already Implemented

- nflverse play-by-play ingestion
- FTN charting ingestion
- pass-play filtering
- pre-snap and game-context feature engineering
- project targets for `success`, `explosive`, `completion`, and `epa`
- baseline and nonlinear models
- `full` vs `no_motion` ablations
- subgroup metrics, motion-lift summaries, season summaries, and project markdown outputs
- a tracking-lite Big Data Bowl integration based on the public competition input files
- `all` vs `tracking_only` evaluation slices for tracking-aware experiments
- a centralized experiment runner for `inspect`, `fetch`, `prepare`, `train`, and `run`

## Current Constraints

### 1. Full 2024 tracking is not available locally

The public competition bundle currently gives us:

- full weekly `2023` training inputs
- one limited `test_input.csv`

After correcting for NFL season boundaries, that means true `2024` tracking coverage is still very small. So the strongest tracking analysis should currently be framed as a `2023` tracking study, not a full `2024` tracking evaluation.

### 2. Tracking response features are only an initial version

The current tracking branch already creates play-level alignment and movement summaries, but it is still a first pass. The project would be better matched by stronger defensive-response features such as:

- shell or depth-shift proxies
- defender displacement by side or role
- box-count change proxies
- cleaner pre-alignment vs post-adjustment feature splits

### 3. Evaluation is still lighter than the final project should be

The current pipeline can do season-based experiments, but the next round should make the conclusions more stable and easier to defend with:

- 2023-only tracking backtests
- rolling or week-based validation inside 2023
- clearer comparison tables that separate strong results from directional ones

### 4. Final interpretation and presentation still need work

The pipeline now produces useful metrics, but the project still needs a cleaner analysis layer for the final write-up:

- best-model interpretation
- concise ranking of where motion helps and hurts
- report-ready visuals and summary tables
- a tighter final narrative around what we can say confidently versus what remains limited by data

## Recommended Next Steps

### 1. Build a 2023-only tracking backtest

This is the highest-value next step because it lets us evaluate tracking features where coverage is real rather than sparse.

Suggested scope:

- create a dedicated 2023 tracking config
- split by weeks or rolling windows within 2023
- report tracking-aware results on a meaningful holdout

### 2. Improve tracking-derived defensive-response features

Once the backtest exists, strengthen the tracking branch so it better answers the project question about pre-motion alignment versus post-motion defensive adjustment.

### 3. Improve evaluation summaries and interpretation

Add cleaner output tables and lightweight interpretation so the final story is easier to explain and defend.

### 4. Keep 2024 tracking as optional future expansion

If fuller `2024` tracking data becomes available later, the current pipeline is already set up to ingest it. Until then, treat the limited `2024` slice as supplemental only.

## Recommended Build Order

1. Add a dedicated 2023 tracking backtest config and reporting flow.
2. Improve tracking-derived defensive-response features.
3. Add clearer interpretation and comparison outputs for the final report.
4. Expand 2024 tracking only if a better data source becomes available.
