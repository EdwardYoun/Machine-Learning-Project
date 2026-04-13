# Proposal Completion Next Steps

This repo now covers the core proposal workflow:

- nflverse + FTN ingestion
- pass-play filtering
- pre-snap/context feature engineering
- multiple targets (`success`, `explosive`, `completion`, `epa`)
- baseline vs nonlinear models
- `full` vs `no_motion` ablations
- season-aware train/test evaluation
- subgroup reporting and motion-lift summaries

## Remaining Gaps

### 1. Likely coverage context is still weak

The current FTN fields expose motion, pressure, box count, QB location, and hash alignment, but they do not directly provide defensive coverage labels. The proposal would be better matched by adding one of:

- a coverage-labeled charting source
- manually engineered coverage proxies
- Big Data Bowl tracking-derived shell/rotation features

### 2. Tracking-based defensive response is not implemented yet

The proposal explicitly mentions testing whether post-motion defensive response is more predictive than pre-motion alignment alone. That requires a tracking pipeline that can:

- load the Big Data Bowl tracking tables
- align pre-snap motion frames with the play table
- derive pre-motion and post-motion defensive structure features
- compare models using alignment-only vs response-aware features

### 3. Evaluation is a single holdout split right now

The current framework respects season boundaries, but a fuller proposal-grade evaluation should add rolling backtests such as:

- train `2022`, test `2023`
- train `2022-2023`, test `2024`
- train `2022-2024`, test `2025`

That would make the motion conclusions more stable and easier to defend.

### 4. Interpretation can go further

To better answer the "when and why" question, the next analysis layer should add:

- permutation importance on the best model per target
- partial dependence or grouped effect plots for `is_motion`, `n_blitzers`, and `n_pass_rushers`
- subgroup ranking by positive and negative motion lift

## Recommended Build Order

1. Add rolling season backtests.
2. Add feature-importance and effect summaries for the best model per target.
3. Add tracking-data ingestion and pre/post-motion defensive response features.
4. Add coverage-context features once a reliable source is available.
