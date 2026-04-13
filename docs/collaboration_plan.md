# Collaboration Plan

## Team Goal

Ship a clean, defensible course project that answers the proposal question:
how much value pre-snap motion adds on NFL passing plays, and when that value is most visible.

## Current Status

### What Is Already Done

- the repo has a working Python project structure with configs, package code, tests, and docs
- nflverse play-by-play and FTN charting ingestion are wired into the pipeline
- pass-play filtering and proposal-aligned target engineering are implemented
- baseline and nonlinear models run across `success`, `explosive`, `completion`, and `epa`
- `full` vs `no_motion` ablations are built into training
- subgroup metrics, motion-lift summaries, season summaries, and proposal markdown outputs are generated
- Big Data Bowl competition inputs are integrated into a tracking-lite play-level feature branch
- tracking-aware reporting now distinguishes `all` vs `tracking_only` evaluation slices
- a centralized experiment runner exists for `inspect`, `fetch`, `prepare`, `train`, and `run`
- the repo has been cleaned up for GitHub upload with ignore rules and contributor documentation

### What Is Still Left To Do

- add a stronger 2023 tracking backtest so tracking results are evaluated on data with real coverage
- improve tracking-derived alignment and defensive-response features
- add richer evaluation summaries and clearer comparison tables for the final report
- decide which results are strong enough to present as the main project findings
- translate experiment outputs into final visuals, report sections, and presentation material
- if a better data source becomes available, extend or replace the current limited 2024 tracking setup

## Working Model

- Keep one shared `main` branch for stable, presentation-ready work.
- Use short-lived feature branches for each contributor.
- Merge only after one teammate has read the diff and the experiment output summary.
- Use the centralized runner so everyone is exercising the same commands.

## Work Segmentation

### Contributor 1: Data and Tracking Pipeline

Primary ownership:

- `src/pre_snap_motion/data/`
- `configs/tracking_experiment.yaml`
- tracking-related documentation

Core tasks:

- maintain local tracking ingestion and cache behavior
- improve play-level tracking features
- add any future full-2024 or alternate tracking sources
- validate join coverage and season labeling

Good stopping point:

- tracking cache builds cleanly
- coverage summaries are updated
- no leakage fields are introduced

### Contributor 2: Modeling and Evaluation

Primary ownership:

- `src/pre_snap_motion/modeling/`
- `src/pre_snap_motion/evaluation/`
- experiment metric outputs and ablations

Core tasks:

- improve model selection and backtesting design
- add rolling or week-based validation for 2023 tracking experiments
- improve subgroup analysis and motion-lift reporting
- add interpretation outputs for best models

Good stopping point:

- new metrics are written through the existing reporting flow
- comparisons remain reproducible through config + runner
- proposal claims can be backed by exported tables

### Contributor 3: Repo Integration, Narrative, and Delivery

Primary ownership:

- `README.md`
- `docs/`
- `scripts/`
- presentation-facing polish

Core tasks:

- keep GitHub-facing docs current
- maintain the centralized runner and experiment workflow docs
- translate experiment outputs into proposal-ready summary language
- coordinate final visuals, tables, and write-up structure

Good stopping point:

- a new teammate can run the repo from the README
- experiment status is visible without opening source code
- final deliverables are aligned with the repo outputs

## Suggested Near-Term Task Split

### You

- keep the shared picture of what is already solid vs what is still tentative
- own the final integration pass and decide which experiment results are presentation-worthy
- review merges for consistency with the proposal question
- keep the team focused on one primary story for the final submission

### Collaborator A

- build on the tracking pipeline work that is already in place
- build the 2023 tracking backtest branch
- strengthen tracking-derived alignment and defensive-response features
- document exactly what the public tracking bundle can and cannot support

### Collaborator B

- build on the current modeling and reporting outputs
- improve evaluation and interpretation
- add best-model explanations, subgroup ranking, and cleaner summary tables
- help convert experiment outputs into report-ready figures

## Recommended Branches

- `feature/tracking-backtest-2023`
- `feature/modeling-and-reporting`
- `feature/docs-and-runner-polish`

## Coordination Cadence

- 15-minute sync at the start of each working session
- one owner per branch
- one reviewer per merge
- one shared status note updated after each experiment rerun

## Merge Rules

- Do not mix tracking feature engineering and reporting refactors in the same PR.
- Keep config changes with the code they are meant to exercise.
- Paste the exact runner command used for any reported result.
- If a result depends on sparse tracking coverage, call that out in the PR summary.

## Immediate Next Sprint

1. Contributor 1 builds a 2023-only tracking backtest config and reports coverage.
2. Contributor 2 adds rolling evaluation summaries and simplified comparison tables.
3. Contributor 3 turns the current repo state into a polished GitHub-ready narrative and final write-up skeleton.
4. You review the three pieces together and decide the main analysis narrative.
