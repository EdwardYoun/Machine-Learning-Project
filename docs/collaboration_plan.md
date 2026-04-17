# Collaboration Plan

## Team Goal

Ship a clean, defensible course project that answers the proposal question:
how much value pre-snap motion adds on NFL passing plays, and when that value is most visible.

## Current Status

### What Is Already Done

- the repo has a working Python package, configs, tests, docs, and experiment runner
- the final experiment config is `configs/motion_value_v2_final.yaml`
- the final prepared dataset and metrics have been generated
- the repo includes a final handoff doc for slides/report work
- the branch is organized for GitHub review and submission use

### Main Final Findings

- the clearest positive motion result is on `completion`
- the adjusted motion effect is unclear for `success`, `explosive`, and `epa`
- tracking-response analysis remains exploratory because test tracking coverage is only `0.7%`

### What Is Still Left To Do

- turn the existing outputs into presentation visuals
- write the final report sections using the exported summaries and CSV tables
- optionally push a final PR or merge to the stable branch after review

## Working Model

- Keep one shared `main` branch for stable, presentation-ready work.
- Use short-lived feature branches for each contributor.
- Merge only after one teammate has read the diff and the experiment output summary.
- Use the centralized runner so everyone is exercising the same commands.

## Work Segmentation

### Contributor 1: Data and Tracking Pipeline

Primary ownership:

- `src/pre_snap_motion/data/`
- `configs/motion_value_v2_final.yaml`
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

1. Build charts from `selected_models.csv`, `motion_effect_overall.csv`, and `dataset_summary.json`.
2. Turn `proposal_summary.md` and `docs/final_experiment_handoff.md` into presentation bullets.
3. Draft the final report around the current experiment results and limitations.
4. Merge or submit once the write-up materials are complete.
