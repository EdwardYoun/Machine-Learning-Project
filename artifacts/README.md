# Artifacts Snapshot

This folder is the GitHub-friendly snapshot of the final V2 experiment outputs.

## What this folder is

- `artifacts/metrics/` is a curated copy of the final results from the local run of `configs/motion_value_v2_final.yaml`
- `artifacts/models/` contains the final selected holdout winner models for the main `all` evaluation slice

## Why this exists

The full local experiment outputs live under `artifacts/motion-value-v2-final/`, but that directory is treated as a local experiment workspace. The files in this top-level `artifacts/` folder are the small set of final results intended to be visible directly in the repository.

## Most useful files

- `artifacts/metrics/proposal_summary.md`
- `artifacts/metrics/selected_models.csv`
- `artifacts/metrics/motion_effect_overall.csv`
- `artifacts/metrics/dataset_summary.json`
- `artifacts/metrics/validation_metrics.csv`

## Final selected models in this snapshot

- `classification_completion_full_gradient_boosting.pkl`
- `classification_explosive_context_plus_motion_logistic_regression.pkl`
- `classification_success_full_logistic_regression.pkl`
- `regression_epa_context_only_gradient_boosting.pkl`
