# Final Experiment Handoff

This document is the shortest path from repository to slides/report.

## Final configuration

- Final config: `configs/motion_value_v2_final.yaml`
- Final processed dataset: `data/processed/motion-value-v2-final_passing_motion_modeling_dataset.parquet`
- Final metrics directory: `artifacts/motion-value-v2-final/metrics/`

Run order used for the final experiment:

```bash
python3 scripts/run_experiment.py --command prepare --config configs/motion_value_v2_final.yaml
python3 scripts/run_experiment.py --command train --config configs/motion_value_v2_final.yaml
```

## What changed in V2

Use these points when explaining why this version is more robust than the original project:

- Validation-aware model selection instead of only a simple holdout leaderboard
- Threshold-selection support for classification targets
- Target-specific model-family constraints
- Explicit feature-group ablations:
  - `context_only`
  - `context_plus_motion`
  - `full`
- Split-aware motion-effect reporting with confidence intervals
- Defensive-response summaries with sparse-tracking safeguards
- Multi-config comparison workflow used to choose the final experiment setup
- Guardrails that reject stale processed datasets missing V2 motion features

## Key final files

These are the files most useful for the report and slides.

### Executive summary

- `artifacts/motion-value-v2-final/metrics/proposal_summary.md`
- `artifacts/motion-value-v2-final/metrics/dataset_summary.json`

### Best-model tables

- `artifacts/motion-value-v2-final/metrics/selected_models.csv`
- `artifacts/motion-value-v2-final/metrics/best_models.csv`
- `artifacts/motion-value-v2-final/metrics/validation_metrics.csv`

### Motion-effect analysis

- `artifacts/motion-value-v2-final/metrics/motion_effect_overall.csv`
- `artifacts/motion-value-v2-final/metrics/motion_effect_subgroups.csv`
- `artifacts/motion-value-v2-final/metrics/motion_lift_overall.csv`
- `artifacts/motion-value-v2-final/metrics/motion_lift_subgroups.csv`

### Defensive-response analysis

- `artifacts/motion-value-v2-final/metrics/defensive_reaction_overall.csv`
- `artifacts/motion-value-v2-final/metrics/defensive_reaction_subgroups.csv`
- `artifacts/motion-value-v2-final/metrics/tracking_response_lift_overall.csv`

### Coverage and sanity checks

- `artifacts/motion-value-v2-final/metrics/season_summary.csv`
- `artifacts/motion-value-v2-final/metrics/overall_metrics.csv`
- `artifacts/motion-value-v2-final/metrics/model_manifest.json`

## Recommended charts

If you only make a few figures, start here:

1. Best holdout model by target
   Source: `selected_models.csv`
   Plot: bar chart of `balanced_accuracy` for classification targets and `rmse` for `epa`

2. Overall motion effect by target
   Source: `motion_effect_overall.csv`
   Plot: point-and-interval chart using `adjusted_effect`, `effect_ci_lower`, `effect_ci_upper`

3. Feature-group value by target
   Source: `motion_lift_overall.csv`
   Plot: grouped bars for `context_plus_motion` vs `context_only` and `full` vs `context_plus_motion`

4. Defensive-response caveat
   Source: `dataset_summary.json`
   Plot: simple annotation or small bar showing tracking coverage is `68.2%` in train but only `0.7%` in test

## Final results to say clearly

- Motion shows its clearest positive relationship with `completion`.
- Motion effect is `unclear` for `success`, `explosive`, and `epa` after the chosen context controls.
- The strongest final model by the primary target is:
  - `completion`: gradient boosting with `full`
- The final selected holdout winners are:
  - `completion`: gradient boosting with `full`
  - `explosive`: logistic regression with `context_plus_motion`
  - `success`: logistic regression with `full`
  - `epa`: gradient boosting with `context_only`

## Final limitations to say explicitly

- This is an observational modeling project, not a causal estimate of motion effects.
- Tracking-based defensive conclusions are weak because the holdout tracking coverage is only `0.7%`.
- Predictive gains are modest, so the contribution is more about framework quality and analysis structure than about a high-performing deployment model.

## Suggested slide outline

1. Problem and motivation
2. Data sources and sample size
3. V1 to V2 framework improvements
4. Experiment design and feature groups
5. Best-model results by target
6. Overall motion-effect results
7. Defensive-response findings and tracking limitation
8. Takeaways and future work
