# Results Summary

This summary reflects the final outputs in `artifacts/motion-value-v2-final/metrics/` from `configs/motion_value_v2_final.yaml`.

The final experiment uses:

- `2023` as the training season
- `2024` as the test season
- V2 feature groups: `context_only`, `context_plus_motion`, and `full`
- validation-aware model selection with target-specific model families
- a local tracking integration, while explicitly treating test-split tracking conclusions as directional because coverage is sparse

## Dataset and coverage

- Total pass-play rows: `40,809`
- Train rows: `20,693`
- Test rows: `20,116`
- FTN charting coverage: `100.0%`
- Overall tracking coverage: `34.9%`
- Train tracking coverage: `68.2%`
- Test tracking coverage: `0.7%`
- Overall motion rate: `45.7%`

| Season | Rows | Tracking coverage | Motion rate | Success rate | Explosive rate | Completion rate | Mean EPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 20,693 | 68.2% | 42.0% | 0.444 | 0.082 | 0.571 | -0.001 |
| 2024 | 20,116 | 0.7% | 49.5% | 0.458 | 0.082 | 0.578 | 0.052 |

## Final selected models

### Full test set (`all`)

| Target | Best model | Feature set | Metric |
| --- | --- | --- | --- |
| Completion | Gradient boosting | `full` | Balanced accuracy = `0.5543`, AUROC = `0.5869` |
| Explosive | Logistic regression | `context_plus_motion` | Balanced accuracy = `0.5117`, AUROC = `0.6084` |
| Success | Logistic regression | `full` | Balanced accuracy = `0.5030`, AUROC = `0.5655` |
| EPA | Gradient boosting | `context_only` | RMSE = `1.6686` |

### Tracking-only test slice (`tracking_only`)

| Target | Best model | Feature set | Metric |
| --- | --- | --- | --- |
| Completion | Gradient boosting | `full` | Balanced accuracy = `0.5480`, AUROC = `0.5808` |
| Explosive | Logistic regression | `full` | Balanced accuracy = `0.6620`, AUROC = `0.7464` |
| Success | Gradient boosting | `context_only` | Balanced accuracy = `0.5431`, AUROC = `0.5776` |
| EPA | Ridge regression | `full` | RMSE = `1.5919` |

The `tracking_only` slice contains only `143` test plays and should be treated as directional, not conclusive.

## Headline findings

1. The clearest overall motion signal is on completion.
   After context controls, motion is associated with a `+0.0305` increase in completion probability on the test split.
2. Motion effects are unclear for success, explosive rate, and EPA.
   The adjusted motion-effect summary is not strong enough to call those targets positive or negative overall.
3. Motion-related features still help some predictive tasks even when the top-line adjusted effect is mixed.
   `context_plus_motion` wins for explosive prediction, and `full` wins for completion prediction.
4. EPA is best modeled without relying on motion/tracking additions.
   The best EPA model is gradient boosting with `context_only`, which is consistent with the idea that the current motion features do not improve this broader outcome.
5. Defensive-response findings remain limited by holdout tracking coverage.
   Tracking coverage is strong in train but only `0.7%` in test, so any defense-reaction claims should be framed as exploratory.

## Overall motion effect

| Target | Adjusted effect | 95% CI | Interpretation |
| --- | ---: | --- | --- |
| Success | `+0.0036` | `[-0.0032, 0.0060]` | Unclear |
| Explosive | `-0.0001` | `[0.0032, 0.0090]` | Unclear |
| Completion | `+0.0305` | `[0.0241, 0.0404]` | Helps |
| EPA | `-0.0018` | `[-0.0241, 0.0149]` | Unclear |

These effects are estimated after controlling for:

- `down_bucket`
- `distance_bucket`
- `field_zone`
- `score_state`

## Feature-group interpretation

- `completion`: the best final model uses `full`, and motion shows the clearest positive adjusted effect here
- `explosive`: the best final model uses `context_plus_motion`, suggesting motion/context features help more than tracking-response features
- `success`: the best final model uses `full`, but the adjusted motion effect remains unclear
- `epa`: the best final model uses `context_only`, which suggests the current motion and tracking additions do not improve this target

## Bottom line

The strongest defensible final claim is:

> Pre-snap motion shows its clearest positive relationship with completion, while its overall effect on success, explosive plays, and EPA is unclear under the current controls and dataset.

The project is strongest as a robust experimental framework with transparent ablations, validation-aware model selection, and interpretable motion-effect summaries. It is not strongest as a high-accuracy deployment model or as a definitive tracking-based study of defensive reaction.

## Final source files

- `artifacts/motion-value-v2-final/metrics/proposal_summary.md`
- `artifacts/motion-value-v2-final/metrics/selected_models.csv`
- `artifacts/motion-value-v2-final/metrics/motion_effect_overall.csv`
- `artifacts/motion-value-v2-final/metrics/motion_lift_overall.csv`
- `artifacts/motion-value-v2-final/metrics/defensive_reaction_overall.csv`
- `artifacts/motion-value-v2-final/metrics/dataset_summary.json`
- `artifacts/motion-value-v2-final/metrics/season_summary.csv`
