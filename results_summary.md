# Results Summary

This summary reflects the current outputs in `artifacts/reading-the-defense-tracking/metrics/` from `configs/tracking_experiment.yaml`, which trains on the 2023 season and tests on the 2024 season.

## Dataset and coverage

- Total pass-play rows: 40,809
- Train rows: 20,693
- Test rows: 20,116
- FTN charting coverage: 100.0%
- Overall tracking coverage: 34.9%
- Train tracking coverage: 68.2% (14,107 rows)
- Test tracking coverage: 0.7% (143 rows)
- Overall motion rate: 45.7%

| Season | Rows | Tracking coverage | Motion rate | Success rate | Explosive rate | Completion rate | Mean EPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 20,693 | 68.2% | 42.0% | 0.444 | 0.082 | 0.571 | -0.001 |
| 2024 | 20,116 | 0.7% | 49.5% | 0.458 | 0.082 | 0.578 | 0.052 |

## Headline findings

1. Motion features help completion prediction on the full test set.
   The full model beats the `no_motion` ablation for completion in both families: logistic regression improves from 0.569 to 0.594 AUROC (+0.025), and random forest improves from 0.568 to 0.590 (+0.022).
2. Motion has only modest value for explosive-play prediction.
   Logistic regression improves slightly from 0.596 to 0.602 AUROC (+0.006), while random forest gets worse from 0.595 to 0.588 (-0.007).
3. Motion hurts the current success and EPA models.
   For success, logistic regression drops from 0.576 to 0.565 AUROC and random forest drops from 0.560 to 0.546 when motion features are added. For EPA, RMSE worsens from 1.691 to 1.734 for ridge regression and from 1.715 to 1.771 for random forest.
4. Simpler models are currently the strongest overall.
   On the full test set, the best models are logistic regression for all three classification targets and ridge regression for EPA.
5. The current tracking-response features are not adding stable value yet.
   On the full test set, completion and explosive prediction are both best with the `no_tracking_response` ablation, and the full-vs-`no_tracking_response` deltas are near zero or negative across most targets.

## Best observed models

### Full test set (`all`)

| Target | Best model | Feature set | Metric |
| --- | --- | --- | --- |
| Completion | Logistic regression | `no_tracking_response` | AUROC = 0.597 |
| Explosive | Logistic regression | `no_tracking_response` | AUROC = 0.608 |
| Success | Logistic regression | `no_motion` | AUROC = 0.576 |
| EPA | Ridge regression | `no_motion` | RMSE = 1.691 |

### Tracking-only test slice (`tracking_only`)

| Target | Best model | Feature set | Metric |
| --- | --- | --- | --- |
| Completion | Random forest | `no_tracking_response` | AUROC = 0.575 |
| Explosive | Random forest | `no_tracking_response` | AUROC = 0.750 |
| Success | Logistic regression | `no_motion` | AUROC = 0.559 |
| EPA | Random forest | `full` | RMSE = 1.569 |

The `tracking_only` slice should be treated as directional rather than conclusive because it contains only 143 test plays.

## Subgroup patterns

- Completion is where motion helps most consistently. On the full test set, the largest broad-sample gains for logistic regression show up on early downs (+0.040 AUROC), in scoring range (+0.036), in the red zone (+0.035), on long-yardage plays (+0.032), and against standard boxes (+0.032).
- Success is where motion hurts most consistently. The largest broad-sample drops for logistic regression appear in one-score games (-0.015 AUROC), on long-yardage plays (-0.014), when backed up (-0.016), against standard boxes (-0.013), and on early downs (-0.011).
- Explosive-play gains are real but small. On the full test set, the best broad-sample logistic-regression subgroup lifts are only around +0.006 to +0.008 AUROC, including one-score games (+0.008), medium distance (+0.008), scoring range (+0.008), and early downs (+0.007).
- Tracking-response lifts do not show a stable pattern on the full test set. Outside tiny subgroups, the deltas are generally small, and several of the larger broad-sample changes are negative for explosive prediction, which suggests the current response features are not yet carrying robust holdout signal.

## Interpretation so far

- The clearest result is that pre-snap motion information helps explain completion probability more than it helps explain overall success or EPA in the current feature set.
- The fact that `no_motion` wins for success and EPA suggests the current motion features may be capturing descriptive context without adding enough incremental signal for those harder targets.
- The fact that `no_tracking_response` often matches or beats `full` suggests the current defensive-response tracking features are still too weak, too sparse, or both.
- Because the 2024 test split has only 143 tracking rows, we should treat any tracking-specific claim as provisional until we run a denser tracking backtest.

## Bottom line

So far, the strongest defensible claim is that motion helps completion prediction, offers only limited help for explosive-play prediction, and does not currently improve success or EPA prediction. We do not yet have strong evidence that the current tracking-response features improve holdout performance, and the sparse 2024 tracking coverage is the main reason to frame those tracking findings as preliminary.

## Source files

- `artifacts/reading-the-defense-tracking/metrics/dataset_summary.json`
- `artifacts/reading-the-defense-tracking/metrics/season_summary.csv`
- `artifacts/reading-the-defense-tracking/metrics/overall_metrics.csv`
- `artifacts/reading-the-defense-tracking/metrics/best_models.csv`
- `artifacts/reading-the-defense-tracking/metrics/motion_lift_overall.csv`
- `artifacts/reading-the-defense-tracking/metrics/motion_lift_subgroups.csv`
- `artifacts/reading-the-defense-tracking/metrics/tracking_response_lift_overall.csv`
- `artifacts/reading-the-defense-tracking/metrics/tracking_response_lift_subgroups.csv`
