# Motion Value Summary

## Dataset
- Total pass-play rows: 48
- Train rows: 36
- Test rows: 12
- Seasons included: 2023, 2024
- FTN charting coverage: 100.0%
- Tracking coverage: 0.0%
- Tracking coverage by split: train 0.0%, test 0.0%
- Tracking note: test-split coverage is sparse, so defensive-response conclusions should be treated as directional.
- Motion rate: 50.0%
- Selection design: rolling week-based validation inside the training season (3 folds, min train weeks=3, validation window=1 week(s))

## Target Rates
- success: 0.500
- explosive: 0.500
- completion: 0.500
- epa: 0.500

## Validation-Selected Models
- [all] classification / completion: logistic_regression with `context_plus_motion` (validation_balanced_accuracy=1.0000)
- [all] classification / explosive: logistic_regression with `context_plus_motion` (validation_balanced_accuracy=1.0000)
- [all] classification / success: logistic_regression with `context_plus_motion` (validation_balanced_accuracy=1.0000)
- [all] regression / epa: ridge_regression with `context_plus_motion` (validation_rmse=0.0039, test_rmse=0.0023)

## Motion Lift
- [all] test / classification / completion / logistic_regression: auroc lift=0.5000
- [all] test / classification / explosive / logistic_regression: auroc lift=0.5000
- [all] test / classification / success / logistic_regression: auroc lift=0.5000
- [all] test / regression / epa / ridge_regression: rmse lift=0.4977

## Defensive Response Contribution
- [all] test / classification / completion / logistic_regression: auroc lift=0.0000
- [all] test / classification / explosive / logistic_regression: auroc lift=0.0000
- [all] test / classification / success / logistic_regression: auroc lift=0.0000
- [all] test / regression / epa / ridge_regression: rmse lift=0.0000
