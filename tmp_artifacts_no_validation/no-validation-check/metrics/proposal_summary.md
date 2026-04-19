# Motion Value Summary

## Dataset
- Total pass-play rows: 16
- Train rows: 8
- Test rows: 8
- Seasons included: 2023, 2024
- FTN charting coverage: 100.0%
- Tracking coverage: 0.0%
- Tracking coverage by split: train 0.0%, test 0.0%
- Tracking note: test-split coverage is sparse, so defensive-response conclusions should be treated as directional.
- Motion rate: 50.0%

## Target Rates
- success: 0.500
- explosive: 0.500
- completion: 0.500
- epa: 0.500

## Selected Models
- [all] classification / completion: logistic_regression with `context_plus_motion` (test_balanced_accuracy=1.0000)
- [all] classification / explosive: logistic_regression with `context_plus_motion` (test_balanced_accuracy=1.0000)
- [all] classification / success: logistic_regression with `context_plus_motion` (test_balanced_accuracy=1.0000)
- [all] regression / epa: ridge_regression with `context_plus_motion` (test_rmse=0.0111)

## Motion Lift
- [all] test / classification / completion / logistic_regression: auroc lift=0.2500
- [all] test / classification / explosive / logistic_regression: auroc lift=0.2500
- [all] test / classification / success / logistic_regression: auroc lift=0.2500
- [all] test / regression / epa / ridge_regression: rmse lift=0.4370

## Defensive Response Contribution
- [all] test / classification / completion / logistic_regression: auroc lift=0.0000
- [all] test / classification / explosive / logistic_regression: auroc lift=0.0000
- [all] test / classification / success / logistic_regression: auroc lift=0.0000
- [all] test / regression / epa / ridge_regression: rmse lift=0.0000
