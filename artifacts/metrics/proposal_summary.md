# Motion Value Summary

## Dataset
- Total pass-play rows: 40,809
- Train rows: 20,693
- Test rows: 20,116
- Seasons included: 2023, 2024
- FTN charting coverage: 100.0%
- Tracking coverage: 34.9%
- Tracking coverage by split: train 68.2%, test 0.7%
- Tracking note: test-split coverage is sparse, so defensive-response conclusions should be treated as directional.
- Motion rate: 45.7%

## Target Rates
- success: 0.451
- explosive: 0.082
- completion: 0.574
- epa: 0.025

## Overall Motion Effect
- test / success: motion unclear by 0.0036 (CI -0.0032 to 0.0060) after context controls (18,690 rows across 33 groups).
- test / explosive: motion unclear by -0.0001 (CI 0.0032 to 0.0090) after context controls (18,690 rows across 33 groups).
- test / completion: motion helps by 0.0305 (CI 0.0241 to 0.0404) after context controls (18,690 rows across 33 groups).
- test / epa: motion unclear by -0.0018 (CI -0.0241 to 0.0149) after context controls (18,690 rows across 33 groups).

## Selected Models
- [all] classification / completion: gradient_boosting with `full` (test_balanced_accuracy=0.5543)
- [all] classification / explosive: logistic_regression with `context_plus_motion` (test_balanced_accuracy=0.5117)
- [all] classification / success: logistic_regression with `full` (test_balanced_accuracy=0.5030)
- [all] regression / epa: gradient_boosting with `context_only` (test_rmse=1.6686)
- [tracking_only] classification / completion: gradient_boosting with `full` (test_balanced_accuracy=0.5480)
- [tracking_only] classification / explosive: logistic_regression with `full` (test_balanced_accuracy=0.6620)
- [tracking_only] classification / success: gradient_boosting with `context_only` (test_balanced_accuracy=0.5431)
- [tracking_only] regression / epa: ridge_regression with `full` (test_rmse=1.5919)

## Motion Lift
- [all] test / classification / completion / gradient_boosting: auroc lift=0.0219
- [all] test / classification / completion / logistic_regression: auroc lift=0.0267
- [all] test / classification / explosive / gradient_boosting: auroc lift=-0.0180
- [all] test / classification / explosive / logistic_regression: auroc lift=0.0126
- [all] test / classification / success / gradient_boosting: auroc lift=-0.0210
- [all] test / classification / success / logistic_regression: auroc lift=-0.0117
- [all] test / regression / epa / gradient_boosting: rmse lift=-0.0653
- [all] test / regression / epa / ridge_regression: rmse lift=-0.0439
- [tracking_only] test / classification / completion / gradient_boosting: auroc lift=-0.0365
- [tracking_only] test / classification / completion / logistic_regression: auroc lift=-0.0606
- [tracking_only] test / classification / explosive / gradient_boosting: auroc lift=0.0122
- [tracking_only] test / classification / explosive / logistic_regression: auroc lift=-0.0133
- [tracking_only] test / classification / success / gradient_boosting: auroc lift=-0.0494
- [tracking_only] test / classification / success / logistic_regression: auroc lift=-0.0508
- [tracking_only] test / regression / epa / gradient_boosting: rmse lift=-0.0293
- [tracking_only] test / regression / epa / ridge_regression: rmse lift=-0.0032

## Defensive Response Contribution
- [all] test / classification / completion / gradient_boosting: auroc lift=-0.0029
- [all] test / classification / completion / logistic_regression: auroc lift=-0.0017
- [all] test / classification / explosive / gradient_boosting: auroc lift=0.0015
- [all] test / classification / explosive / logistic_regression: auroc lift=-0.0075
- [all] test / classification / success / gradient_boosting: auroc lift=0.0053
- [all] test / classification / success / logistic_regression: auroc lift=0.0007
- [all] test / regression / epa / gradient_boosting: rmse lift=0.0122
- [all] test / regression / epa / ridge_regression: rmse lift=-0.0004
- [tracking_only] test / classification / completion / gradient_boosting: auroc lift=0.0106
- [tracking_only] test / classification / completion / logistic_regression: auroc lift=0.0468
- [tracking_only] test / classification / explosive / gradient_boosting: auroc lift=-0.0183
- [tracking_only] test / classification / explosive / logistic_regression: auroc lift=0.0293
- [tracking_only] test / classification / success / gradient_boosting: auroc lift=0.0335
- [tracking_only] test / classification / success / logistic_regression: auroc lift=0.0500
- [tracking_only] test / regression / epa / gradient_boosting: rmse lift=0.0236
- [tracking_only] test / regression / epa / ridge_regression: rmse lift=0.0123
