# Proposal Summary

## Dataset
- Total pass-play rows: 40,809
- Train rows: 20,693
- Test rows: 20,116
- Seasons included: 2023, 2024
- FTN charting coverage: 100.0%
- Tracking coverage: 34.9%
- Motion rate: 45.7%

## Target Rates
- success: 0.451
- explosive: 0.082
- completion: 0.574
- epa: 0.025

## Best Models
- classification / completion: logistic_regression with `no_tracking_response` (auroc=0.5971)
- classification / explosive: logistic_regression with `no_tracking_response` (auroc=0.6080)
- classification / success: logistic_regression with `no_motion` (auroc=0.5760)
- regression / epa: ridge_regression with `no_motion` (rmse=1.6913)

## Motion Lift
- classification / completion / logistic_regression: auroc lift=0.0255
- classification / completion / random_forest: auroc lift=0.0222
- classification / explosive / logistic_regression: auroc lift=0.0062
- classification / explosive / random_forest: auroc lift=-0.0074
- classification / success / logistic_regression: auroc lift=-0.0114
- classification / success / random_forest: auroc lift=-0.0143
- regression / epa / random_forest: rmse lift=-0.0561
- regression / epa / ridge_regression: rmse lift=-0.0430

## Tracking Response Lift
- classification / completion / logistic_regression: auroc lift=-0.0030
- classification / completion / random_forest: auroc lift=0.0003
- classification / explosive / logistic_regression: auroc lift=-0.0055
- classification / explosive / random_forest: auroc lift=0.0044
- classification / success / logistic_regression: auroc lift=0.0001
- classification / success / random_forest: auroc lift=-0.0005
- regression / epa / random_forest: rmse lift=0.0003
- regression / epa / ridge_regression: rmse lift=0.0003
