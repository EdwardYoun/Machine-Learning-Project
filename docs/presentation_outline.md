# 8-Slide Presentation Outline

Use this version as the final project presentation.

- Metrics: `artifacts/motion-value-v2-academic/metrics/`
- Figures: `reports/figures_academic/`
- Goal: present the completed project clearly in `8` slides with no appendix or question buffer

## Slide 1. Problem and Goal

Graph to include:

- None

Content points:

- Title: `Quantifying the Value of Pre-Snap Motion in NFL Passing Plays`
- Motivation: pre-snap motion is common in modern offenses, but its true value is difficult to separate from game context.
- Main objective: estimate whether motion improves passing outcomes after controlling for situation.
- Secondary objective: test whether motion improves predictive performance beyond standard context features.
- Final project result: an academically defensible evaluation on `2023-2024` NFL passing plays.

References:

- NFL Big Data Bowl official overview: https://operations.nfl.com/gameday/analytics/big-data-bowl
- nflreadpy load functions: https://nflreadpy.nflverse.com/api/load_functions/

## Slide 2. Data and Scope

Graph to include:

- `06_dataset_snapshot.png`

Content points:

- Dataset size: `40,809` pass plays.
- Seasons: `2023` and `2024`.
- Sources:
  - nflverse play-by-play
  - FTN charting
  - Big Data Bowl tracking files when available
- Split design:
  - `2023` for training and rolling validation
  - `2024` for final held-out testing
- Key dataset note:
  - train tracking coverage is strong
  - test tracking coverage is sparse

References:

- nflreadr FTN charting docs: https://nflreadr.nflverse.com/reference/load_ftn_charting.html
- FTN charting data dictionary: https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html

## Slide 3. Features, Targets, and Models

Graph to include:

- `10_target_rates_by_season.png`

Content points:

- Feature sets:
  - `context_only`
  - `context_plus_motion`
  - `full`
- Targets:
  - `completion`
  - `success`
  - `explosive`
  - `EPA`
- Models:
  - logistic regression
  - ridge regression
  - gradient boosting
- Reason for the season-rate graph:
  - classification targets are fairly stable across seasons
  - EPA shifts more, which helps explain why it is harder to model

References:

- nflreadpy load functions: https://nflreadpy.nflverse.com/api/load_functions/
- FTN charting data dictionary: https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html

## Slide 4. Evaluation Methodology

Graph to include:

- `07_validation_vs_test_selected_models.png`

Content points:

- Evaluation design: rolling-origin validation inside `2023`.
- Final `2024` test set was not used for model selection.
- Selection metrics:
  - `balanced accuracy` for classification
  - `RMSE` for regression
- Additional diagnostics:
  - `AUROC`
  - `log loss`
  - `Brier score`
  - `expected calibration error`
- Main message of the graph:
  - blue = validation score used for selection
  - gold = final held-out test score
  - this shows the final pipeline is validation-driven rather than test-tuned

References:

- Brodersen et al. (2010), balanced accuracy: https://doi.org/10.1109/ICPR.2010.764
- Fawcett (2006), ROC/AUROC: https://doi.org/10.1016/j.patrec.2005.10.010
- Brier (1950), Brier score: https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

## Slide 5. Main Result: Estimated Motion Effect

Graph to include:

- `02_motion_effect_overall.png`

Content points:

- This is the main result of the project.
- After context controls, motion has the clearest positive effect on `completion`.
- Estimated completion effect:
  - about `+3.05 percentage points`
  - `95% CI: +1.64 to +4.11`
- Motion is not clearly beneficial for:
  - `success`
  - `explosive`
  - `EPA`
- Core claim:
  - motion most reliably helps completion
  - it does not clearly improve every passing outcome

References:

- Project artifact: `artifacts/motion-value-v2-academic/metrics/motion_effect_overall.csv`

## Slide 6. Predictive Value of Motion

Graph to include:

- `03_motion_lift_classification.png`

Content points:

- We also tested whether motion improves prediction, not just average outcome differences.
- Motion provides the clearest predictive lift on `completion`.
- Predictive gains for `success` and `explosive` are smaller and less consistent.
- Interpretation:
  - motion adds useful signal
  - the prediction story is real but modest
- This supports a measured conclusion rather than a high-accuracy forecasting claim.

References:

- Fawcett (2006), ROC/AUROC: https://doi.org/10.1016/j.patrec.2005.10.010
- Brier (1950), Brier score: https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

## Slide 7. Selected Models

Graph to include:

- `01_selected_models.png`

Content points:

- Different targets preferred different winning models and feature sets.
- Final all-play winners:
  - completion: gradient boosting with `full`
  - explosive: logistic regression with `context_only`
  - success: gradient boosting with `context_only`
  - EPA: gradient boosting with `full`
- Interpretation:
  - richer feature sets matter most for completion and EPA
  - simpler context models remain competitive for explosive and success
- This shows the project was target-specific rather than built around a single one-size-fits-all model.

References:

- Project artifact: `artifacts/motion-value-v2-academic/metrics/selected_models.csv`

## Slide 8. Final Takeaways

Graph to include:

- `05_tracking_coverage_by_season.png`

Content points:

- Train-side tracking coverage is strong, but held-out `2024` tracking coverage is very sparse.
- Because of that, the offensive motion story is stronger than the defensive-response tracking story.
- Final conclusions:
  - pre-snap motion most clearly improves `completion`
  - motion provides modest predictive value for `completion`
  - broader effects on `success`, `explosive`, and `EPA` remain inconclusive
- Possible future directions:
  - fuller held-out tracking coverage
  - a dedicated completion-prediction mode
  - broader comparison to published football analytics baselines
- Final sentence:
  - this completed project succeeds best as an offensive motion-value study with a validation-aware experimental design

References:

- NFL Big Data Bowl official overview: https://operations.nfl.com/gameday/analytics/big-data-bowl
- Project artifact: `artifacts/motion-value-v2-academic/metrics/dataset_summary.json`

## Figure Order

1. no graph
2. `reports/figures_academic/06_dataset_snapshot.png`
3. `reports/figures_academic/10_target_rates_by_season.png`
4. `reports/figures_academic/07_validation_vs_test_selected_models.png`
5. `reports/figures_academic/02_motion_effect_overall.png`
6. `reports/figures_academic/03_motion_lift_classification.png`
7. `reports/figures_academic/01_selected_models.png`
8. `reports/figures_academic/05_tracking_coverage_by_season.png`

## Short Works Cited

1. NFL Big Data Bowl official page. `https://operations.nfl.com/gameday/analytics/big-data-bowl`
2. nflreadpy load functions. `https://nflreadpy.nflverse.com/api/load_functions/`
3. nflreadr FTN charting documentation. `https://nflreadr.nflverse.com/reference/load_ftn_charting.html`
4. nflreadr FTN charting data dictionary. `https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html`
5. Fawcett, T. (2006). `An Introduction to ROC Analysis`. DOI: `10.1016/j.patrec.2005.10.010`
6. Brier, G. W. (1950). `Verification of Forecasts Expressed in Terms of Probability`. DOI: `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2`
7. Brodersen, K. H., Ong, C. S., Stephan, K. E., and Buhmann, J. M. (2010). `The Balanced Accuracy and Its Posterior Distribution`. DOI: `10.1109/ICPR.2010.764`
