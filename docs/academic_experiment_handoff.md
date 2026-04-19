# Academic Experiment Handoff

This is the class-oriented rerun design for the proposal question.

## Recommended config

- Academic config: `configs/motion_value_v2_academic.yaml`
- Project output directory: `artifacts/motion-value-v2-academic/`
- Processed dataset: `data/processed/motion-value-v2-academic_passing_motion_modeling_dataset.parquet`

## Why this config exists

The historical final config selected models directly from the `2024` test season because it had no separate validation split.

The academic config keeps:

- `2024` as the untouched final test season
- `2023` as the development season

And adds:

- rolling week-based validation inside `2023`
- threshold tuning from those validation folds
- test reporting only after model selection is complete

## Interpretation

Use this config when the goal is a defensible class-project evaluation rather than matching the original Kaggle competition workflow.

The intended evidence hierarchy is:

1. primary: out-of-season `2023 -> 2024` generalization
2. secondary: exploratory tracking-response analysis with explicit caveats

## Run order

```powershell
python scripts/run_experiment.py --command fetch --config configs/motion_value_v2_academic.yaml
python scripts/run_experiment.py --command prepare --config configs/motion_value_v2_academic.yaml
python scripts/run_experiment.py --command train --config configs/motion_value_v2_academic.yaml
python scripts/generate_report_graphs.py --metrics-dir artifacts/motion-value-v2-academic/metrics --output-dir reports/figures_academic
```

## Expected output differences

- `selected_models.csv` should show `selection_split=validation`
- `validation_metrics.csv` should contain real validation rows
- `07_validation_vs_test_selected_models.png` should render again because there is now a genuine validation selection metric
- tracking-based conclusions may still remain limited if the available public Kaggle bundle is sparse
