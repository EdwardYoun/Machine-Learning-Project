# Graph Guide

Use this workflow to regenerate all presentation charts from the final V2 artifact snapshot.

## Command

```powershell
python scripts/generate_report_graphs.py --metrics-dir artifacts/metrics --output-dir reports/figures
```

## Output files

The script writes a full chart set to `reports/figures/`:

- `01_selected_models.png`
- `02_motion_effect_overall.png`
- `03_motion_lift_classification.png`
- `04_tracking_response_lift.png`
- `05_tracking_coverage_by_season.png`
- `06_dataset_snapshot.png`
- `07_validation_vs_test_selected_models.png`
- `08_completion_motion_effect_subgroups.png`
- `09_completion_motion_lift_subgroups.png`
- `10_target_rates_by_season.png`
- `11_classification_leaderboard.png`
- `chart_manifest.md`

## Best charts for the final deck

- `01_selected_models.png`
- `02_motion_effect_overall.png`
- `05_tracking_coverage_by_season.png`
- `06_dataset_snapshot.png`

## Backup / appendix charts

- `03_motion_lift_classification.png`
- `04_tracking_response_lift.png`
- `07_validation_vs_test_selected_models.png`
- `08_completion_motion_effect_subgroups.png`
- `09_completion_motion_lift_subgroups.png`
- `10_target_rates_by_season.png`
- `11_classification_leaderboard.png`

## Notes

- The plotting script expects the final snapshot files currently stored in `artifacts/metrics/`.
- For the academic rerun, point the same script at `artifacts/motion-value-v2-academic/metrics/` and write to a separate output folder such as `reports/figures_academic/`.
- The defensive reaction CSV is currently empty, so there is no dedicated defensive reaction plot in this first pass.
- `07_validation_vs_test_selected_models.png` only appears when the metrics contain a genuine validation-selected score. That chart is intentionally omitted for test-selected snapshots.
- If newer metrics are generated later, rerun the same command and the figures will refresh automatically.
