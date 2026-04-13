# Contributing

## Workflow

1. Create a short-lived branch for a single task area.
2. Make changes small and reviewable.
3. Run the most relevant local verification before opening a pull request.
4. Summarize what changed, how it was tested, and any known gaps.

## Recommended Commands

Use the centralized runner for repeatable experiments:

```powershell
python scripts/run_experiment.py --command inspect --config configs/tracking_experiment.yaml
python scripts/run_experiment.py --command prepare --config configs/tracking_experiment.yaml
python scripts/run_experiment.py --command train --config configs/tracking_experiment.yaml
python scripts/run_experiment.py --command run --config configs/quickstart.yaml
```

For lightweight verification:

```powershell
.\.py312\runtime\python.exe -m compileall src tests scripts
```

If `pytest` is installed in your environment:

```powershell
pytest
```

## Repo Conventions

- Keep proposal-facing experiment configs in `configs/`.
- Keep reusable pipeline code in `src/pre_snap_motion/`.
- Do not commit local datasets, cached artifacts, or local Python runtimes.
- Treat tracking conclusions carefully when holdout coverage is sparse.

## Pull Request Checklist

- The branch has a focused scope.
- The README or docs were updated if workflow or outputs changed.
- New configs have distinct `project_name` values.
- Generated artifacts are not included in the commit.
- The verification steps you ran are listed in the PR description.
