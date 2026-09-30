# Day 5 - Training and MLflow Experiment Tracking

## Implemented

- `ExperimentTracker` with real MLflow tracking
- Explicit parameter logging
- Per-epoch metric logging from Keras `History`
- TensorFlow/Keras model artifact logging
- Run tags for dataset split and preprocessing
- Reproducibility metadata JSON
- Python, NumPy, and TensorFlow seed handling
- Programmatic experiment comparison with `mlflow.search_runs`
- CLI for comparing runs
- Unit tests plus a real local MLflow smoke test

## Tracking workflow

```text
model.fit(...)
     |
     v
Keras History
     |
     v
ExperimentTracker
     |
     +---- parameters
     +---- per-epoch metrics
     +---- reproducibility metadata
     +---- TensorFlow model artifact
     |
     v
MLflow local tracking store
```

## Local tracking

By default MLflow uses its configured tracking URI. For a simple local setup, the MLflow UI can be started against the local tracking directory:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open the local URL shown by MLflow.

## Comparison

```bash
python -m src.training.compare_experiments --experiment chest_disease_clf
```

The comparison utility uses `mlflow.search_runs()` and exposes available parameters/metrics rather than hard-coding example results.

## Reproducibility

Each tracked training run records:

- random seed
- TensorFlow seed
- NumPy seed
- dataset split strategy
- preprocessing strategy
- Python and TensorFlow environment information
- training parameters

## Important distinction

The repository does **not** fabricate accuracy or ROC-AUC values. Day 5 provides the infrastructure to record actual training results when the real dataset is trained. Medical evaluation and ROC-AUC/PR-AUC analysis are handled on Day 6.
