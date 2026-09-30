# Day 6 - Medical Evaluation and Rigor

## Implemented

- `MedicalEvaluator` for binary NORMAL vs PNEUMONIA evaluation
- Per-class sensitivity, specificity, precision, and F1
- Aggregate accuracy, macro F1, and weighted F1
- Confusion matrix
- ROC-AUC
- PR-AUC
- Maximum-confidence calibration
- PNEUMONIA-probability calibration
- Decision-threshold sensitivity/specificity analysis
- Human-readable text report plus JSON report
- Unit tests using deterministic synthetic predictions

## Why sensitivity and specificity matter

Accuracy alone can hide clinically important class-specific errors. The evaluator therefore reports sensitivity and specificity for both classes and exposes the underlying TP/TN/FP/FN counts.

## Calibration distinction

The evaluator deliberately tracks two related but different quantities:

1. **Maximum-confidence calibration:** whether the model's stated confidence matches empirical correctness.
2. **PNEUMONIA-probability calibration:** whether the probability assigned to PNEUMONIA matches the observed pneumonia rate.

Threshold analysis uses the PNEUMONIA probability (`predictions[:, 1]`) rather than the maximum class confidence.

## Threshold analysis

Thresholds from `0.0` through `1.0` are evaluated by default. The report identifies the threshold with the highest sensitivity while meeting a configurable minimum specificity (default `0.80`). This is an evaluation aid, not a clinically validated operating point.

## Run

```bash
python -m pytest -q
```

Example report generation after a trained model is available:

```python
from src.evaluation.medical_evaluator import MedicalEvaluator

evaluator = MedicalEvaluator(model, test_images, test_labels)
evaluator.generate_report("artifacts/day6/medical_evaluation.txt")
```

## Important boundary

Day 6 reports actual test-set metrics only. The repository does not contain fabricated performance numbers, and the metrics are not presented as evidence of clinical safety or deployment readiness.
