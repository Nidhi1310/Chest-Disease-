# Medical AI Evaluation Strategy

## Why accuracy alone is insufficient
A single accuracy number can hide class-specific failures. For pneumonia detection, false negatives are particularly important.

## Required metrics
### Sensitivity
TP / (TP + FN). Among disease cases, how many were detected?

### Specificity
TN / (TN + FP). Among normal cases, how many were correctly identified?

### Precision
TP / (TP + FP). When the model predicts the positive class, how often is it correct?

### F1
Combines precision and sensitivity.

## Aggregate evaluation
The later evaluation stage will include per-class metrics, macro and weighted context, ROC-AUC, PR-AUC, confusion matrix, and calibration analysis.

No metric target here should be interpreted as evidence of clinical safety.
