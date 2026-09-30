# Real Training Results

## Run provenance

- Workflow: `Real Dataset Training #6`
- Training configuration: frozen VGG16 baseline, 5 epochs, batch size 16, seed 42
- Dataset subset: 500 NORMAL + 500 PNEUMONIA-positive studies selected from the NIH ChestX-ray14-derived source
- Split: 650 train / 175 validation / 175 test images
- Split strategy: patient-level
- MLflow run ID: `e2074308b47d45d692bdbc9fbdec4efb`

> These results are from this evaluated subset and are not clinical validation or evidence of clinical safety.

## Test-set metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.5543 |
| Macro F1 | 0.5536 |
| Weighted F1 | 0.5573 |
| ROC-AUC | 0.6031 |
| PR-AUC | 0.5476 |

## Pneumonia-positive class

| Metric | Result |
|---|---:|
| Sensitivity / Recall | 0.6522 |
| Specificity | 0.4906 |
| Precision | 0.4545 |
| F1 | 0.5357 |
| TP | 45 |
| FN | 24 |
| FP | 54 |
| TN | 52 |

## Confusion matrix

```text
                Pred NORMAL  Pred PNEUMONIA
Actual NORMAL         52             54
Actual PNEUMONIA      24             45
```

## Threshold analysis

- Minimum specificity constraint: 0.80
- Selected threshold: 0.79
- Sensitivity at selected threshold: 0.3768
- Specificity at selected threshold: 0.8208

## Interpretation

The frozen VGG16 configuration is a baseline implementation rather than a tuned or clinically validated model. The results show why the project evaluates sensitivity, specificity, ROC-AUC, PR-AUC, calibration, and threshold behavior instead of presenting accuracy alone.

Source artifact: successful GitHub Actions run `Real Dataset Training #6`.