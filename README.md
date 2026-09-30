# Chest Disease Classification

**Leakage-aware chest X-ray classification pipeline using VGG16 transfer learning, medical-focused evaluation, experiment tracking, and a prediction API.**

> **Educational project — not a clinical diagnostic system.**

[![Project Tests](https://github.com/Nidhi1310/Chest-Disease-/actions/workflows/day1-tests.yml/badge.svg)](https://github.com/Nidhi1310/Chest-Disease-/actions/workflows/day1-tests.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-black)

## Overview

This project is an end-to-end transfer-learning baseline for **NORMAL vs PNEUMONIA chest X-ray classification**.

The pipeline is designed around three principles:

- **Prevent data leakage** with patient-level dataset splitting.
- **Keep training and inference preprocessing identical** using the official VGG16 preprocessing routine.
- **Evaluate medical performance beyond accuracy** using sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC, calibration, and threshold analysis.

### Pipeline

```text
Dataset
   |
   v
Patient-level split
   |
   v
VGG16 preprocessing
   |
   v
Frozen VGG16 + classification head
   |
   v
Training + MLflow tracking
   |
   v
Medical evaluation
   |
   v
Flask prediction API
```

## Current Development Status

The repository currently contains the planned implementation through **Day 7**.

| Stage | Focus | Status |
|---|---|---|
| Day 1 | Patient-level splitting & leakage prevention | Implemented |
| Day 2 | VGG16-specific preprocessing | Implemented |
| Day 3 | Reusable patient-level data pipeline | Implemented |
| Day 4 | VGG16 transfer-learning baseline | Implemented |
| Day 5 | MLflow experiment tracking & reproducibility | Implemented |
| Day 6 | Medical evaluation & threshold analysis | Implemented |
| Day 7 | Prediction API & production interface | Implemented |

> **Important:** implementation does not imply final model performance. No accuracy, sensitivity, or AUC is claimed until the model is trained and evaluated on the real dataset.

## Model Architecture

The baseline intentionally uses a simple, explainable transfer-learning design:

```text
Input: 224 × 224 × 3
        |
        v
VGG16 (ImageNet weights)
Frozen backbone
        |
        v
GlobalAveragePooling2D
        |
        v
Dense(128, ReLU)
        |
        v
Dropout(0.3)
        |
        v
Dense(2, Softmax)
```

The VGG16 backbone is frozen for the baseline experiment. Medical evaluation is separated from the training metrics so that accuracy is not treated as the complete assessment.

See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

## Live Demo

**Frontend:** https://chest-disease-ai.onrender.com  
**Prediction API:** https://chest-disease-api.onrender.com  
**API docs:** https://chest-disease-api.onrender.com/docs  
**Health:** https://chest-disease-api.onrender.com/health

The frontend is connected to the deployed Flask prediction API. The deployment has been verified end-to-end with a live chest X-ray prediction smoke test.

## Real Training Results

The first successful real-data baseline run was completed through GitHub Actions using 1,000 selected studies and a patient-level split. The resulting held-out test set contains 175 images.

| Metric | Test result |
|---|---:|
| Accuracy | 55.43% |
| Macro F1 | 0.5536 |
| ROC-AUC | 0.6031 |
| PR-AUC | 0.5476 |
| Pneumonia sensitivity | 65.22% |
| Pneumonia specificity | 49.06% |

These are baseline results on this evaluated subset, not clinical validation. Full results and the confusion matrix are documented in [RESULTS.md](RESULTS.md).

## Medical Evaluation

Day 6 provides a reusable evaluation layer for binary medical classification.

- Sensitivity / Recall
- Specificity
- Precision
- F1 score
- Confusion matrix
- ROC-AUC
- PR-AUC
- Calibration
- Probability threshold analysis

The default threshold analysis searches for a threshold that maximizes sensitivity subject to a minimum specificity constraint.

See [METRICS.md](METRICS.md).

## Experiment Tracking

The training layer uses **MLflow** to record:

- Explicit model/training parameters
- Per-epoch metrics
- Model artifacts
- Reproducibility metadata
- Comparable experiment runs

The repository intentionally does **not** fabricate training results. Actual metrics are recorded only after training on the real dataset.

## Reproducible Training

The repository includes a real-dataset training entry point at `scripts/train.py`. It consumes the validated Day 3 train/validation/test manifests and re-checks patient/image separation before calling `model.fit()`.

Expected manifest columns:

```text
image_path,patient_id,label
```

Example:

```bash
python -m scripts.train \\
  --train-manifest data/splits/train.csv \\
  --validation-manifest data/splits/validation.csv \\
  --test-manifest data/splits/test.csv \\
  --dataset-root /path/to/dataset
```

The run trains the frozen VGG16 baseline, records the experiment in MLflow, saves the Keras model, and generates the Day 6 medical evaluation report.

No patient identifiers are inferred from filenames, and no performance values are claimed until this command is run against the real dataset.

## Prediction API

The Day 7 Flask API exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and model readiness |
| `GET /info` | Model input/output contract |
| `GET /docs` | API contract |
| `POST /predict` | Image classification |

The API validates image content with Pillow, applies dimension/mode checks, cleans temporary uploads, and returns sanitized client-facing errors.

For production serving, the project uses Gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

See [docs/DAY-7-IMPLEMENTATION.md](docs/DAY-7-IMPLEMENTATION.md).

## Repository Structure

```text
Chest-Disease-/
├── src/
│   ├── api/                 # Prediction API utilities
│   ├── data_pipeline/       # Reusable dataset splitting pipeline
│   ├── evaluation/          # Medical evaluation
│   ├── models/              # VGG16 model definition
│   ├── preprocessing.py     # Official VGG16 preprocessing
│   ├── split_strategy/      # Patient-level split logic
│   └── training/            # MLflow + reproducibility
│
├── tests/                   # Automated tests
├── docs/                    # Day-by-day implementation notes
├── examples/                # Example patient mappings
├── app.py                   # Flask application entry point
├── Dockerfile               # Containerized API serving
├── requirements.txt         # Python dependencies
├── DESIGN.md                # Data leakage and design decisions
├── PREPROCESSING.md         # Preprocessing contract
├── METRICS.md               # Evaluation definitions
└── MODEL_ARCHITECTURE.md    # Model design and rationale
```

## Dataset Provenance

The real-data training workflow uses the public **NIH ChestX-ray14-derived NIHChestXR dataset** published by Chehab Lab on Hugging Face. The dataset provides `patient_id`, `scan_id`, image data, and multi-label disease annotations; the published dataset contains 112,120 images from 30,805 unique patients. The project uses **No Finding** as the NORMAL class and **Pneumonia-positive** studies as the PNEUMONIA class, while retaining the source patient ID for leakage-safe splitting. citeturn304556search0

For reproducibility, `scripts/prepare_public_dataset.py` selects a manageable balanced subset and writes the required `image_path,patient_id,label` mapping. The GitHub Actions workflow then performs the Day 3 patient-level split, trains the Day 4 model, runs Day 6 evaluation, and uploads the resulting model and reports as workflow artifacts.

## Dataset & Leakage Prevention

The splitter expects records containing:

```text
image_path, patient_id, label
```

Patients must belong to **exactly one** of train, validation, or test.

The data pipeline produces split CSV files and an integrity report and checks for patient overlap.

> Patient identifiers are required for a trustworthy patient-level split. The project does not invent patient IDs from image filenames.

The dataset itself is not included in this repository.

## Getting Started

Create a Python environment and install the declared dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

TensorFlow 2.21.0 is used with Python 3.13 in the current development environment.

## Design Documentation

| Document | Purpose |
|---|---|
| [DESIGN.md](DESIGN.md) | Dataset and leakage decisions |
| [PREPROCESSING.md](PREPROCESSING.md) | VGG16 preprocessing contract |
| [METRICS.md](METRICS.md) | Medical evaluation definitions |
| [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) | Baseline model architecture |
| [docs/DAY-1.md](docs/DAY-1.md) | Day 1 implementation |
| [docs/DAY-4-IMPLEMENTATION.md](docs/DAY-4-IMPLEMENTATION.md) | Day 4 model implementation |
| [docs/DAY-5-IMPLEMENTATION.md](docs/DAY-5-IMPLEMENTATION.md) | Day 5 MLflow implementation |
| [docs/DAY-6-IMPLEMENTATION.md](docs/DAY-6-IMPLEMENTATION.md) | Day 6 evaluation implementation |
| [docs/DAY-7-IMPLEMENTATION.md](docs/DAY-7-IMPLEMENTATION.md) | Day 7 API implementation |

## Project Boundary

This repository demonstrates an engineering workflow for medical image classification:

**data integrity → preprocessing consistency → baseline modeling → experiment tracking → medical evaluation → API serving**

It is an educational engineering project and should not be used to make clinical decisions.
