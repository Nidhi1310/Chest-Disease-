# Day 1 Implementation

## Implemented

- Patient-level dataset splitting.
- Required mapping validation.
- Deterministic behavior with a random seed.
- Zero-overlap patient leakage validation.
- Train/validation/test CSV generation.
- Data integrity report generation.
- Unit tests for leakage, row preservation, determinism, and invalid input.

## Required input

The splitter expects columns: image_path, patient_id, label.

## Run locally

Create environment: `python -m venv .venv`

Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Install: `pip install -r requirements.txt`

Run tests: `pytest -q`

Run the split when real metadata exists:
`python -m src.split_strategy.patient_level_split --mapping path/to/patient_mapping.csv`

Default outputs:
- data/splits/train.csv
- data/splits/validation.csv
- data/splits/test.csv
- artifacts/day1/data_integrity_report.txt

## Current limitation

The repository contains only example metadata, not the real chest X-ray dataset or verified patient IDs. Do not claim real patient counts or model results until the real mapping is supplied.

## Day 1 boundary

The splitter is implemented now. VGG16 preprocessing, model training, MLflow, medical evaluation, and the API remain later days.
