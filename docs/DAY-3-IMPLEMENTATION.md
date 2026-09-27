# Day 3 - Patient-Level Data Pipeline

## Goal

Turn the Day 1 leakage-safe split logic into an explicit, reusable data-pipeline component.

## Implemented

- PatientLevelSplitter class
- Patient-level train/validation/test manifests
- Zero patient overlap validation
- Input image preservation validation
- Split summaries: patients, images, NORMAL, PNEUMONIA
- Minority-class warning when a split is below 5%
- Optional dataset-root image existence validation
- Human-readable integrity report
- Unit tests for Day 3 behavior

## Required real metadata

The repository still does not contain the real chest X-ray dataset.

Before creating real splits, provide a CSV with:

image_path,patient_id,label

Every image needs a trustworthy patient ID.

## Run

python -m src.data_pipeline.data_splitter --mapping path/to/patient_mapping.csv

With image existence validation:

python -m src.data_pipeline.data_splitter --mapping path/to/patient_mapping.csv --dataset-root path/to/dataset

Outputs:

data/splits/train.csv
data/splits/validation.csv
data/splits/test.csv
artifacts/day3/data_integrity_report.txt

## Critical rule

No patient can occur in more than one split.

The pipeline validates this before outputs are accepted.

## Important limitation

If the source dataset does not provide patient IDs, this pipeline must not invent them from filenames or folder names without documented evidence. In that case, the dataset is not yet suitable for the planned patient-level evaluation.
