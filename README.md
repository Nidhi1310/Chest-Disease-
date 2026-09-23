# Chest Disease Classification

Day 1 - project setup and design foundation.

This project is an educational transfer-learning baseline for NORMAL vs PNEUMONIA chest X-ray classification.

> Not a clinical diagnostic system.

Planned pipeline: patient-level split -> preprocessing -> VGG16 transfer learning -> training -> medical evaluation -> prediction API.

See DESIGN.md, PREPROCESSING.md, and METRICS.md.

## Day 1 implementation

The repository now includes a leakage-safe patient-level splitter at `src/split_strategy/patient_level_split.py`, unit tests under `tests/`, an example patient mapping under `examples/`, and a GitHub Actions test workflow.

The splitter produces train/validation/test CSVs and a data integrity report while validating that no patient appears in more than one split.
