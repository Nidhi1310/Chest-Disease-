# Tests for patient-level splitting.

import pytest

from src.split_strategy.patient_level_split import load_mapping, split_patient_level


def make_rows(patient_count=20):
    rows = []
    for number in range(1, patient_count + 1):
        patient_id = f"p{number:03d}"
        label = "PNEUMONIA" if number % 2 == 0 else "NORMAL"
        rows.append({
            "image_path": f"images/{patient_id}_1.jpg",
            "patient_id": patient_id,
            "label": label,
        })
        rows.append({
            "image_path": f"images/{patient_id}_2.jpg",
            "patient_id": patient_id,
            "label": label,
        })
    return rows


def test_no_patient_leakage():
    splits = split_patient_level(make_rows(), seed=42)
    train = {row["patient_id"] for row in splits["train"]}
    validation = {row["patient_id"] for row in splits["validation"]}
    test = {row["patient_id"] for row in splits["test"]}
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_all_rows_are_preserved():
    rows = make_rows()
    splits = split_patient_level(rows, seed=42)
    original = {row["image_path"] for row in rows}
    split_paths = {
        row["image_path"]
        for split in ("train", "validation", "test")
        for row in splits[split]
    }
    assert split_paths == original


def test_split_is_deterministic():
    rows = make_rows()
    first = split_patient_level(rows, seed=123)
    second = split_patient_level(rows, seed=123)
    assert first == second


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError, match="summing to 1.0"):
        split_patient_level(make_rows(), ratios=(0.8, 0.1, 0.2))


def test_missing_mapping_columns_are_rejected(tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("image_path,patient_id\nimg.jpg,p1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_mapping(mapping)
