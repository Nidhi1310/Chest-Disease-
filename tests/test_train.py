from pathlib import Path

import pytest

from scripts.train import load_manifest, validate_manifest_integrity


def write_manifest(path: Path, rows: str) -> None:
    path.write_text("image_path,patient_id,label\n" + rows, encoding="utf-8")


def test_validate_manifest_integrity_rejects_patient_overlap():
    train = [{"image_path": "a.png", "patient_id": "p1", "label": "NORMAL"}]
    validation = [{"image_path": "b.png", "patient_id": "p1", "label": "PNEUMONIA"}]
    with pytest.raises(ValueError, match="Patient leakage detected"):
        validate_manifest_integrity([("train", train), ("validation", validation), ("test", [])])


def test_validate_manifest_integrity_rejects_image_overlap():
    train = [{"image_path": "a.png", "patient_id": "p1", "label": "NORMAL"}]
    test = [{"image_path": "a.png", "patient_id": "p2", "label": "PNEUMONIA"}]
    with pytest.raises(ValueError, match="Image leakage detected"):
        validate_manifest_integrity([("train", train), ("validation", [{"image_path": "b.png", "patient_id": "p3", "label": "NORMAL"}]), ("test", test)])


def test_load_manifest_requires_patient_id(tmp_path: Path):
    path = tmp_path / "train.csv"
    path.write_text("image_path,label\na.png,NORMAL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(path)
