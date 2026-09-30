from pathlib import Path

import numpy as np
from PIL import Image

from src.training.image_sequence import ManifestImageSequence


def test_manifest_sequence_loads_and_preprocesses_images(tmp_path: Path):
    for index in range(3):
        Image.new("RGB", (64, 64), color=(index * 50, 100, 150)).save(
            tmp_path / f"image_{index}.png"
        )

    rows = [
        {"image_path": "image_0.png", "patient_id": "p0", "label": "NORMAL"},
        {"image_path": "image_1.png", "patient_id": "p1", "label": "PNEUMONIA"},
        {"image_path": "image_2.png", "patient_id": "p2", "label": "1"},
    ]

    sequence = ManifestImageSequence(
        rows,
        dataset_root=tmp_path,
        batch_size=2,
        shuffle=False,
    )

    images, labels = sequence[0]

    assert images.shape == (2, 224, 224, 3)
    assert images.dtype == np.float32
    assert labels.tolist() == [0, 1]
    assert len(sequence) == 2


def test_manifest_sequence_rejects_unknown_labels(tmp_path: Path):
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=128).save(image)

    rows = [{"image_path": "image.png", "patient_id": "p0", "label": "UNKNOWN"}]

    try:
        ManifestImageSequence(rows, dataset_root=tmp_path)
    except ValueError as exc:
        assert "Unsupported label" in str(exc)
    else:
        raise AssertionError("Unknown labels must be rejected")
