import csv
from pathlib import Path

from PIL import Image

from scripts.prepare_public_dataset import prepare_subset


class FakeStream:
    def __iter__(self):
        for index in range(2):
            yield {
                "patient_id": str(index + 1),
                "scan_id": 0,
                "labels": [0],
                "image": Image.new("L", (32, 32), color=100 + index),
            }

        for index in range(2):
            yield {
                "patient_id": str(index + 10),
                "scan_id": 0,
                "labels": [7],
                "image": Image.new("L", (32, 32), color=180 + index),
            }


def test_prepare_subset_uses_source_patient_ids(monkeypatch, tmp_path: Path):
    class FakeDatasets:
        @staticmethod
        def load_dataset(*args, **kwargs):
            assert args == ("chehablab/NIHChestXR",)
            assert kwargs == {"split": "train", "streaming": True}
            return FakeStream()

    import sys
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasets())

    counts = prepare_subset(
        output_dir=tmp_path / "images",
        mapping_path=tmp_path / "mapping.csv",
        per_class=2,
    )

    assert counts == {"NORMAL": 2, "PNEUMONIA": 2}
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 4

    with (tmp_path / "mapping.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["patient_id"] for row in rows] == ["1", "2", "10", "11"]
    assert [row["label"] for row in rows] == [
        "NORMAL",
        "NORMAL",
        "PNEUMONIA",
        "PNEUMONIA",
    ]
