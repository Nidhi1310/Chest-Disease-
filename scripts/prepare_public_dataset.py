"""Prepare a reproducible NORMAL-vs-PNEUMONIA subset from NIH ChestX-ray14.

Source: chehablab/NIHChestXR on Hugging Face.
The dataset supplies patient_id/scan_id metadata, so patient identifiers are
never inferred from filenames.

This script downloads only the selected rows through Hugging Face streaming,
writes JPEGs plus a mapping CSV, and leaves the patient-level split to the
project's Day 3 splitter.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

NO_FINDING_LABEL = 0
PNEUMONIA_LABEL = 7


def prepare_subset(
    *,
    output_dir: str | Path,
    mapping_path: str | Path,
    per_class: int = 500,
) -> dict[str, int]:
    if per_class < 1:
        raise ValueError("per_class must be at least 1.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with "
            "'python -m pip install datasets'."
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping_file = Path(mapping_path)
    mapping_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "chehablab/NIHChestXR",
        split="train",
        streaming=True,
    )

    selected = []
    counts = {"NORMAL": 0, "PNEUMONIA": 0}

    for row in dataset:
        labels = list(row.get("labels", []))

        if labels == [NO_FINDING_LABEL] and counts["NORMAL"] < per_class:
            class_name = "NORMAL"
            label = "NORMAL"
        elif PNEUMONIA_LABEL in labels and counts["PNEUMONIA"] < per_class:
            class_name = "PNEUMONIA"
            label = "PNEUMONIA"
        else:
            continue

        patient_id = str(row["patient_id"])
        scan_id = str(row["scan_id"])
        filename = f"{patient_id}_{scan_id}_{label.lower()}.jpg"
        image_path = output / filename

        image = row["image"].convert("RGB")
        image.save(image_path, format="JPEG", quality=95)

        selected.append(
            {
                "image_path": filename,
                "patient_id": patient_id,
                "label": label,
            }
        )
        counts[class_name] += 1
        LOGGER.info(
            "Selected %s %d/%d | patient=%s scan=%s",
            class_name,
            counts[class_name],
            per_class,
            patient_id,
            scan_id,
        )

        if all(value >= per_class for value in counts.values()):
            break

    if any(value < per_class for value in counts.values()):
        raise RuntimeError(
            "Could not collect the requested number of samples. "
            f"Collected: {counts}"
        )

    with mapping_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "patient_id", "label"],
        )
        writer.writeheader()
        writer.writerows(selected)

    LOGGER.info("Wrote %d images to %s", len(selected), output)
    LOGGER.info("Wrote mapping to %s", mapping_file)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a patient-aware NIH ChestX-ray14 binary subset."
    )
    parser.add_argument("--output-dir", default="data/public_subset/images")
    parser.add_argument(
        "--mapping",
        default="data/public_subset/patient_mapping.csv",
    )
    parser.add_argument("--per-class", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    counts = prepare_subset(
        output_dir=args.output_dir,
        mapping_path=args.mapping,
        per_class=args.per_class,
    )
    print(f"Prepared subset: {counts}")


if __name__ == "__main__":
    main()
