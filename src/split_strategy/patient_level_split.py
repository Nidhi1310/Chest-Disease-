"""Leakage-safe patient-level dataset splitting.

Expected CSV columns:
    image_path, patient_id, label

Labels may be NORMAL/PNEUMONIA or 0/1.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
REQUIRED_COLUMNS = {"image_path", "patient_id", "label"}
NORMAL_LABEL = "NORMAL"
PNEUMONIA_LABEL = "PNEUMONIA"


@dataclass(frozen=True)
class PatientGroup:
    patient_id: str
    rows: tuple[dict[str, str], ...]
    image_count: int
    pneumonia_count: int


def normalize_label(value: str) -> str:
    raw = value.strip().upper()
    aliases = {
        "0": NORMAL_LABEL,
        "NORMAL": NORMAL_LABEL,
        "1": PNEUMONIA_LABEL,
        "PNEUMONIA": PNEUMONIA_LABEL,
    }
    if raw not in aliases:
        raise ValueError(
            f"Unsupported label {value!r}. Use NORMAL/PNEUMONIA or 0/1."
        )
    return aliases[raw]


def load_mapping(mapping_path: str | Path) -> list[dict[str, str]]:
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Mapping is missing required columns: {sorted(missing)}"
            )

        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            label = (row.get("label") or "").strip()

            if not image_path or not patient_id:
                raise ValueError(
                    f"Row {index}: image_path and patient_id are required."
                )

            normalized = dict(row)
            normalized["image_path"] = image_path
            normalized["patient_id"] = patient_id
            normalized["label"] = normalize_label(label)
            rows.append(normalized)

    if not rows:
        raise ValueError("Mapping file contains no image records.")

    return rows


def build_patient_groups(rows: Iterable[dict[str, str]]) -> list[PatientGroup]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["patient_id"], []).append(row)

    groups = [
        PatientGroup(
            patient_id=patient_id,
            rows=tuple(patient_rows),
            image_count=len(patient_rows),
            pneumonia_count=sum(
                row["label"] == PNEUMONIA_LABEL for row in patient_rows
            ),
        )
        for patient_id, patient_rows in grouped.items()
    ]

    if len(groups) < 3:
        raise ValueError(
            "At least 3 unique patients are required for train/validation/test."
        )

    return groups


def split_objective(
    counts: dict[str, tuple[int, int, int]],
    targets: dict[str, tuple[float, float]],
) -> float:
    score = 0.0
    for split in SPLITS:
        images, pneumonia, _ = counts[split]
        target_images, target_pneumonia = targets[split]

        image_scale = max(target_images, 1.0)
        pneumonia_scale = max(target_pneumonia, 1.0)

        score += ((images - target_images) / image_scale) ** 2
        score += ((pneumonia - target_pneumonia) / pneumonia_scale) ** 2

    return score


def validate_split_integrity(
    splits: dict[str, list[dict[str, str]]],
    expected_patient_ids: set[str] | None = None,
) -> None:
    missing_splits = set(SPLITS) - set(splits)
    if missing_splits:
        raise ValueError(f"Missing split(s): {sorted(missing_splits)}")

    patient_sets = {
        split: {row["patient_id"] for row in splits[split]}
        for split in SPLITS
    }

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"Patient leakage detected between {left} and {right}: "
                    f"{examples}"
                )

    combined_patients = set().union(*(patient_sets[split] for split in SPLITS))

    if expected_patient_ids is not None:
        missing_patients = expected_patient_ids - combined_patients
        unexpected_patients = combined_patients - expected_patient_ids

        if missing_patients:
            raise ValueError(
                f"Patients disappeared during splitting: "
                f"{sorted(missing_patients)[:5]}"
            )
        if unexpected_patients:
            raise ValueError(
                f"Unexpected patients appeared during splitting: "
                f"{sorted(unexpected_patients)[:5]}"
            )

    total_rows = sum(len(splits[split]) for split in SPLITS)
    if total_rows == 0:
        raise ValueError("All dataset splits are empty.")


def split_patient_level(
    rows: list[dict[str, str]],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must contain three values summing to 1.0.")
    if any(r <= 0 for r in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    patients = build_patient_groups(rows)
    expected_patient_ids = {patient.patient_id for patient in patients}

    total_images = sum(patient.image_count for patient in patients)
    total_pneumonia = sum(patient.pneumonia_count for patient in patients)

    targets = {
        split: (total_images * ratio, total_pneumonia * ratio)
        for split, ratio in zip(SPLITS, ratios)
    }

    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda patient: patient.image_count, reverse=True)

    assignments: dict[str, list[dict[str, str]]] = {
        split: [] for split in SPLITS
    }
    counts: dict[str, tuple[int, int, int]] = {
        split: (0, 0, 0) for split in SPLITS
    }

    for patient in shuffled:
        best_split: str | None = None
        best_score: float | None = None

        for split in SPLITS:
            candidate_counts = dict(counts)
            images, pneumonia, patient_count = candidate_counts[split]
            candidate_counts[split] = (
                images + patient.image_count,
                pneumonia + patient.pneumonia_count,
                patient_count + 1,
            )

            score = split_objective(candidate_counts, targets)

            if patient_count == 0:
                score *= 0.85

            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assert best_split is not None

        assignments[best_split].extend(patient.rows)

        images, pneumonia, patient_count = counts[best_split]
        counts[best_split] = (
            images + patient.image_count,
            pneumonia + patient.pneumonia_count,
            patient_count + 1,
        )

    validate_split_integrity(assignments, expected_patient_ids)
    return assignments


def build_integrity_report(
    splits: dict[str, list[dict[str, str]]],
    seed: int,
    ratios: tuple[float, float, float],
) -> str:
    validate_split_integrity(splits)

    all_rows = [row for split in SPLITS for row in splits[split]]
    total_patients = len({row["patient_id"] for row in all_rows})

    lines = [
        "CHEST DISEASE DATA INTEGRITY REPORT",
        "=" * 38,
        f"Random seed: {seed}",
        (
            "Target split: "
            f"train={ratios[0]:.0%}, "
            f"validation={ratios[1]:.0%}, "
            f"test={ratios[2]:.0%}"
        ),
        f"Total patients: {total_patients}",
        f"Total images: {len(all_rows)}",
        "",
    ]

    for split in SPLITS:
        split_rows = splits[split]
        patients = {row["patient_id"] for row in split_rows}
        normal = sum(row["label"] == NORMAL_LABEL for row in split_rows)
        pneumonia = sum(
            row["label"] == PNEUMONIA_LABEL for row in split_rows
        )

        if split_rows:
            normal_line = (
                f"  NORMAL: {normal} "
                f"({normal / len(split_rows):.1%})"
            )
            pneumonia_line = (
                f"  PNEUMONIA: {pneumonia} "
                f"({pneumonia / len(split_rows):.1%})"
            )
        else:
            normal_line = "  NORMAL: 0 (0.0%)"
            pneumonia_line = "  PNEUMONIA: 0 (0.0%)"

        lines.extend(
            [
                split.upper(),
                f"  Patients: {len(patients)}",
                f"  Images: {len(split_rows)}",
                normal_line,
                pneumonia_line,
                "",
            ]
        )

    lines.append("Leakage check: PASSED")
    return "\n".join(lines)


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "patient_id", "label"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def run(
    mapping_path: str | Path,
    output_dir: str | Path = "data/splits",
    report_path: str | Path = "artifacts/day1/data_integrity_report.txt",
    seed: int = 42,
) -> None:
    rows = load_mapping(mapping_path)
    splits = split_patient_level(rows, seed=seed)

    output = Path(output_dir)
    for split in SPLITS:
        write_split_csv(output / f"{split}.csv", splits[split])

    report = build_integrity_report(
        splits=splits,
        seed=seed,
        ratios=DEFAULT_RATIOS,
    )

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe patient-level dataset splits."
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help="CSV containing image_path, patient_id, label",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory for split CSVs",
    )
    parser.add_argument(
        "--report",
        default="artifacts/day1/data_integrity_report.txt",
        help="Path for the integrity report",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run(
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        report_path=args.report,
        seed=args.seed,
    )

    print(f"Patient-level split completed. Report: {args.report}")


if __name__ == "__main__":
    main()
