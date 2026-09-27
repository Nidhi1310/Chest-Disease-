"""Day 3 patient-level data pipeline.

Turns the Day 1 split utilities into an explicit pipeline component:
load metadata -> split patients -> validate leakage/class balance -> write manifests
and an integrity report.

The real dataset is deliberately not bundled in this repository.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from src.split_strategy.patient_level_split import (
    DEFAULT_RATIOS,
    NORMAL_LABEL,
    PNEUMONIA_LABEL,
    build_integrity_report,
    load_mapping,
    split_patient_level,
    validate_split_integrity,
    write_split_csv,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitSummary:
    split: str
    patient_count: int
    image_count: int
    normal_count: int
    pneumonia_count: int

    @property
    def pneumonia_fraction(self) -> float:
        if self.image_count == 0:
            return 0.0
        return self.pneumonia_count / self.image_count


class PatientLevelSplitter:
    """Create leakage-safe train/validation/test manifests."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        split_ratios: tuple[float, float, float] = DEFAULT_RATIOS,
        seed: int = 42,
        minority_warning_threshold: float = 0.05,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.split_ratios = split_ratios
        self.seed = seed
        self.minority_warning_threshold = minority_warning_threshold

    def split(self, patient_mapping_file: str | Path) -> dict[str, list[dict[str, str]]]:
        """Load metadata, create patient-level splits, and validate them."""
        rows = load_mapping(patient_mapping_file)

        if self.dataset_root is not None:
            self._validate_dataset_root(rows)

        splits = split_patient_level(
            rows=rows,
            ratios=self.split_ratios,
            seed=self.seed,
        )

        self.validate_split(splits, expected_rows=rows)
        self._log_class_balance(splits)
        LOGGER.info("Patient-level split completed successfully.")
        return splits

    def validate_split(
        self,
        splits: dict[str, list[dict[str, str]]],
        expected_rows: list[dict[str, str]] | None = None,
    ) -> None:
        """Validate leakage and ensure every input record survives the split."""
        expected_patient_ids = None
        expected_paths = None

        if expected_rows is not None:
            expected_patient_ids = {row["patient_id"] for row in expected_rows}
            expected_paths = {row["image_path"] for row in expected_rows}

        validate_split_integrity(
            splits=splits,
            expected_patient_ids=expected_patient_ids,
        )

        if expected_paths is not None:
            actual_paths = {
                row["image_path"]
                for split_rows in splits.values()
                for row in split_rows
            }

            missing = expected_paths - actual_paths
            unexpected = actual_paths - expected_paths

            if missing:
                raise ValueError(
                    "Images disappeared during splitting: "
                    f"{sorted(missing)[:5]}"
                )

            if unexpected:
                raise ValueError(
                    "Unexpected images appeared during splitting: "
                    f"{sorted(unexpected)[:5]}"
                )

    def summarize(
        self,
        splits: dict[str, list[dict[str, str]]],
    ) -> list[SplitSummary]:
        """Return per-split patient/image/class counts."""
        summaries = []

        for split in ("train", "validation", "test"):
            rows = splits[split]
            summaries.append(
                SplitSummary(
                    split=split,
                    patient_count=len({row["patient_id"] for row in rows}),
                    image_count=len(rows),
                    normal_count=sum(
                        row["label"] == NORMAL_LABEL for row in rows
                    ),
                    pneumonia_count=sum(
                        row["label"] == PNEUMONIA_LABEL for row in rows
                    ),
                )
            )

        return summaries

    def write_outputs(
        self,
        splits: dict[str, list[dict[str, str]]],
        output_dir: str | Path = "data/splits",
        report_path: str | Path = "artifacts/day3/data_integrity_report.txt",
    ) -> Path:
        """Write split manifests and the integrity report."""
        self.validate_split(splits)

        output = Path(output_dir)

        for split_name, rows in splits.items():
            write_split_csv(output / f"{split_name}.csv", rows)

        report = build_integrity_report(
            splits=splits,
            seed=self.seed,
            ratios=self.split_ratios,
        )

        warnings = self._class_balance_warnings(splits)

        if warnings:
            report += "\n\nCLASS BALANCE WARNINGS\n"
            report += "\n".join(f"- {warning}" for warning in warnings)

        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report + "\n", encoding="utf-8")

        return report_file

    def run(
        self,
        patient_mapping_file: str | Path,
        output_dir: str | Path = "data/splits",
        report_path: str | Path = "artifacts/day3/data_integrity_report.txt",
    ) -> Path:
        """Run the complete Day 3 pipeline."""
        splits = self.split(patient_mapping_file)

        report_file = self.write_outputs(
            splits=splits,
            output_dir=output_dir,
            report_path=report_path,
        )

        LOGGER.info("Integrity report written to %s", report_file)
        return report_file

    def _validate_dataset_root(self, rows: list[dict[str, str]]) -> None:
        if self.dataset_root is None:
            return

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.dataset_root}"
            )

        missing_paths = []

        for row in rows:
            image = Path(row["image_path"])
            candidate = image if image.is_absolute() else self.dataset_root / image

            if not candidate.exists():
                missing_paths.append(str(candidate))

                if len(missing_paths) >= 10:
                    break

        if missing_paths:
            raise FileNotFoundError(
                "Dataset contains image paths that do not exist. "
                f"Examples: {missing_paths}"
            )

    def _class_balance_warnings(
        self,
        splits: dict[str, list[dict[str, str]]],
    ) -> list[str]:
        warnings = []

        for summary in self.summarize(splits):
            if summary.image_count == 0:
                warnings.append(f"{summary.split} split is empty.")
                continue

            minority_count = min(
                summary.normal_count,
                summary.pneumonia_count,
            )
            minority_fraction = minority_count / summary.image_count

            if minority_fraction < self.minority_warning_threshold:
                warnings.append(
                    f"{summary.split} minority class is only "
                    f"{minority_fraction:.1%} of images."
                )

        return warnings

    def _log_class_balance(
        self,
        splits: dict[str, list[dict[str, str]]],
    ) -> None:
        for summary in self.summarize(splits):
            LOGGER.info(
                "%s | patients=%d images=%d NORMAL=%d PNEUMONIA=%d",
                summary.split,
                summary.patient_count,
                summary.image_count,
                summary.normal_count,
                summary.pneumonia_count,
            )

        for warning in self._class_balance_warnings(splits):
            LOGGER.warning(warning)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create validated patient-level dataset manifests."
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help="CSV with image_path, patient_id, label",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional root used to validate that image paths exist",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
    )
    parser.add_argument(
        "--report",
        default="artifacts/day3/data_integrity_report.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()

    splitter = PatientLevelSplitter(
        dataset_root=args.dataset_root,
        seed=args.seed,
    )

    report = splitter.run(
        patient_mapping_file=args.mapping,
        output_dir=args.output_dir,
        report_path=args.report,
    )

    print(f"Day 3 split pipeline completed. Report: {report}")


if __name__ == "__main__":
    main()
