"""Train, track, save, and evaluate the chest disease baseline.

Expected split manifests contain:
    image_path,patient_id,label

The training entry point re-validates patient integrity across the three
manifests before model.fit() so a corrupted or mismatched manifest cannot
silently introduce cross-split leakage.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.medical_evaluator import MedicalEvaluator
from src.models.chest_disease_model import ModelConfig, SimpleChestDiseaseModel
from src.training.experiment_tracker import ExperimentTracker
from src.training.image_sequence import ManifestImageSequence
from src.training.reproducibility import set_global_seed

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"image_path", "patient_id", "label"}
EXPECTED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    seed: int = 42


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load and validate one Day 3 split manifest."""
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"Manifest {manifest} is missing required columns: {sorted(missing)}"
            )
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")

    return rows


def validate_manifest_integrity(
    manifests: Iterable[tuple[str, list[dict[str, str]]]],
) -> None:
    """Reject patient or image overlap across train/validation/test manifests."""
    seen_patients: dict[str, str] = {}
    seen_images: dict[str, str] = {}

    for split_name, rows in manifests:
        for row in rows:
            patient_id = str(row["patient_id"]).strip()
            image_path = str(row["image_path"]).strip()
            if not patient_id:
                raise ValueError(f"{split_name} contains an empty patient_id.")
            if not image_path:
                raise ValueError(f"{split_name} contains an empty image_path.")

            prior_patient_split = seen_patients.get(patient_id)
            if prior_patient_split is not None and prior_patient_split != split_name:
                raise ValueError(
                    f"Patient leakage detected: {patient_id!r} appears in "
                    f"{prior_patient_split} and {split_name}."
                )
            seen_patients[patient_id] = split_name

            prior_image_split = seen_images.get(image_path)
            if prior_image_split is not None and prior_image_split != split_name:
                raise ValueError(
                    f"Image leakage detected: {image_path!r} appears in "
                    f"{prior_image_split} and {split_name}."
                )
            seen_images[image_path] = split_name


def train_and_evaluate(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    test_manifest: str | Path,
    dataset_root: str | Path | None,
    output_model: str | Path,
    report_path: str | Path,
    metadata_dir: str | Path,
    mlflow_experiment: str,
    tracking_uri: str | None,
    config: TrainingConfig,
    image_net_weights: bool = True,
) -> dict[str, Any]:
    """Run the complete real-dataset training path."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")

    set_global_seed(config.seed, deterministic=True)

    train_rows = load_manifest(train_manifest)
    validation_rows = load_manifest(validation_manifest)
    test_rows = load_manifest(test_manifest)
    manifests = list(zip(EXPECTED_SPLITS, (train_rows, validation_rows, test_rows)))
    validate_manifest_integrity(manifests)

    train_data = ManifestImageSequence(
        train_rows, dataset_root=dataset_root, batch_size=config.batch_size,
        shuffle=True, seed=config.seed,
    )
    validation_data = ManifestImageSequence(
        validation_rows, dataset_root=dataset_root, batch_size=config.batch_size,
        shuffle=False, seed=config.seed,
    )
    test_data = ManifestImageSequence(
        test_rows, dataset_root=dataset_root, batch_size=config.batch_size,
        shuffle=False, seed=config.seed,
    )

    model_config = ModelConfig()
    weights = "imagenet" if image_net_weights else None
    model = SimpleChestDiseaseModel(model_config).build(weights=weights)

    LOGGER.info(
        "Training baseline | train=%d validation=%d test=%d",
        len(train_rows), len(validation_rows), len(test_rows),
    )

    history = model.fit(
        train_data, validation_data=validation_data, epochs=config.epochs, verbose=1
    )

    tracker = ExperimentTracker(
        experiment_name=mlflow_experiment, tracking_uri=tracking_uri
    )
    run_id = tracker.log_training(
        model=model,
        config={**asdict(config), **asdict(model_config)},
        history=history,
        seed=config.seed,
        dataset_split="patient_level_0.70_0.15_0.15",
        preprocessing="vgg16_specific",
        run_name="vgg16_baseline",
        metadata_dir=metadata_dir,
    )

    output_model_path = Path(output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)

    evaluator = MedicalEvaluator(
        model=model,
        test_images=test_data,
        test_labels=test_data.labels,
        prediction_batch_size=config.batch_size,
    )
    report_file = evaluator.generate_report(report_path)

    summary = {
        "mlflow_run_id": run_id,
        "model_path": str(output_model_path),
        "evaluation_report": str(report_file),
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "test_images": len(test_rows),
    }
    LOGGER.info("Training completed: %s", summary)
    return summary


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the VGG16 chest disease baseline."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-model", default="artifacts/model/chest_disease_vgg16.keras")
    parser.add_argument("--report", default="artifacts/day6/medical_evaluation.txt")
    parser.add_argument("--metadata-dir", default="artifacts/training")
    parser.add_argument("--mlflow-experiment", default="chest_disease_clf")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-imagenet-weights", action="store_true")
    args = parser.parse_args()

    configure_logging()
    summary = train_and_evaluate(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        test_manifest=args.test_manifest,
        dataset_root=args.dataset_root,
        output_model=args.output_model,
        report_path=args.report,
        metadata_dir=args.metadata_dir,
        mlflow_experiment=args.mlflow_experiment,
        tracking_uri=args.tracking_uri,
        config=TrainingConfig(
            batch_size=args.batch_size, epochs=args.epochs, seed=args.seed
        ),
        image_net_weights=not args.no_imagenet_weights,
    )

    print("Training pipeline completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
