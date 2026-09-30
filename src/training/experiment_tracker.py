"""Real MLflow experiment tracking for chest-disease training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow

from .reproducibility import save_training_metadata

LOGGER = logging.getLogger(__name__)


class ExperimentTracker:
    """Track parameters, metrics, model artifacts, and reproducibility metadata."""

    def __init__(
        self,
        experiment_name: str = "chest_disease_clf",
        tracking_uri: str | None = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None):
        """Start an MLflow run in the configured experiment."""
        return mlflow.start_run(run_name=run_name)

    def log_training(
        self,
        *,
        model: Any,
        config: Any,
        history: Any,
        seed: int,
        dataset_split: str = "patient_level_0.70_0.15_0.15",
        preprocessing: str = "vgg16_specific",
        run_name: str | None = None,
        metadata_dir: str | Path = "artifacts/training",
    ) -> str:
        """Log one completed training run and return its MLflow run ID.

        The method is intentionally explicit rather than relying only on
        autologging: parameters and per-epoch metrics are visible in the code.
        """
        params = self._config_to_params(config)

        with self.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.set_tags(
                {
                    "project": "chest-disease-classification",
                    "dataset_split": dataset_split,
                    "preprocessing": preprocessing,
                    "tracking": "explicit",
                }
            )

            history_dict = getattr(history, "history", history)
            self._log_history(history_dict)

            metadata_path = save_training_metadata(
                Path(metadata_dir) / "training_metadata.json",
                run_id=run.info.run_id,
                seed=seed,
                dataset_split=dataset_split,
                preprocessing=preprocessing,
                extra={"parameters": params},
            )
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

            # MLflow's current TensorFlow flavor supports Keras model logging.
            # Use `name=` rather than the deprecated `artifact_path=` argument.
            mlflow.tensorflow.log_model(model, name="model")

            LOGGER.info("Logged MLflow run %s", run.info.run_id)
            return run.info.run_id

    @staticmethod
    def _config_to_params(config: Any) -> dict[str, Any]:
        """Convert a config object/dict into MLflow-compatible parameters."""
        if config is None:
            return {}

        if isinstance(config, dict):
            params = dict(config)
        elif hasattr(config, "__dict__"):
            params = dict(vars(config))
        else:
            params = {
                name: getattr(config, name)
                for name in dir(config)
                if not name.startswith("_") and not callable(getattr(config, name))
            }

        params.setdefault("model", "VGG16")
        params.setdefault("optimizer", "Adam")
        params.setdefault("dropout", 0.3)
        return params

    @staticmethod
    def _log_history(history: dict[str, list[Any]]) -> None:
        """Log scalar Keras history values once per epoch."""
        if not history:
            return

        epoch_count = max(len(values) for values in history.values())
        for epoch in range(epoch_count):
            metrics: dict[str, float] = {}
            for name, values in history.items():
                if epoch >= len(values):
                    continue
                value = values[epoch]
                try:
                    metrics[name] = float(value)
                except (TypeError, ValueError):
                    continue

            if metrics:
                mlflow.log_metrics(metrics, step=epoch)

    @staticmethod
    def latest_run(experiment_name: str = "chest_disease_clf"):
        """Return runs for an experiment ordered by start time descending."""
        return mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["start_time DESC"],
        )


def compare_experiments(
    experiment_name: str = "chest_disease_clf",
    columns: list[str] | None = None,
):
    """Return a compact DataFrame for comparing tracked experiments."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
    )

    default_columns = [
        "run_id",
        "params.model",
        "params.learning_rate",
        "params.batch_size",
        "params.epochs",
        "params.dropout",
        "metrics.val_accuracy",
        "metrics.val_loss",
    ]
    selected = columns or default_columns
    available = [column for column in selected if column in runs.columns]
    return runs[available].copy()
