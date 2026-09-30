import json

import mlflow
import numpy as np

from src.training.experiment_tracker import ExperimentTracker, compare_experiments
from src.training.reproducibility import build_training_metadata, save_training_metadata


class FakeHistory:
    history = {
        "loss": [0.8, 0.5],
        "accuracy": [0.70, 0.82],
        "val_loss": [0.7, 0.45],
        "val_accuracy": [0.73, 0.84],
    }


def test_config_to_params_supports_dataclass_like_objects():
    class Config:
        learning_rate = 0.001
        batch_size = 32
        epochs = 2

    params = ExperimentTracker._config_to_params(Config())

    assert params["learning_rate"] == 0.001
    assert params["batch_size"] == 32
    assert params["epochs"] == 2
    assert params["model"] == "VGG16"
    assert params["dropout"] == 0.3


def test_history_logging_uses_one_step_per_epoch(monkeypatch):
    logged = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics, step=None: logged.append((step, metrics)))

    ExperimentTracker._log_history(FakeHistory.history)

    assert [step for step, _ in logged] == [0, 1]
    assert logged[1][1]["val_accuracy"] == 0.84


def test_build_training_metadata_contains_reproducibility_fields():
    metadata = build_training_metadata(
        run_id="run-123",
        seed=42,
        extra={"learning_rate": 0.001},
    )

    assert metadata["mlflow_run_id"] == "run-123"
    assert metadata["random_seed"] == 42
    assert metadata["tensorflow_seed"] == 42
    assert metadata["numpy_seed"] == 42
    assert metadata["dataset_split"] == "patient_level_0.70_0.15_0.15"
    assert metadata["preprocessing"] == "vgg16_specific"
    assert metadata["environment"]["tensorflow"]


def test_save_training_metadata_writes_valid_json(tmp_path):
    output = save_training_metadata(
        tmp_path / "metadata.json",
        run_id="run-abc",
        seed=7,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.exists()
    assert payload["mlflow_run_id"] == "run-abc"
    assert payload["random_seed"] == 7


def test_real_mlflow_run_can_log_params_and_metrics(tmp_path, monkeypatch):
    tracking_uri = tmp_path.as_uri()
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        tracker = ExperimentTracker(
            experiment_name="day5-test",
            tracking_uri=tracking_uri,
        )
        with tracker.start_run(run_name="smoke") as run:
            mlflow.log_params({"model": "VGG16", "learning_rate": 0.001})
            mlflow.log_metrics({"val_accuracy": 0.84}, step=1)

            run_id = run.info.run_id

        runs = compare_experiments("day5-test")
        assert run_id in set(runs["run_id"])
        assert "params.model" in runs.columns
        assert "metrics.val_accuracy" in runs.columns
    finally:
        mlflow.set_tracking_uri(previous_uri)
