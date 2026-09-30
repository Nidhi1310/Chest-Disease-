"""Reproducibility helpers for model training."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # Some TensorFlow/runtime combinations do not expose or permit this
            # after initialization. The core RNG seeds are still applied.
            pass


def build_training_metadata(
    *,
    run_id: str | None,
    seed: int,
    dataset_split: str = "patient_level_0.70_0.15_0.15",
    preprocessing: str = "vgg16_specific",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata needed to understand and reproduce a training run."""
    metadata: dict[str, Any] = {
        "mlflow_run_id": run_id,
        "random_seed": seed,
        "python_hash_seed": str(seed),
        "tensorflow_seed": seed,
        "numpy_seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_split": dataset_split,
        "preprocessing": preprocessing,
        "environment": {
            "python": sys.version,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
        },
    }

    if extra:
        metadata["extra"] = extra

    return metadata


def save_training_metadata(
    path: str | Path,
    *,
    run_id: str | None,
    seed: int,
    dataset_split: str = "patient_level_0.70_0.15_0.15",
    preprocessing: str = "vgg16_specific",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write reproducibility metadata as a JSON file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_training_metadata(
        run_id=run_id,
        seed=seed,
        dataset_split=dataset_split,
        preprocessing=preprocessing,
        extra=extra,
    )

    output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
