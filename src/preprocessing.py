"""Backbone-specific preprocessing for the VGG16 baseline.

Contract:
    - Input image: RGB pixels in [0, 255]
    - Resize: 224 x 224
    - Channels: 3
    - Preprocessing: tf.keras.applications.vgg16.preprocess_input

The same implementation is intended for training and inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingStats:
    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    std: float


class VGG16Preprocessor:
    """Reusable VGG16 preprocessing pipeline."""

    TARGET_SIZE = (224, 224)
    CHANNELS = 3
    EXPECTED_SHAPE = (224, 224, 3)

    def preprocess_array(self, image_array: np.ndarray) -> np.ndarray:
        """Resize is not performed here; the array must already be 224x224x3."""
        array = np.asarray(image_array, dtype=np.float32)

        if array.shape != self.EXPECTED_SHAPE:
            raise ValueError(
                f"Expected image shape {self.EXPECTED_SHAPE}, "
                f"received {array.shape}."
            )

        if not np.isfinite(array).all():
            raise ValueError("Image contains NaN or infinite values.")

        if array.min() < 0 or array.max() > 255:
            raise ValueError(
                "Raw image values must be in the [0, 255] range."
            )

        # Copy first because preprocess_input may modify compatible NumPy arrays
        # in place according to the TensorFlow API.
        preprocessed = preprocess_input(np.array(array, copy=True))

        self.validate_preprocessed(preprocessed)
        return preprocessed.astype(np.float32, copy=False)

    def preprocess_image(self, image_path: str | Path) -> np.ndarray:
        """Load, resize and apply the official VGG16 preprocessing."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = load_img(
            path,
            color_mode="rgb",
            target_size=self.TARGET_SIZE,
        )
        array = img_to_array(image)

        return self.preprocess_array(array)

    @classmethod
    def validate_preprocessed(cls, image_array: np.ndarray) -> None:
        """Validate shape and numerical sanity after VGG16 preprocessing."""
        array = np.asarray(image_array)

        if array.shape != cls.EXPECTED_SHAPE:
            raise ValueError(
                f"Preprocessed image must have shape {cls.EXPECTED_SHAPE}; "
                f"received {array.shape}."
            )

        if not np.isfinite(array).all():
            raise ValueError("Preprocessed image contains non-finite values.")

        # VGG16 preprocessing converts RGB -> BGR and subtracts ImageNet
        # channel means. For a valid [0,255] input this produces values near
        # [-124, 151], depending on pixel/channel values.
        if array.min() < -130.0 or array.max() > 160.0:
            raise ValueError(
                "Preprocessed values are outside the expected VGG16 range."
            )

    @staticmethod
    def stats(image_array: np.ndarray) -> PreprocessingStats:
        """Return compact preprocessing statistics for debugging/logging."""
        array = np.asarray(image_array)
        stats = PreprocessingStats(
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            minimum=float(array.min()),
            maximum=float(array.max()),
            mean=float(array.mean()),
            std=float(array.std()),
        )

        LOGGER.info(
            "Preprocessing stats | shape=%s dtype=%s min=%.2f max=%.2f "
            "mean=%.2f std=%.2f",
            stats.shape,
            stats.dtype,
            stats.minimum,
            stats.maximum,
            stats.mean,
            stats.std,
        )

        return stats
