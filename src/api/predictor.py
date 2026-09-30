"""Model-loading and prediction wrapper used by the Flask API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.preprocessing import VGG16Preprocessor


class ChestDiseasePredictor:
    """Load a trained Keras model and expose safe image prediction."""

    class_names = ("NORMAL", "PNEUMONIA")

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any | None = None,
        preprocessor: VGG16Preprocessor | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model = model
        self.preprocessor = preprocessor or VGG16Preprocessor()

        if self.model is None and self.model_path:
            self.model = tf.keras.models.load_model(self.model_path)

    def predict_image(self, filepath: str | Path) -> dict[str, Any]:
        """Predict NORMAL/PNEUMONIA from one validated image file."""
        if self.model is None:
            raise RuntimeError("Prediction model is not loaded")

        image = self.preprocessor.preprocess_image(filepath)
        batch = np.expand_dims(image, axis=0)
        probabilities = np.asarray(self.model.predict(batch, verbose=0), dtype=float)

        if probabilities.shape != (1, 2):
            raise ValueError("Model returned an unexpected prediction shape")

        class_index = int(np.argmax(probabilities[0]))
        class_probabilities = probabilities[0]

        return {
            "predicted_class": self.class_names[class_index],
            "confidence": float(class_probabilities[class_index]),
            "probabilities": {
                name: float(class_probabilities[index])
                for index, name in enumerate(self.class_names)
            },
        }
