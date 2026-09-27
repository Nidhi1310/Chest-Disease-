"""Simple VGG16 transfer-learning baseline for chest disease classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Day 4 baseline model."""

    input_shape: tuple[int, int, int] = (224, 224, 3)
    classes: int = 2
    dense_units: int = 128
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3


class SimpleChestDiseaseModel:
    """Build the deliberately simple VGG16 classification baseline."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def build(self, weights: str | None = "imagenet") -> tf.keras.Model:
        """Build and compile the frozen-backbone transfer-learning model.

        ``weights='imagenet'`` is the intended training configuration.
        Tests can pass ``weights=None`` to avoid downloading external weights.
        """
        if self.config.classes != 2:
            raise ValueError("Day 4 baseline expects exactly 2 classes.")

        if not 0.0 <= self.config.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        base_model = VGG16(
            input_shape=self.config.input_shape,
            weights=weights,
            include_top=False,
        )

        base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential(
            [
                base_model,
                GlobalAveragePooling2D(name="global_average_pooling"),
                Dense(
                    self.config.dense_units,
                    activation="relu",
                    name="classification_dense",
                ),
                Dropout(self.config.dropout_rate, name="classification_dropout"),
                Dense(self.config.classes, activation="softmax", name="classifier"),
            ],
            name="simple_chest_disease_vgg16",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        LOGGER.info("Built Day 4 model with weights=%s", weights)
        return model


def print_model_insights(model: tf.keras.Model) -> dict[str, Any]:
    """Log and return critical architecture information for inspection."""
    total_params = model.count_params()
    trainable_params = sum(
        int(tf.keras.backend.count_params(weight))
        for weight in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    insights: dict[str, Any] = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }

    LOGGER.info(
        "Model Architecture Insights:\n"
        "Total Parameters: %s\n"
        "Trainable Parameters: %s\n"
        "Non-trainable Parameters: %s\n"
        "Input Shape: %s\n"
        "Output Shape: %s\n"
        "Training only the classification head while the VGG16 features remain frozen.",
        f"{total_params:,}",
        f"{trainable_params:,}",
        f"{non_trainable_params:,}",
        model.input_shape,
        model.output_shape,
    )

    return insights
