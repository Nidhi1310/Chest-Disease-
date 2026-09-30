"""Manifest-backed Keras sequence for patient-level chest X-ray training."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import tensorflow as tf

from src.preprocessing import VGG16Preprocessor


class ManifestImageSequence(tf.keras.utils.Sequence):
    """Load images from a validated split manifest without inventing patient IDs."""

    LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1, "0": 0, "1": 1}

    def __init__(
        self,
        rows: Iterable[Mapping[str, str]],
        *,
        dataset_root: str | Path | None = None,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 42,
        preprocessor: VGG16Preprocessor | None = None,
    ) -> None:
        super().__init__()
        self.rows = [dict(row) for row in rows]
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.preprocessor = preprocessor or VGG16Preprocessor()

        if not self.rows:
            raise ValueError("Manifest contains no rows.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.paths = [self._resolve_path(row["image_path"]) for row in self.rows]
        self.labels = np.asarray(
            [self._parse_label(row["label"]) for row in self.rows],
            dtype=np.int32,
        )
        self.indices = np.arange(len(self.rows))
        self.on_epoch_end()

    def __len__(self) -> int:
        return ceil(len(self.rows) / self.batch_size)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("batch_index out of range")

        start = batch_index * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]

        images = np.stack(
            [self.preprocessor.preprocess_image(self.paths[index]) for index in batch_indices],
            axis=0,
        )
        labels = self.labels[batch_indices]
        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _resolve_path(self, image_path: str) -> Path:
        path = Path(image_path)
        candidate = (
            path if path.is_absolute() or self.dataset_root is None
            else self.dataset_root / path
        )
        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {candidate}")
        return candidate

    @classmethod
    def _parse_label(cls, label: str) -> int:
        normalized = str(label).strip().upper()
        if normalized not in cls.LABEL_MAP:
            raise ValueError(
                f"Unsupported label {label!r}; expected NORMAL/PNEUMONIA or 0/1."
            )
        return cls.LABEL_MAP[normalized]
