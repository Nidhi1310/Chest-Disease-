"""Download the versioned trained model for deployment.

MODEL_URL must point to a publicly accessible .keras model artifact.
MODEL_PATH controls the destination file.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import Request, urlopen


def download_model() -> Path:
    url = os.getenv("MODEL_URL")
    destination = Path(os.getenv("MODEL_PATH", "model/chest_disease_vgg16.keras"))

    if not url:
        raise RuntimeError("MODEL_URL is not configured.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    request = Request(url, headers={"User-Agent": "Chest-Disease-Deployment/1.0"})
    with urlopen(request, timeout=300) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)

    if destination.stat().st_size == 0:
        raise RuntimeError("Downloaded model file is empty.")

    print(f"Model downloaded to {destination}")
    return destination


if __name__ == "__main__":
    download_model()
