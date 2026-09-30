"""Flask prediction API for the chest disease classifier."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.api.image_validation import validate_image_content
from src.api.predictor import ChestDiseasePredictor

LOGGER = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH")
predictor = ChestDiseasePredictor(model_path=MODEL_PATH)


def create_app(test_predictor: ChestDiseasePredictor | None = None) -> Flask:
    """Create the API application, allowing dependency injection for tests."""
    application = Flask(__name__)
    frontend_url = os.getenv("FRONTEND_URL", "*")
    CORS(application, resources={r"/*": {"origins": frontend_url}})
    active_predictor = test_predictor or predictor

    @application.get("/")
    def root():
        """Return a simple public landing page for the deployed API."""
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Chest Disease Classification API</title></head>"
            "<body style='font-family:Arial,sans-serif;max-width:760px;margin:60px auto;padding:0 24px'>"
            "<h1>Chest Disease Classification API</h1>"
            "<p>Educational NORMAL vs PNEUMONIA chest X-ray classification service.</p>"
            "<p><strong>Not a clinical diagnostic system.</strong></p>"
            "<h2>Endpoints</h2><ul>"
            "<li><a href='/health'>/health</a> — service/model health</li>"
            "<li><a href='/info'>/info</a> — model contract</li>"
            "<li><a href='/docs'>/docs</a> — API documentation</li>"
            "<li>POST /predict — upload a JPG/PNG image for prediction</li>"
            "</ul></body></html>"
        )

    @application.get("/health")
    def health():
        """Return API and model readiness information."""
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": active_predictor.model is not None,
                "model_type": "VGG16_binary_classifier",
                "expected_input": "224x224 PNG/JPG chest X-ray",
                "output_classes": list(active_predictor.class_names),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @application.get("/info")
    def info():
        """Return model contract information without exposing local paths."""
        return jsonify(
            {
                "model_type": "VGG16_binary_classifier",
                "input_shape": [224, 224, 3],
                "preprocessing": "VGG16 preprocess_input",
                "classes": list(active_predictor.class_names),
            }
        )

    @application.get("/docs")
    def docs():
        """Return a compact API contract."""
        return jsonify(
            {
                "title": "Chest Disease Classification API",
                "version": "1.0",
                "endpoints": {
                    "POST /predict": {
                        "input": "multipart/form-data with image file",
                        "output": "JSON with prediction and probabilities",
                    },
                    "GET /health": "System health and model readiness",
                    "GET /info": "Model input/output information",
                },
            }
        )

    @application.post("/predict")
    def predict():
        filepath = None
        request_id = str(uuid4())

        try:
            if "image" not in request.files:
                return jsonify({"error": "Invalid input", "request_id": request_id}), 400

            upload = request.files["image"]
            if not upload.filename:
                return jsonify({"error": "Invalid input", "request_id": request_id}), 400

            suffix = Path(upload.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                return jsonify({"error": "Invalid input", "request_id": request_id}), 400

            with NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                filepath = temporary.name
                upload.save(filepath)

            validate_image_content(filepath)
            result = active_predictor.predict_image(filepath)

            return jsonify(
                {
                    "prediction": result["predicted_class"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "request_id": request_id,
                }
            ), 200

        except ValueError as exc:
            LOGGER.warning("Validation error [%s]: %s", request_id, exc)
            return jsonify({"error": "Invalid input", "request_id": request_id}), 400
        except RuntimeError:
            LOGGER.warning("Model unavailable [%s]", request_id)
            return jsonify({"error": "Model unavailable", "request_id": request_id}), 503
        except Exception:
            LOGGER.exception("Prediction error [%s]", request_id)
            return jsonify({"error": "Prediction failed", "request_id": request_id}), 500
        finally:
            if filepath:
                try:
                    Path(filepath).unlink(missing_ok=True)
                except OSError:
                    LOGGER.exception("Temporary file cleanup failed [%s]", request_id)

    return application


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
