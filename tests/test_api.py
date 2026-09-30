from io import BytesIO

from PIL import Image

from app import create_app


class FakePredictor:
    class_names = ("NORMAL", "PNEUMONIA")
    model = object()

    def predict_image(self, filepath):
        return {
            "predicted_class": "PNEUMONIA",
            "confidence": 0.9,
            "probabilities": {"NORMAL": 0.1, "PNEUMONIA": 0.9},
        }


def make_png():
    buffer = BytesIO()
    Image.new("L", (100, 100), color=128).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_health_endpoint_reports_model_state():
    client = create_app(FakePredictor()).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["output_classes"] == ["NORMAL", "PNEUMONIA"]


def test_info_and_docs_endpoints():
    client = create_app(FakePredictor()).test_client()

    info = client.get("/info")
    docs = client.get("/docs")

    assert info.status_code == 200
    assert docs.status_code == 200
    assert info.get_json()["preprocessing"] == "VGG16 preprocess_input"
    assert "POST /predict" in docs.get_json()["endpoints"]


def test_predict_accepts_valid_png():
    client = create_app(FakePredictor()).test_client()

    response = client.post(
        "/predict",
        data={"image": (make_png(), "xray.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "PNEUMONIA"
    assert payload["confidence"] == 0.9
    assert "request_id" in payload


def test_predict_rejects_invalid_extension_without_leaking_details():
    client = create_app(FakePredictor()).test_client()

    response = client.post(
        "/predict",
        data={"image": (BytesIO(b"not an image"), "payload.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid input"
    assert "Traceback" not in response.get_data(as_text=True)


def test_predict_returns_503_when_model_is_unavailable():
    class UnavailablePredictor(FakePredictor):
        model = None

        def predict_image(self, filepath):
            raise RuntimeError("Prediction model is not loaded")

    client = create_app(UnavailablePredictor()).test_client()

    response = client.post(
        "/predict",
        data={"image": (make_png(), "xray.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    assert response.get_json()["error"] == "Model unavailable"
