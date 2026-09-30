# Day 7 - Prediction API and Production Considerations

## Implemented

- Flask prediction API in `app.py`
- `/health` readiness endpoint
- `/info` model contract endpoint
- `/docs` API contract endpoint
- `/predict` multipart image endpoint
- Content-based image validation with Pillow
- Dimension and mode limits
- Temporary-upload cleanup in `finally`
- Sanitized client-facing errors
- Request IDs for tracing
- Dependency injection for API tests
- Gunicorn production entry point

## API contract

### `GET /health`
Returns API status, model-loaded state, input contract, output classes, and UTC timestamp.

### `GET /info`
Returns model type, input shape, preprocessing strategy, and class names. Local filesystem model paths are never exposed.

### `GET /docs`
Returns the supported endpoints and their input/output contract.

### `POST /predict`
Accepts an image upload using the multipart field `image`. Supported filename extensions are PNG/JPG/JPEG, followed by actual image-content validation.

Client-facing failures are sanitized to `Invalid input`, `Model unavailable`, or `Prediction failed`; implementation details are logged server-side instead.

## Production server

Use Gunicorn rather than Flask's development server:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## Model loading

Set `MODEL_PATH` to a trained Keras model before starting the service. Without a model, `/health` reports `model_loaded: false` and `/predict` returns HTTP 503.

## Important boundary

The API is a deployment interface, not evidence of clinical readiness. Medical performance remains dependent on the leakage-safe split, preprocessing consistency, actual training results, and Day 6 evaluation.
