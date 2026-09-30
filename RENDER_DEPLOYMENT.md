# Render Deployment

## Live services

**Frontend:** https://chest-disease-ai.onrender.com
**API:** https://chest-disease-api.onrender.com
**API docs:** https://chest-disease-api.onrender.com/docs
**Health:** https://chest-disease-api.onrender.com/health

## Architecture

```text
React + TypeScript frontend
        |
        v
Render Static Site
        |
        | HTTPS + CORS
        v
Flask + Gunicorn API
        |
        v
VGG16 .keras model
```

## API service settings

- Region: Singapore
- Runtime: Python
- Build command: `pip install -r requirements-api.txt && mkdir -p model && curl -L --fail "$MODEL_URL" -o "$MODEL_PATH"`
- Start command: `GUNICORN_CMD_ARGS="--timeout 120" gunicorn --bind 0.0.0.0:$PORT --workers 1 app:app`
- Port: `10000`
- Health endpoint: `/health`

Environment variables:

- `PYTHON_VERSION=3.13.15`
- `PORT=10000`
- `FRONTEND_URL=https://chest-disease-ai.onrender.com`
- `MODEL_URL=https://github.com/Nidhi1310/Chest-Disease-/releases/download/chest-disease-model/chest_disease_vgg16.keras`
- `MODEL_PATH=model/chest_disease_vgg16.keras`
- `GUNICORN_CMD_ARGS=--timeout 120`

## Frontend service settings

- Runtime: Static Site
- Build command: `cd frontend && npm install && npm run build`
- Publish directory: `frontend/dist`
- Branch: `main`

The frontend defaults to the live API URL and can be overridden locally with `VITE_API_URL`.

## Model release

The trained model is published as the GitHub Release asset `chest_disease_vgg16.keras`.

The deployment downloads that versioned artifact at build time, so the model binary is not stored directly in the repository.

## Verification

The final portfolio smoke test verifies that the frontend is reachable, the API health endpoint responds, CORS is configured, the API contract is valid, a public chest X-ray can be submitted to `/predict`, and the live response contains a valid class, confidence, probability distribution, and request ID.

## Docker

The repository Dockerfile uses `requirements-api.txt`, downloads the same versioned model release asset, and starts Gunicorn on `0.0.0.0:$PORT` with a 120-second timeout.

## Note

TensorFlow/VGG16 inference can be slow on a small CPU instance. The API uses lazy model loading so `/health` can respond without waiting for TensorFlow model initialization.
