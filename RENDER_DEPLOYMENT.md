# Render Deployment

## Service

Render service: `chest-disease-classifier`

Public URL after a successful deployment:
`https://chest-disease-classifier-aoc7.onrender.com`

## Runtime configuration

- Python version: `3.13.15`
- Build: `pip install -r requirements.txt` and download the versioned model release asset
- Start: `gunicorn --bind 0.0.0.0:$PORT --workers 1 app:app`
- Model path: `model/chest_disease_vgg16.keras`
- Health endpoint: `/health`

## Model release

The real-training workflow publishes the trained model as the `chest-disease-model` GitHub Release. The Render build downloads:

`https://github.com/Nidhi1310/Chest-Disease-/releases/download/chest-disease-model/chest_disease_vgg16.keras`

Do not commit the `.keras` model or the dataset to the repository.

## Deployment sequence

1. Run **Actions → Real Dataset Training → Run workflow**.
2. Wait for every step to pass, including **Publish model release** and the API smoke test.
3. In Render, open `chest-disease-classifier` and deploy manually.
4. Verify `/health` returns `model_loaded: true`.
5. Verify `/info` and `/docs`.
6. Send a JPG/PNG image to `POST /predict`.

## Note on compute

TensorFlow/VGG16 may exceed very small memory limits. Start with the configured Render service for a deployment test; increase memory if the service is killed during model loading.
