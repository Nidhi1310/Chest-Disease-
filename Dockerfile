FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV MODEL_URL=https://github.com/Nidhi1310/Chest-Disease-/releases/download/chest-disease-model/chest_disease_vgg16.keras
ENV MODEL_PATH=/app/model/chest_disease_vgg16.keras
ENV PORT=5000

RUN mkdir -p /app/model

EXPOSE 5000

CMD ["sh", "-c", "python -m scripts.download_model && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app"]
