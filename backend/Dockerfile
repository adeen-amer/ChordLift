FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt && \
    pip install --no-cache-dir soundfile

COPY . .

RUN mkdir -p cache downloads uploads

ENV PORT=8000
ENV CHORD_ENGINE=ml
ENV CHORD_ML_MODEL=chordia
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
