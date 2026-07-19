FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
# madmom ships source-only and needs numpy/Cython present before its C
# extensions build; --no-build-isolation lets that build see them (see the
# comment above madmom in requirements-ml.txt for the full rationale).
RUN pip install --no-cache-dir numpy==1.23.5 Cython && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt -r requirements-ml.txt && \
    pip install --no-cache-dir soundfile && \
    python -c "import madmom"

COPY . .

RUN mkdir -p cache downloads uploads

ENV PORT=8000
ENV CHORD_ENGINE=ml
ENV CHORD_ML_MODEL=chordia
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
