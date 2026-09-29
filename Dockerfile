# ============================================================
# OlmOCR Production Pipeline — Dockerfile
# FastAPI + llama-mtmd-cli + OpenCV preprocessing
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ cmake \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime libs for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY ocr_engine.py ocr_api.py run_olm_ocr.py ./

# Directories for models, uploads, output
RUN mkdir -p /models /uploads /ocr_output && \
    useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app /models /uploads /ocr_output

USER appuser

# Environment defaults (override via K8s env or docker run -e)
ENV OCR_MODEL_PATH=/models/allenai_olmOCR-2-7B-1025-Q8_0.gguf \
    OCR_MMPROJ_PATH=/models/mmproj-allenai_olmOCR-2-7B-1025-f16.gguf \
    OCR_CLI_PATH=/usr/local/bin/llama-mtmd-cli \
    OCR_UPLOAD_DIR=/uploads \
    OCR_OUTPUT_DIR=/ocr_output \
    OCR_GPU_LAYERS=999 \
    OCR_MAX_TOKENS=6144 \
    OCR_CTX_SIZE=8192

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8003}/health || exit 1

EXPOSE 8003

CMD ["sh", "-c", "uvicorn ocr_api:app --host 0.0.0.0 --port ${PORT:-8003}"]
