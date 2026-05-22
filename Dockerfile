FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.st_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./

# CPU-only torch keeps image size & cold-start RAM reasonable on Railway
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision --no-cache-dir \
    && pip install -r requirements.txt

COPY vektor/ ./vektor/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Pre-warm the model cache so first request doesn't pay the download cost
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/healthz" || exit 1

ENTRYPOINT ["./scripts/entrypoint.sh"]
