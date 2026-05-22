#!/bin/sh
# Container entrypoint: build a sample index on first boot if missing, then serve.
set -e

INDEX_DIR="${VEKTOR_INDEX_DIR:-/app/index_store}"
INDEX_NAME="${VEKTOR_INDEX_NAME:-sample}"
DENSE_PATH="$INDEX_DIR/$INDEX_NAME.hnsw"

mkdir -p "$INDEX_DIR"

if [ ! -f "$DENSE_PATH" ]; then
    echo "[vektor] no index found at $DENSE_PATH, building from data/sample.jsonl"
    python -m scripts.build_index \
        --source data/sample.jsonl \
        --backend hnsw \
        --out "$INDEX_DIR/$INDEX_NAME" \
    || python scripts/build_index.py \
        --source data/sample.jsonl \
        --backend hnsw \
        --out "$INDEX_DIR/$INDEX_NAME"
    echo "[vektor] index built"
else
    echo "[vektor] reusing existing index at $DENSE_PATH"
fi

export VEKTOR_DENSE_INDEX_PATH="$INDEX_DIR/$INDEX_NAME.hnsw"
export VEKTOR_SPARSE_INDEX_PATH="$INDEX_DIR/$INDEX_NAME.bm25"
export VEKTOR_DOC_STORE_PATH="$INDEX_DIR/$INDEX_NAME.docs.jsonl"
export VEKTOR_ENABLE_RERANKER="${VEKTOR_ENABLE_RERANKER:-false}"

PORT="${PORT:-8000}"
echo "[vektor] starting api on 0.0.0.0:$PORT"
exec uvicorn vektor.api.main:app --host 0.0.0.0 --port "$PORT"
