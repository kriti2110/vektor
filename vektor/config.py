from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VEKTOR_", env_file=".env", extra="ignore")

    # Embedding
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_batch_size: int = 64
    embed_cache_path: Path = Path("./embeddings_cache.sqlite")

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 32

    # Index
    index_path: Path = Path("./index_store")
    index_backend: str = "hnsw"  # flat | hnsw | ivf
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100

    # Rerank
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 50

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    query_cache_ttl: int = 3600

    # Feedback
    feedback_db_path: Path = Path("./feedback_store.sqlite")


settings = Settings()
