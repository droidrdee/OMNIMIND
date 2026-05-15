"""Application configuration loaded from environment variables.

Settings are parsed once and cached via :func:`get_settings` so that the
configuration is read from disk a single time per process and shared across
all importers.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from ``.env``.

    Every field corresponds to an environment variable of the same name in
    upper-case. ``pydantic-settings`` automatically performs the conversion
    (e.g. ``OPENAI_API_KEY`` -> ``openai_api_key``) and coerces values to the
    declared Python type.
    """

    # --- OpenAI -----------------------------------------------------------
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Tavily Web Search ------------------------------------------------
    tavily_api_key: str

    # --- ChromaDB ---------------------------------------------------------
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "omnimind_docs"

    # --- Embeddings -------------------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Router thresholds ------------------------------------------------
    retrieval_confidence_threshold: float = 0.75
    live_query_keywords: List[str] = [
        "today",
        "latest",
        "current",
        "now",
        "recent",
    ]

    # --- Ingestion --------------------------------------------------------
    max_upload_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval --------------------------------------------------------
    top_k_retrieval: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton :class:`Settings` instance.

    The result is cached with :func:`functools.lru_cache` so that the
    environment is parsed only once. Tests can call ``get_settings.cache_clear()``
    to force a reload after mutating environment variables.
    """

    return Settings()
