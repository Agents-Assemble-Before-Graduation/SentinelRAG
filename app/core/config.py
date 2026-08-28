"""Application configuration module using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # General App Configuration
    APP_NAME: str = "SentinelRAG"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["*"]

    # Frontend Configuration
    FRONTEND_PORT: int = 8501
    BACKEND_API_URL: str = "http://localhost:8000"

    # PostgreSQL Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/sentinelrag",
        description="Async PostgreSQL connection URL",
    )
    DATABASE_URL_SYNC: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/sentinelrag",
        description="Synchronous PostgreSQL connection URL for Alembic migrations",
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

    # Qdrant Vector Database Configuration
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant service URL")
    QDRANT_API_KEY: str | None = Field(default=None, description="Qdrant API Key if authenticated")
    QDRANT_TIMEOUT: float = Field(default=5.0, description="Qdrant request timeout in seconds")

    # Ingestion & Chunking Configuration
    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=50 * 1024 * 1024, description="Maximum file upload size (50MB)"
    )
    DEFAULT_CHUNK_SIZE: int = Field(default=500, description="Target chunk character size")
    DEFAULT_CHUNK_OVERLAP: int = Field(default=100, description="Chunk overlap character size")
    MIN_CHUNK_SIZE: int = Field(default=50, description="Minimum chunk character size")
    DEFAULT_COLLECTION_NAME: str = Field(
        default="sentinel_chunks", description="Default Qdrant collection name"
    )

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = Field(
        default="fastembed", description="Embedding provider ('fastembed', 'deterministic', 'openai')"
    )
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Default embedding model"
    )
    EMBEDDING_DIMENSION: int = Field(default=384, description="Embedding vector dimensions")
    RERANKER_PROVIDER: str = Field(
        default="fastembed", description="Reranker provider ('fastembed', 'mock')"
    )
    RERANKER_MODEL: str = Field(default="bge-reranker-large", description="Default reranker model")

    # LLM Provider & Model Configuration (to be configured in future phases)
    LLM_PROVIDER: str = Field(default="openai", description="Default LLM provider")
    LLM_MODEL: str = Field(default="gpt-4o", description="Default LLM model name")
    LLM_API_KEY: str | None = Field(default=None, description="LLM provider API key")
    LLM_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible LLM API endpoint",
    )

    # RAG Query Configuration
    RAG_RETRIEVAL_MODE: str = Field(
        default="dense", description="Default retrieval mode ('dense', 'bm25', 'hybrid')"
    )
    RAG_RERANK_ENABLED: bool = Field(
        default=False, description="Whether to enable cross-encoder reranking by default"
    )
    RAG_DENSE_WEIGHT: float = Field(
        default=0.5, description="Weight of dense retrieval in hybrid scoring"
    )
    RAG_BM25_WEIGHT: float = Field(
        default=0.5, description="Weight of BM25 retrieval in hybrid scoring"
    )
    RAG_TOP_K: int = Field(default=5, description="Number of chunks to retrieve per query")
    RAG_SCORE_THRESHOLD: float = Field(
        default=0.3, description="Minimum similarity score for retrieved chunks"
    )
    RAG_MAX_CONTEXT_CHARS: int = Field(
        default=12000, description="Maximum characters in assembled RAG context"
    )

    # ── Phase 8: Security & Cost Limits ───────────────────────────────────────
    MAX_LLM_CALLS: int = Field(
        default=10,
        description="Hard cap on total LLM calls per query (across all agents)",
    )
    MAX_CONTEXT_TOKENS: int = Field(
        default=8192,
        description="Maximum token budget for context window",
    )
    MAX_QUERY_LENGTH: int = Field(
        default=2000,
        description="Maximum allowed question length in characters",
    )
    REQUEST_TIMEOUT: float = Field(
        default=30.0,
        description="Per-request timeout in seconds for the full RAG pipeline",
    )
    MAX_REPAIR_ATTEMPTS: int = Field(
        default=2,
        description="Hard cap on repair loop iterations (overrides hardcoded value)",
    )
    MAX_COST_USD_PER_QUERY: float = Field(
        default=0.10,
        description="Maximum estimated USD cost allowed per single query",
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'. Must be one of {valid_levels}")
        return upper_v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() == "testing"


@lru_cache
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()
