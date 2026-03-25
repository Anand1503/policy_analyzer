"""
Application configuration — single source of truth for all settings.
Hardened: SECRET_KEY validation, rate limit config, ML guards, timeout, metrics.
"""

import os
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # ─── App ─────────────────────────────────────────────────
    APP_NAME: str = "Intelligent Policy Analyzer"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # ─── Server ──────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # ─── Database ────────────────────────────────────────────
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "policy_db"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Sync URL for Alembic offline mode."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ─── JWT Auth ────────────────────────────────────────────
    SECRET_KEY: str = "dev-only-change-me-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # ─── File Upload ─────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads/raw_documents"
    EXTRACTED_DIR: str = "./uploads/extracted_text"
    MAX_UPLOAD_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "doc", "html", "txt", "png", "jpg", "jpeg", "tiff", "bmp"]

    # ─── ML Models (all local, no external APIs) ──────────────
    EMBEDDING_MODEL: str = "./models/all-MiniLM-L6-v2"
    CLASSIFIER_MODEL: str = "./models/legal-bert"
    SUMMARIZER_MODEL: str = "./models/t5-base"
    CHROMA_PERSIST_DIR: str = "./data/vector_store"
    SPACY_MODEL: str = "en_core_web_sm"
    LEGAL_BERT_NUM_LABELS: int = 10

    # ─── ML Robustness ───────────────────────────────────────
    MAX_CLAUSES: int = 500
    CLASSIFIER_MAX_LENGTH: int = 512
    ML_INFERENCE_TIMEOUT_SECONDS: int = 120
    PRELOAD_MODELS: bool = True

    # ─── Embedding Validation ────────────────────────────────
    EXPECTED_EMBEDDING_DIM: int = 384       # all-MiniLM-L6-v2 = 384

    # ─── Rate Limiting (per-user per endpoint) ───────────────
    RATE_LIMIT_CLASSIFY: int = 10           # /classify → 10/min
    RATE_LIMIT_RISK: int = 10               # /risk → 10/min
    RATE_LIMIT_EXPLAIN: int = 20            # /explain → 20/min
    RATE_LIMIT_UPLOAD: int = 30             # /upload → 30/min

    # ─── Retry ───────────────────────────────────────────────
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 1.0         # 1s → 2s → 4s

    # ─── Metrics ─────────────────────────────────────────────
    ENABLE_METRICS: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# ─── Security Validation ─────────────────────────────────────
if settings.APP_ENV == "production" and "dev-only" in settings.SECRET_KEY:
    import warnings
    warnings.warn(
        "CRITICAL: SECRET_KEY contains default value in production! "
        "Set SECRET_KEY environment variable with: openssl rand -hex 32",
        RuntimeWarning,
        stacklevel=2,
    )

# ─── Model Path Validation ───────────────────────────────────
_model_checks = {
    "CLASSIFIER_MODEL": settings.CLASSIFIER_MODEL,
    "SUMMARIZER_MODEL": settings.SUMMARIZER_MODEL,
    "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
}
for _name, _path in _model_checks.items():
    if _path.startswith("./") and not os.path.isdir(_path):
        import warnings
        warnings.warn(
            f"ML model path '{_path}' ({_name}) does not exist. "
            f"Models will be downloaded on first use, which may take time.",
            UserWarning,
            stacklevel=2,
        )
