"""
Intelligent Policy Analyzer — Unified Backend
FastAPI application entry point.
Hardened: model preloading, request ID, metrics, rate limiting, health probes, graceful shutdown.
"""

import os

# ─── Set HuggingFace cache to local models/ directory BEFORE any ML imports ──
# This prevents PermissionError on Windows when HF tries to write to C:\Users\.cache
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_models_dir = os.path.join(_project_root, "models")
os.environ["HF_HOME"] = _models_dir
os.environ["TRANSFORMERS_CACHE"] = _models_dir
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Note: online mode enabled so models download to ./models/ on first use
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.config import settings
from app.core.database import init_db, check_db_health
from app.core.middleware import RequestIdMiddleware, MaxBodySizeMiddleware
from app.core.metrics import MetricsMiddleware, metrics_endpoint

load_dotenv()
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} ({settings.APP_ENV})")

    # ── Database ─────────────────────────────────────────────
    try:
        await init_db()
        logger.info("Database initialized (all 7 models)")
    except Exception as e:
        logger.warning(f"DB init failed (app will still start): {e}")

    # ── Directories ──────────────────────────────────────────
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.EXTRACTED_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    # ── Model Pre-loading ────────────────────────────────────
    if settings.PRELOAD_MODELS:
        try:
            from app.ml.model_loader import preload_models
            preload_models()
        except Exception as e:
            logger.warning(f"Model pre-load failed (will lazy-load on first request): {e}")

    yield

    # ── Graceful Shutdown ────────────────────────────────────
    logger.info("Shutting down...")
    try:
        from app.ml.model_loader import cleanup_models
        cleanup_models()
    except Exception:
        pass
    logger.info("Shutdown complete")


# ─── Create App ──────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered privacy policy and legal document analysis system",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# ─── Middleware (order matters: outermost first) ─────────────
app.add_middleware(RequestIdMiddleware)
if settings.ENABLE_METRICS:
    app.add_middleware(MetricsMiddleware)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.MAX_UPLOAD_MB * 1024 * 1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)

# ─── Include Routers ─────────────────────────────────────────
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.analysis import router as analysis_router
from app.api.evaluation import router as evaluation_router
from app.api.tenant import router as tenant_router

app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis_router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_V1_PREFIX)
app.include_router(tenant_router, prefix=settings.API_V1_PREFIX)


# ─── Prometheus Metrics Endpoint ─────────────────────────────
if settings.ENABLE_METRICS:
    from starlette.routing import Route
    app.routes.append(Route("/metrics", metrics_endpoint))


# ─── Health Endpoints ────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "service": settings.APP_NAME,
        "version": "2.1.0",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }


@app.get("/health/live", tags=["System"])
async def health_live():
    """Liveness probe — is the process running?"""
    return {"status": "alive"}


@app.get("/health/ready", tags=["System"])
async def health_ready():
    """Readiness probe — DB, ML models, ChromaDB, OCR."""
    from app.ml.model_loader import get_models_status

    db_ok = await check_db_health()
    models = get_models_status()
    classifier_ok = models["classifier"]

    # ChromaDB check
    chroma_ok = False
    try:
        import os
        chroma_ok = os.path.isdir(settings.CHROMA_PERSIST_DIR)
    except Exception:
        pass

    # OCR (Tesseract) check
    ocr_ok = False
    try:
        import shutil
        ocr_ok = shutil.which("tesseract") is not None
    except Exception:
        pass

    # Embedding model check
    embedding_ok = False
    try:
        import os
        embedding_ok = os.path.isdir(settings.EMBEDDING_MODEL) or "sentence-transformers" in settings.EMBEDDING_MODEL
    except Exception:
        pass

    all_ready = db_ok and classifier_ok

    return {
        "status": "ready" if all_ready else "degraded",
        "checks": {
            "database": "ok" if db_ok else "unavailable",
            "classifier": "loaded" if classifier_ok else "not_loaded",
            "summarizer": "loaded" if models["summarizer"] else "not_loaded",
            "chromadb": "available" if chroma_ok else "unavailable",
            "ocr_tesseract": "available" if ocr_ok else "unavailable",
            "embedding_model": "available" if embedding_ok else "unavailable",
        },
        "model_versions": {
            "classifier": models["classifier_version"],
            "summarizer": models["summarizer_version"],
        },
    }


@app.get("/health", tags=["System"])
async def health():
    return await health_ready()


@app.get(f"{settings.API_V1_PREFIX}/system/health", tags=["System"])
async def system_health():
    """Full system health check at /api/v1/system/health."""
    return await health_ready()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
