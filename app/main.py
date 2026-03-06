"""
Intelligent Policy Analyzer — Unified Backend
FastAPI application entry point.
Hardened: model preloading, request ID, metrics, rate limiting, health probes, graceful shutdown.
"""

import os
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
    """Readiness probe — DB, ML models, Ollama."""
    from app.ml.model_loader import get_models_status

    db_ok = await check_db_health()
    models = get_models_status()
    classifier_ok = models["classifier"]

    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(settings.OLLAMA_BASE_URL)
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    all_ready = db_ok and classifier_ok

    return {
        "status": "ready" if all_ready else "degraded",
        "checks": {
            "database": "ok" if db_ok else "unavailable",
            "classifier": "loaded" if classifier_ok else "not_loaded",
            "summarizer": "loaded" if models["summarizer"] else "not_loaded",
            "ollama": "connected" if ollama_ok else "unavailable",
        },
        "model_versions": {
            "classifier": models["classifier_version"],
            "summarizer": models["summarizer_version"],
        },
        "model_checksums": {
            "classifier": models.get("classifier_checksum", {}),
            "summarizer": models.get("summarizer_checksum", {}),
        },
    }


@app.get("/health", tags=["System"])
async def health():
    return await health_ready()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
