"""
Async SQLAlchemy database engine and session management.
Hardened: All models imported, production pool config, health check helper.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Async engine with production-grade pool settings
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections every 30 min
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Declarative base for all ORM models
Base = declarative_base()


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables (dev only — use Alembic in prod)."""
    async with engine.begin() as conn:
        # Import ALL models so SQLAlchemy registers them
        from app.models.models import (  # noqa: F401
            Tenant, User, Document, Clause, AnalysisResult,
            ClauseClassification, ClauseRiskScore, ClauseExplanation,
            DocumentComplianceReport, DocumentSummary, ClauseSummary,
            EvaluationRun, AuditLog,
        )
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified (all 13 models)")


async def check_db_health() -> bool:
    """Ping the database — used by /health/ready."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
