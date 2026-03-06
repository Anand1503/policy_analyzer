"""
Audit Trail Engine — Non-blocking audit logging.
Module 8: Logs all critical operations for tenant compliance.

Design:
  - Non-blocking: fire-and-forget async writes
  - Tenant-scoped: every log includes tenant_id
  - No effect on request latency
"""

import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Audit Actions
# ═══════════════════════════════════════════════════════════════

class AuditAction:
    UPLOAD = "UPLOAD"
    DELETE = "DELETE"
    CLASSIFY = "CLASSIFY"
    RISK_ANALYZE = "RISK_ANALYZE"
    EXPLAIN = "EXPLAIN"
    COMPLIANCE_RUN = "COMPLIANCE_RUN"
    SUMMARY_RUN = "SUMMARY_RUN"
    EVALUATION_RUN = "EVALUATION_RUN"
    USER_INVITE = "USER_INVITE"
    ROLE_CHANGE = "ROLE_CHANGE"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"


# ═══════════════════════════════════════════════════════════════
# Audit Logger
# ═══════════════════════════════════════════════════════════════

async def log_audit_event(
    tenant_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an audit event non-blocking.

    Fire-and-forget: errors are logged but never propagate to caller.
    """
    try:
        asyncio.create_task(
            _write_audit_log(
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                metadata=metadata,
            )
        )
    except Exception as e:
        # Never let audit logging break the request
        logger.warning(f"[audit] Failed to schedule audit log: {e}")


async def _write_audit_log(
    tenant_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> None:
    """Write audit log to database (non-blocking background task)."""
    try:
        from app.core.database import AsyncSessionLocal
        from app.models.models import AuditLog

        async with AsyncSessionLocal() as session:
            async with session.begin():
                log_entry = AuditLog(
                    tenant_id=uuid.UUID(tenant_id) if tenant_id else None,
                    user_id=uuid.UUID(user_id) if user_id else None,
                    action=action,
                    resource_type=resource_type,
                    resource_id=uuid.UUID(resource_id) if resource_id else None,
                    metadata_json=metadata,
                )
                session.add(log_entry)

        logger.debug(
            f"[audit] {action} by {user_id} on {resource_type}"
            f"{'/' + resource_id if resource_id else ''}"
        )
    except Exception as e:
        # Never let audit logging break anything
        logger.warning(f"[audit] Failed to write audit log: {e}")


async def get_audit_logs(
    tenant_id: str,
    limit: int = 50,
    action_filter: Optional[str] = None,
) -> list:
    """Retrieve audit logs for a tenant (tenant-scoped)."""
    from app.core.database import AsyncSessionLocal
    from app.models.models import AuditLog
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        query = (
            select(AuditLog)
            .where(AuditLog.tenant_id == uuid.UUID(tenant_id))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        if action_filter:
            query = query.where(AuditLog.action == action_filter)

        result = await session.execute(query)
        logs = result.scalars().all()

        return [
            {
                "id": str(log.id),
                "tenant_id": str(log.tenant_id) if log.tenant_id else None,
                "user_id": str(log.user_id) if log.user_id else None,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": str(log.resource_id) if log.resource_id else None,
                "metadata": log.metadata_json,
                "created_at": str(log.created_at) if log.created_at else None,
            }
            for log in logs
        ]
