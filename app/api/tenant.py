"""
Tenant API — Multi-tenant management endpoints.
Module 8: Tenant info, user management, role changes, audit logs.
ADMIN-only operations where marked.
"""

import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.database import get_db, AsyncSessionLocal
from app.core.security import get_tenant_user, hash_password
from app.core.rbac import require_role, require_min_role, Role
from app.core.rate_limiter import rate_limiter
from app.core.audit import log_audit_event, get_audit_logs, AuditAction
from app.models.models import Tenant, User
from app.schemas.tenant import (
    TenantResponse,
    TenantUserResponse,
    InviteUserRequest,
    UpdateRoleRequest,
    AuditLogResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tenant", tags=["Tenant"])


# ═══════════════════════════════════════════════════════════════
# Tenant Info
# ═══════════════════════════════════════════════════════════════

@router.get("/me", response_model=TenantResponse)
async def get_my_tenant(
    ctx: dict = Depends(get_tenant_user),
    _rate=Depends(rate_limiter.limit("tenant_info", 30, 60)),
):
    """Get current user's tenant info."""
    tenant_id = ctx.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=404, detail="No tenant associated with this user")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Tenant).where(Tenant.id == uuid.UUID(tenant_id))
        )
        tenant = result.scalar_one_or_none()
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")

        return TenantResponse(
            id=tenant.id,
            name=tenant.name,
            subscription_plan=tenant.subscription_plan,
            status=tenant.status,
            created_at=str(tenant.created_at) if tenant.created_at else None,
        )


# ═══════════════════════════════════════════════════════════════
# User Management (ADMIN only)
# ═══════════════════════════════════════════════════════════════

@router.get("/users", response_model=list[TenantUserResponse])
async def list_tenant_users(
    ctx: dict = Depends(require_min_role("ANALYST")),
    _rate=Depends(rate_limiter.limit("tenant_users", 20, 60)),
):
    """List all users in the current tenant. Requires ANALYST or higher."""
    tenant_id = ctx.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant context")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User)
            .where(User.tenant_id == uuid.UUID(tenant_id))
            .order_by(User.created_at.desc())
        )
        users = result.scalars().all()

        return [
            TenantUserResponse(
                id=u.id,
                email=u.email,
                username=u.username,
                full_name=u.full_name,
                role=u.role,
                is_active=u.is_active,
                last_login_at=str(u.last_login_at) if u.last_login_at else None,
                created_at=str(u.created_at) if u.created_at else None,
            )
            for u in users
        ]


@router.post("/invite", response_model=TenantUserResponse, status_code=201)
async def invite_user(
    request: InviteUserRequest,
    ctx: dict = Depends(require_role("ADMIN")),
    _rate=Depends(rate_limiter.limit("tenant_invite", 10, 60)),
):
    """Invite a new user to the tenant. ADMIN only."""
    tenant_id = ctx.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant context")

    # Validate role
    if request.role not in ("ADMIN", "ANALYST", "VIEWER"):
        raise HTTPException(status_code=400, detail="Invalid role. Must be ADMIN, ANALYST, or VIEWER")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Check email uniqueness
            existing = await session.execute(
                select(User).where(User.email == request.email)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=409, detail="Email already registered")

            # Check username uniqueness
            existing_username = await session.execute(
                select(User).where(User.username == request.username)
            )
            if existing_username.scalar_one_or_none():
                raise HTTPException(status_code=409, detail="Username already taken")

            new_user = User(
                tenant_id=uuid.UUID(tenant_id),
                email=request.email,
                username=request.username,
                full_name=request.full_name,
                hashed_password=hash_password(request.password),
                role=request.role,
            )
            session.add(new_user)
            await session.flush()
            user_id = new_user.id

    # Audit log (non-blocking)
    await log_audit_event(
        tenant_id=tenant_id,
        user_id=ctx["user_id"],
        action=AuditAction.USER_INVITE,
        resource_type="user",
        resource_id=str(user_id),
        metadata={"email": request.email, "role": request.role},
    )

    return TenantUserResponse(
        id=user_id,
        email=request.email,
        username=request.username,
        full_name=request.full_name,
        role=request.role,
        is_active=True,
    )


@router.patch("/user/{user_id}/role")
async def update_user_role(
    user_id: uuid.UUID,
    request: UpdateRoleRequest,
    ctx: dict = Depends(require_role("ADMIN")),
    _rate=Depends(rate_limiter.limit("tenant_role", 10, 60)),
):
    """Change a user's role within the tenant. ADMIN only."""
    tenant_id = ctx.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant context")

    if request.role not in ("ADMIN", "ANALYST", "VIEWER"):
        raise HTTPException(status_code=400, detail="Invalid role")

    # Prevent self-demotion
    if str(user_id) == ctx["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(User).where(
                    User.id == user_id,
                    User.tenant_id == uuid.UUID(tenant_id),
                )
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found in your tenant")

            old_role = user.role
            user.role = request.role

    # Audit log (non-blocking)
    await log_audit_event(
        tenant_id=tenant_id,
        user_id=ctx["user_id"],
        action=AuditAction.ROLE_CHANGE,
        resource_type="user",
        resource_id=str(user_id),
        metadata={"old_role": old_role, "new_role": request.role},
    )

    return {"message": f"Role updated: {old_role} → {request.role}", "user_id": str(user_id)}


# ═══════════════════════════════════════════════════════════════
# Audit Logs (ADMIN only)
# ═══════════════════════════════════════════════════════════════

@router.get("/audit", response_model=list[AuditLogResponse])
async def list_audit_logs(
    limit: int = Query(default=50, le=200, ge=1),
    action: str = Query(default=None),
    ctx: dict = Depends(require_role("ADMIN")),
    _rate=Depends(rate_limiter.limit("tenant_audit", 20, 60)),
):
    """List audit logs for the tenant. ADMIN only."""
    tenant_id = ctx.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="No tenant context")

    logs = await get_audit_logs(
        tenant_id=tenant_id,
        limit=limit,
        action_filter=action,
    )
    return [AuditLogResponse(**log) for log in logs]
