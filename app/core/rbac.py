"""
RBAC — Role-Based Access Control.
Module 8: Permission matrix, role enforcement decorators.

Roles:
  ADMIN   — Full access (delete, invite, manage roles)
  ANALYST — Upload, analyze, view
  VIEWER  — View reports only

Design:
  - Pure logic (no DB access)
  - Used as FastAPI dependencies
  - Backward compatible (existing endpoints continue to work)
"""

import logging
from enum import Enum
from typing import Optional
from functools import wraps

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Roles & Permissions
# ═══════════════════════════════════════════════════════════════

class Role(str, Enum):
    ADMIN = "ADMIN"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class Permission(str, Enum):
    # Document operations
    UPLOAD_DOCUMENT = "upload_document"
    VIEW_DOCUMENT = "view_document"
    DELETE_DOCUMENT = "delete_document"
    LIST_DOCUMENTS = "list_documents"

    # Analysis operations
    RUN_CLASSIFICATION = "run_classification"
    RUN_RISK_ANALYSIS = "run_risk_analysis"
    RUN_EXPLANATION = "run_explanation"
    RUN_COMPLIANCE = "run_compliance"
    RUN_SUMMARY = "run_summary"
    VIEW_ANALYSIS = "view_analysis"

    # Evaluation operations
    RUN_EVALUATION = "run_evaluation"
    VIEW_EVALUATION = "view_evaluation"

    # Tenant operations
    MANAGE_USERS = "manage_users"
    INVITE_USER = "invite_user"
    UPDATE_ROLE = "update_role"
    VIEW_TENANT = "view_tenant"
    VIEW_AUDIT = "view_audit"


# Role → Permissions matrix
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.UPLOAD_DOCUMENT,
        Permission.VIEW_DOCUMENT,
        Permission.DELETE_DOCUMENT,
        Permission.LIST_DOCUMENTS,
        Permission.RUN_CLASSIFICATION,
        Permission.RUN_RISK_ANALYSIS,
        Permission.RUN_EXPLANATION,
        Permission.RUN_COMPLIANCE,
        Permission.RUN_SUMMARY,
        Permission.VIEW_ANALYSIS,
        Permission.RUN_EVALUATION,
        Permission.VIEW_EVALUATION,
        Permission.MANAGE_USERS,
        Permission.INVITE_USER,
        Permission.UPDATE_ROLE,
        Permission.VIEW_TENANT,
        Permission.VIEW_AUDIT,
    },
    Role.ANALYST: {
        Permission.UPLOAD_DOCUMENT,
        Permission.VIEW_DOCUMENT,
        Permission.LIST_DOCUMENTS,
        Permission.RUN_CLASSIFICATION,
        Permission.RUN_RISK_ANALYSIS,
        Permission.RUN_EXPLANATION,
        Permission.RUN_COMPLIANCE,
        Permission.RUN_SUMMARY,
        Permission.VIEW_ANALYSIS,
        Permission.RUN_EVALUATION,
        Permission.VIEW_EVALUATION,
        Permission.VIEW_TENANT,
    },
    Role.VIEWER: {
        Permission.VIEW_DOCUMENT,
        Permission.LIST_DOCUMENTS,
        Permission.VIEW_ANALYSIS,
        Permission.VIEW_EVALUATION,
        Permission.VIEW_TENANT,
    },
}

# Role hierarchy (for require_min_role)
ROLE_HIERARCHY = {
    Role.ADMIN: 3,
    Role.ANALYST: 2,
    Role.VIEWER: 1,
}


# ═══════════════════════════════════════════════════════════════
# Permission Checking
# ═══════════════════════════════════════════════════════════════

def has_permission(role: str, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    try:
        r = Role(role)
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS.get(r, set())


def has_min_role(user_role: str, min_role: str) -> bool:
    """Check if user's role meets minimum role requirement."""
    try:
        user_r = Role(user_role)
        min_r = Role(min_role)
    except ValueError:
        return False
    return ROLE_HIERARCHY.get(user_r, 0) >= ROLE_HIERARCHY.get(min_r, 0)


# ═══════════════════════════════════════════════════════════════
# FastAPI Dependencies (Decorators)
# ═══════════════════════════════════════════════════════════════

def require_role(role: str):
    """
    FastAPI dependency that enforces exact role match.

    Usage:
        @router.post("/...", dependencies=[Depends(require_role("ADMIN"))])
        def endpoint(ctx: dict = Depends(require_role("ADMIN"))):
    """
    async def dependency(token: str = Depends(oauth2_scheme)):
        from app.core.security import get_tenant_user
        current_user = await get_tenant_user(token)
        user_role = current_user.get("role", "VIEWER")
        if user_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {role}. Your role: {user_role}",
            )
        return current_user
    return dependency


def require_min_role(min_role: str):
    """
    FastAPI dependency that enforces minimum role level.

    Usage:
        @router.post("/...", dependencies=[Depends(require_min_role("ANALYST"))])
    """
    async def dependency(token: str = Depends(oauth2_scheme)):
        from app.core.security import get_tenant_user
        current_user = await get_tenant_user(token)
        user_role = current_user.get("role", "VIEWER")
        if not has_min_role(user_role, min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires minimum role: {min_role}. Your role: {user_role}",
            )
        return current_user
    return dependency


def require_permission(permission: Permission):
    """
    FastAPI dependency that enforces specific permission.

    Usage:
        @router.delete("/...", dependencies=[Depends(require_permission(Permission.DELETE_DOCUMENT))])
    """
    async def dependency(token: str = Depends(oauth2_scheme)):
        from app.core.security import get_tenant_user
        current_user = await get_tenant_user(token)
        user_role = current_user.get("role", "VIEWER")
        if not has_permission(user_role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission.value}",
            )
        return current_user
    return dependency
