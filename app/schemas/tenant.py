"""
Tenant Schemas — Pydantic models for Module 8.
"""

from uuid import UUID
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class TenantResponse(BaseModel):
    """Tenant details."""
    id: UUID
    name: str
    subscription_plan: str = "free"
    status: str = "active"
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TenantUserResponse(BaseModel):
    """User within a tenant."""
    id: UUID
    email: str
    username: str
    full_name: Optional[str] = None
    role: str = "ANALYST"
    is_active: bool = True
    last_login_at: Optional[str] = None
    created_at: Optional[str] = None


class InviteUserRequest(BaseModel):
    """Request to invite a user to a tenant."""
    email: str
    username: str
    full_name: Optional[str] = None
    password: str
    role: str = "ANALYST"


class UpdateRoleRequest(BaseModel):
    """Request to change a user's role."""
    role: str  # ADMIN, ANALYST, VIEWER


class AuditLogResponse(BaseModel):
    """Audit log entry."""
    id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: Optional[str] = None
