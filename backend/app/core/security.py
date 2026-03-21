"""
JWT authentication & password hashing utilities.
Module 8: Enhanced with tenant_id, role, tenant status validation.
Backward compatible: existing tokens without tenant_id/role still work.
"""

import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings

# ─── Password Hashing ───────────────────────────────────────
# Use bcrypt directly (avoids passlib 1.7/bcrypt 4.x incompatibility).
# SHA-256 pre-hash + base64 ensures passwords > 72 bytes are handled safely.

def _prepare_password(password: str) -> bytes:
    """SHA-256 hash the password then base64-encode it for safe bcrypt input."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)

def hash_password(password: str) -> str:
    prepared = _prepare_password(password)
    hashed = bcrypt.hashpw(prepared, bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    prepared = _prepare_password(plain_password)
    return bcrypt.checkpw(prepared, hashed_password.encode("utf-8"))

# ─── JWT Tokens ──────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT token with user_id, tenant_id, and role.

    Expected data keys:
      - sub: user_id (required)
      - tenant_id: tenant UUID string (optional, for multi-tenant)
      - role: ADMIN/ANALYST/VIEWER (optional, defaults to ANALYST)
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

# ─── FastAPI Dependency (backward compatible) ────────────────
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency that extracts and validates the current user from JWT.
    Returns user_id (UUID string) from the token payload.
    Backward compatible: works with old tokens that don't have tenant_id/role.
    """
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identifier",
        )
    return user_id


# ─── Tenant-Aware Dependency (Module 8) ─────────────────────
async def get_tenant_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency that extracts full tenant context from JWT.

    Returns dict:
      {
          "user_id": "...",
          "tenant_id": "..." or None,
          "role": "ADMIN" | "ANALYST" | "VIEWER"
      }

    Validates:
      - Token is valid
      - user_id is present
      - Tenant status is active (if tenant_id provided)
    """
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identifier",
        )

    tenant_id = payload.get("tenant_id")
    role = payload.get("role", "ANALYST")

    # Validate tenant is active (if tenant_id present)
    if tenant_id:
        from app.core.database import AsyncSessionLocal
        from app.models.models import Tenant
        import uuid as _uuid

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.id == _uuid.UUID(tenant_id))
            )
            tenant = result.scalar_one_or_none()
            if tenant and tenant.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Tenant is {tenant.status}. Access denied.",
                )

    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
    }
