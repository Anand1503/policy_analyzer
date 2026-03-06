"""
Document API — Upload, List, Get, Delete, Status Polling.
Hardened: MIME content validation, rate limiting on upload.
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, get_tenant_user
from app.core.config import settings
from app.core.rate_limiter import rate_limiter
from app.core.rbac import require_min_role, require_permission, Permission
from app.core.audit import log_audit_event, AuditAction
from app.services.document_service import DocumentService
from app.schemas.document import (
    UploadResponse, DocumentResponse, DocumentDetail,
    DocumentListResponse, DocumentStatus, DocumentStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


# ─── Upload ──────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_url: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate=Depends(rate_limiter.limit("upload", settings.RATE_LIMIT_UPLOAD, 60)),
):
    """
    Upload a document for processing.
    Validates: extension, size, MIME content match, executable detection.
    """
    # ── Extension + size validation (existing) ───────────────
    ok, err = DocumentService.validate_file(file.filename or "", file.size or 0)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    # ── MIME content validation (Fix 4) ──────────────────────
    content = await file.read()
    await file.seek(0)  # Reset for save_file

    from app.core.file_validation import validate_upload_content
    valid, mime_err = validate_upload_content(file.filename or "", content)
    if not valid:
        logger.warning(f"[upload] MIME rejection: {file.filename} — {mime_err}")
        raise HTTPException(status_code=400, detail=mime_err)

    # Save file
    doc_id = uuid.uuid4()
    ext = DocumentService.get_extension(file.filename or "")
    file_path, stored_name = await DocumentService.save_file(file, doc_id)

    # Create DB record
    await DocumentService.create_document_record(
        db, doc_id, user_id, stored_name, file.filename or "unknown", ext, file.size or 0, source_url
    )

    # Schedule background processing pipeline
    background_tasks.add_task(DocumentService.process_document, doc_id, file_path, ext)

    return UploadResponse(
        document_id=doc_id,
        status=DocumentStatus.UPLOADED,
        filename=file.filename or "unknown",
        message="Document uploaded. Processing pipeline started.",
    )


# ─── Status Polling ──────────────────────────────────────────

@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Poll document processing status."""
    doc = await DocumentService.get_document_status(db, document_id, user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        processing_time_ms=doc.processing_time_ms,
        total_clauses=doc.total_clauses,
        total_entities=doc.total_entities,
        error_message=doc.error_message,
    )


# ─── List ────────────────────────────────────────────────────

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents for the current user."""
    docs = await DocumentService.list_user_documents(db, user_id)
    return DocumentListResponse(documents=docs, total=len(docs))


# ─── Get Detail ──────────────────────────────────────────────

@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get full document details including extracted text and metrics."""
    doc = await DocumentService.get_document(db, document_id, user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ─── Delete ──────────────────────────────────────────────────

@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    ctx: dict = Depends(require_permission(Permission.DELETE_DOCUMENT)),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document. ADMIN only."""
    user_id = ctx["user_id"]
    ok = await DocumentService.delete_document(db, document_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")

    # Audit log (non-blocking)
    await log_audit_event(
        tenant_id=ctx.get("tenant_id", ""),
        user_id=user_id,
        action=AuditAction.DELETE,
        resource_type="document",
        resource_id=str(document_id),
    )
