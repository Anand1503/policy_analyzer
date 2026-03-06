"""
Pydantic schemas for documents — request/response validation.
Module 1: Enhanced with ProcessingSummary and processing metrics.
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    PROCESSED = "processed"
    CLASSIFIED = "classified"
    ANALYZED = "analyzed"
    FAILED = "failed"


# ─── Upload Response ─────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: UUID
    status: DocumentStatus
    filename: str
    message: str = "Document uploaded successfully"


# ─── Processing Summary (Module 1 core response) ─────────────

class ProcessingSummary(BaseModel):
    """Returned after the document processing pipeline completes."""
    document_id: UUID
    filename: str
    status: str
    processing_time_ms: int
    total_clauses: int
    total_entities: int
    raw_text_length: int


# ─── Document Responses ──────────────────────────────────────

class DocumentResponse(BaseModel):
    id: UUID
    original_filename: str
    file_type: str
    file_size_bytes: int
    status: DocumentStatus
    num_pages: Optional[int] = None
    is_scanned: bool = False
    processing_time_ms: Optional[int] = None
    total_clauses: Optional[int] = None
    total_entities: Optional[int] = None
    uploaded_at: datetime
    extracted_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    analyzed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class DocumentDetail(DocumentResponse):
    extracted_text: Optional[str] = None
    source_url: Optional[str] = None
    error_message: Optional[str] = None
    raw_text_length: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ─── Status Polling ──────────────────────────────────────────

class DocumentStatusResponse(BaseModel):
    document_id: UUID
    status: DocumentStatus
    processing_time_ms: Optional[int] = None
    total_clauses: Optional[int] = None
    total_entities: Optional[int] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
