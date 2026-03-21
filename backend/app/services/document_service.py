"""
Document Service — Intelligent Document Processing & Clause Intelligence Engine.

This is the core service for Module 1. It orchestrates the complete document
processing pipeline following Clean Architecture principles:

    API Layer → DocumentService → ML Layer → Data Layer

Pipeline:
    extract_text() → clean_text() → split_clauses() →
    perform_ner() → generate_embeddings() → store_results()

Each step is modular, instrumented with structured logging and timing,
wrapped in exception handling, and returns structured objects.
"""

import uuid
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.exceptions import (
    DocumentProcessingError,
    FileValidationError,
    ExtractionError,
    ClauseSegmentationError,
    NERError,
    EmbeddingError,
)
from app.models.models import Document, Clause

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

MIME_MAP = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "html": "text/html",
    "htm": "text/html",
    "txt": "text/plain",
}


# ═══════════════════════════════════════════════════════════════
# Document Service
# ═══════════════════════════════════════════════════════════════

class DocumentService:
    """
    Production-grade document processing service.
    
    Responsibilities:
    - File validation and storage
    - Text extraction orchestration
    - Clause intelligence pipeline (clean → split → NER → embed → store)
    - Processing lifecycle management (status transitions, timing, metrics)
    - CRUD operations for document records
    """

    # ──────────────────────────────────────────────────────────
    # File Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def get_extension(filename: str) -> str:
        """Extract lowercase extension from filename."""
        return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    @staticmethod
    def validate_file(filename: str, file_size: int) -> tuple[bool, str]:
        """
        Validate file type and size constraints.
        Returns (ok, error_message).
        """
        ext = DocumentService.get_extension(filename)
        if ext not in settings.ALLOWED_EXTENSIONS:
            return False, f"File type '.{ext}' not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        if file_size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            return False, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit"
        if file_size == 0:
            return False, "File is empty"
        return True, ""

    @staticmethod
    async def save_file(file, doc_id: uuid.UUID) -> tuple[str, str]:
        """
        Save uploaded file to disk.
        Returns (absolute_file_path, stored_filename).
        """
        ext = DocumentService.get_extension(file.filename or "unknown.txt")
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{doc_id}.{ext}"
        file_path = upload_dir / stored_name
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"[upload] File saved: {stored_name} ({file_path})")
        return str(file_path), stored_name

    @staticmethod
    async def create_document_record(
        db: AsyncSession,
        doc_id: uuid.UUID,
        user_id: str,
        filename: str,
        original_filename: str,
        ext: str,
        file_size: int,
        source_url: Optional[str] = None,
    ) -> Document:
        """Insert initial document record with UPLOADED status."""
        doc = Document(
            id=doc_id,
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            file_type=ext,
            file_size_bytes=file_size,
            mime_type=MIME_MAP.get(ext, "application/octet-stream"),
            source_url=source_url,
        )
        db.add(doc)
        await db.commit()
        logger.info(f"[db] Document record created: {doc_id} (status=uploaded)")
        return doc

    # ──────────────────────────────────────────────────────────
    # MAIN PIPELINE: process_document()
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def process_document(
        document_id: uuid.UUID,
        file_path: str,
        file_type: str,
    ) -> Dict[str, Any]:
        """
        Full document processing pipeline (runs as BackgroundTask).
        
        Pipeline steps:
            1. extract_text()       — via extractors/
            2. clean_text()         — normalize whitespace, remove artifacts
            3. split_clauses()      — semantic clause segmentation
            4. perform_ner()        — per-clause entity extraction
            5. generate_embeddings()— store clause vectors in ChromaDB
            6. store_results()      — persist clauses + metrics to PostgreSQL
            
        Status transitions:
            UPLOADED → PROCESSING → PROCESSED
            UPLOADED → PROCESSING → FAILED (on error)
            
        Returns:
            ProcessingSummary dict (also written to DB)
        """
        from app.core.database import AsyncSessionLocal

        pipeline_start = time.perf_counter()
        step_timings = {}
        doc_id_str = str(document_id)

        logger.info(f"{'='*60}")
        logger.info(f"[pipeline] START document processing: {doc_id_str}")
        logger.info(f"[pipeline] File: {file_path} (type={file_type})")
        logger.info(f"{'='*60}")

        try:
            # ── Mark as PROCESSING ───────────────────────────
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    doc = await session.get(Document, document_id)
                    if not doc:
                        raise DocumentProcessingError(
                            f"Document {doc_id_str} not found in database",
                            document_id=doc_id_str,
                            step="init",
                        )
                    doc.status = "processing"
                    original_filename = doc.original_filename

            logger.info(f"[pipeline] Status → PROCESSING")

            # ── Step 1: Extract Text ─────────────────────────
            t0 = time.perf_counter()
            extracted_text = DocumentService._extract_text(file_path, file_type, doc_id_str)
            step_timings["extraction"] = int((time.perf_counter() - t0) * 1000)
            raw_text_length = len(extracted_text)
            logger.info(
                f"[pipeline] Step 1/6 ✓ extract_text: {raw_text_length} chars "
                f"({step_timings['extraction']}ms)"
            )

            # ── Step 2: Clean Text ───────────────────────────
            t0 = time.perf_counter()
            cleaned_text = DocumentService._clean_text(extracted_text)
            step_timings["cleaning"] = int((time.perf_counter() - t0) * 1000)
            logger.info(
                f"[pipeline] Step 2/6 ✓ clean_text: {len(cleaned_text)} chars "
                f"({step_timings['cleaning']}ms)"
            )

            # ── Step 3: Split Clauses ────────────────────────
            t0 = time.perf_counter()
            clauses = DocumentService._split_clauses(cleaned_text, doc_id_str)
            step_timings["clause_splitting"] = int((time.perf_counter() - t0) * 1000)
            logger.info(
                f"[pipeline] Step 3/6 ✓ split_clauses: {len(clauses)} clauses "
                f"({step_timings['clause_splitting']}ms)"
            )

            # ── Step 4: NER per clause ───────────────────────
            t0 = time.perf_counter()
            clauses_with_entities = DocumentService._perform_ner(clauses, doc_id_str)
            total_entities = sum(c.get("entity_count", 0) for c in clauses_with_entities)
            step_timings["ner"] = int((time.perf_counter() - t0) * 1000)
            logger.info(
                f"[pipeline] Step 4/6 ✓ perform_ner: {total_entities} entities "
                f"({step_timings['ner']}ms)"
            )

            # ── Step 5: Generate Embeddings ──────────────────
            t0 = time.perf_counter()
            embedding_ids = DocumentService._generate_embeddings(
                doc_id_str, clauses_with_entities
            )
            step_timings["embeddings"] = int((time.perf_counter() - t0) * 1000)
            logger.info(
                f"[pipeline] Step 5/6 ✓ generate_embeddings: {len(embedding_ids)} stored "
                f"({step_timings['embeddings']}ms)"
            )

            # Attach embedding IDs to clauses
            for clause, emb_id in zip(clauses_with_entities, embedding_ids):
                clause["embedding_id"] = emb_id

            # ── Step 6: Store Results ────────────────────────
            t0 = time.perf_counter()
            processing_time_ms = int((time.perf_counter() - pipeline_start) * 1000)

            await DocumentService._store_results(
                document_id=document_id,
                extracted_text=extracted_text,
                clauses=clauses_with_entities,
                raw_text_length=raw_text_length,
                total_entities=total_entities,
                processing_time_ms=processing_time_ms,
                file_type=file_type,
            )
            step_timings["storage"] = int((time.perf_counter() - t0) * 1000)

            # Recalculate total time after storage
            processing_time_ms = int((time.perf_counter() - pipeline_start) * 1000)

            logger.info(
                f"[pipeline] Step 6/6 ✓ store_results "
                f"({step_timings['storage']}ms)"
            )

            # ── Summary ──────────────────────────────────────
            summary = {
                "document_id": doc_id_str,
                "filename": original_filename,
                "status": "PROCESSED",
                "processing_time_ms": processing_time_ms,
                "total_clauses": len(clauses_with_entities),
                "total_entities": total_entities,
                "raw_text_length": raw_text_length,
                "step_timings": step_timings,
            }

            logger.info(f"{'='*60}")
            logger.info(
                f"[pipeline] COMPLETE: {doc_id_str} → "
                f"{len(clauses_with_entities)} clauses, "
                f"{total_entities} entities, "
                f"{processing_time_ms}ms total"
            )
            logger.info(f"[pipeline] Timings: {step_timings}")
            logger.info(f"{'='*60}")

            return summary

        except (DocumentProcessingError, ExtractionError,
                ClauseSegmentationError, NERError, EmbeddingError) as e:
            # Known pipeline errors — log and mark failed
            # Do NOT re-raise: this runs as a BackgroundTask, re-raising
            # would crash the ASGI application.
            logger.error(
                f"[pipeline] FAILED at step '{e.step}': {str(e)}",
                exc_info=True,
            )
            await DocumentService._mark_failed(document_id, str(e))
            return {"document_id": doc_id_str, "status": "FAILED", "error": str(e)}

        except Exception as e:
            # Unexpected errors
            logger.error(
                f"[pipeline] UNEXPECTED FAILURE for {doc_id_str}: {str(e)}",
                exc_info=True,
            )
            await DocumentService._mark_failed(document_id, f"Unexpected: {str(e)}")
            return {"document_id": doc_id_str, "status": "FAILED", "error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Pipeline Step Implementations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(file_path: str, file_type: str, doc_id: str) -> str:
        """Step 1: Extract raw text from document using appropriate extractor."""
        from app.extractors import pdf_extractor, docx_extractor, html_extractor, txt_extractor

        try:
            if file_type == "pdf":
                result = pdf_extractor.extract_pdf(file_path, doc_id)
            elif file_type in ("docx", "doc"):
                result = docx_extractor.extract_docx(file_path, doc_id)
            elif file_type in ("html", "htm"):
                result = html_extractor.extract_html(file_path, doc_id)
            elif file_type == "txt":
                result = txt_extractor.extract_txt(file_path, doc_id)
            else:
                raise ExtractionError(
                    f"Unsupported file type: {file_type}",
                    document_id=doc_id,
                    step="extract_text",
                )

            text = result.get("text", "")
            if not text or len(text.strip()) < 5:
                raise ExtractionError(
                    f"Extraction produced insufficient text ({len(text)} chars). "
                    f"The document may be a scanned image. OCR tools (Tika/Tesseract) are not installed.",
                    document_id=doc_id,
                    step="extract_text",
                )
            return text

        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"Text extraction failed: {str(e)}",
                document_id=doc_id,
                step="extract_text",
            )

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Step 2: Clean and normalize extracted text."""
        from app.ml.preprocessing import clean_text
        try:
            cleaned = clean_text(raw_text)
            return cleaned
        except Exception as e:
            raise DocumentProcessingError(
                f"Text cleaning failed: {str(e)}",
                step="clean_text",
            )

    @staticmethod
    def _split_clauses(cleaned_text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Step 3: Split text into semantic clause units."""
        from app.ml.preprocessing import extract_clauses
        try:
            clauses = extract_clauses(cleaned_text)
            if not clauses:
                logger.warning(f"[pipeline] No clauses extracted from {doc_id}")
                return []
            return clauses
        except Exception as e:
            raise ClauseSegmentationError(
                f"Clause segmentation failed: {str(e)}",
                document_id=doc_id,
                step="split_clauses",
            )

    @staticmethod
    def _perform_ner(
        clauses: List[Dict[str, Any]],
        doc_id: str,
    ) -> List[Dict[str, Any]]:
        """Step 4: Run NER on each clause and attach entities."""
        from app.ml.ner import extract_clause_entities

        enriched = []
        for clause in clauses:
            try:
                entities = extract_clause_entities(clause["text"])
                enriched.append({
                    **clause,
                    "entities": entities,
                    "entity_count": len(entities),
                })
            except NERError:
                # NER failure on a single clause is non-fatal
                logger.warning(
                    f"[pipeline] NER failed for clause {clause['index']} in {doc_id}, "
                    f"continuing with empty entities"
                )
                enriched.append({
                    **clause,
                    "entities": [],
                    "entity_count": 0,
                })
            except Exception as e:
                logger.warning(f"[pipeline] NER unexpected error clause {clause['index']}: {e}")
                enriched.append({**clause, "entities": [], "entity_count": 0})

        return enriched

    @staticmethod
    def _generate_embeddings(
        doc_id: str,
        clauses: List[Dict[str, Any]],
    ) -> List[str]:
        """Step 5: Generate and store clause embeddings in ChromaDB.
        Hardened: embedding dimension validation (Fix 7).
        """
        from app.ml.rag import get_rag_service
        from app.core.config import settings

        if not clauses:
            return []

        try:
            rag = get_rag_service()
            embedding_ids = rag.store_clause_embeddings(doc_id, clauses)

            # Fix 7: Embedding dimension validation
            try:
                expected_dim = settings.EXPECTED_EMBEDDING_DIM
                # If the RAG service exposes stored embeddings, validate sample
                if hasattr(rag, 'get_embedding_dimension'):
                    actual_dim = rag.get_embedding_dimension()
                    if actual_dim != expected_dim:
                        logger.error(
                            f"[embedding] Dimension mismatch: expected={expected_dim}, "
                            f"actual={actual_dim}. Data may be corrupted."
                        )
                        raise EmbeddingError(
                            f"Embedding dimension mismatch: {actual_dim} != {expected_dim}",
                            document_id=doc_id,
                            step="embedding_validation",
                        )
                    logger.info(f"[embedding] Dimension validated: {actual_dim}")
            except EmbeddingError:
                raise
            except Exception as ve:
                logger.debug(f"[embedding] Dimension check skipped: {ve}")

            return embedding_ids
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Embedding generation failed: {str(e)}",
                document_id=doc_id,
                step="generate_embeddings",
            )

    @staticmethod
    async def _store_results(
        document_id: uuid.UUID,
        extracted_text: str,
        clauses: List[Dict[str, Any]],
        raw_text_length: int,
        total_entities: int,
        processing_time_ms: int,
        file_type: str,
    ) -> None:
        """Step 6: Persist all results to PostgreSQL.
        Hardened: Chroma cleanup on DB commit failure (Fix 8).
        """
        from app.core.database import AsyncSessionLocal

        # Collect embedding IDs for potential Chroma cleanup
        embedding_ids = [c.get("embedding_id") for c in clauses if c.get("embedding_id")]

        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    # Clear old clauses (idempotent re-processing)
                    old_clauses = await session.execute(
                        select(Clause).where(Clause.document_id == document_id)
                    )
                    for old in old_clauses.scalars():
                        await session.delete(old)

                    # Insert clauses
                    for clause_data in clauses:
                        session.add(Clause(
                            document_id=document_id,
                            clause_text=clause_data["text"],
                            clause_index=clause_data["index"],
                            embedding_id=clause_data.get("embedding_id"),
                            entity_count=clause_data.get("entity_count", 0),
                            entities=clause_data.get("entities"),
                        ))

                    # Update document record
                    doc = await session.get(Document, document_id)
                    if doc:
                        doc.extracted_text = extracted_text
                        doc.status = "processed"
                        doc.raw_text_length = raw_text_length
                        doc.total_clauses = len(clauses)
                        doc.total_entities = total_entities
                        doc.processing_time_ms = processing_time_ms
                        doc.extracted_at = datetime.now(timezone.utc)
                        doc.processed_at = datetime.now(timezone.utc)
                        doc.error_message = None

            logger.info(
                f"[db] Stored {len(clauses)} clauses + document metrics for {document_id}"
            )

        except Exception as e:
            # Fix 8: Chroma partial cleanup — delete orphaned embeddings
            if embedding_ids:
                logger.warning(
                    f"[chroma_cleanup] DB commit failed. Deleting {len(embedding_ids)} "
                    f"orphaned embeddings for {document_id}"
                )
                try:
                    from app.ml.rag import get_rag_service
                    rag = get_rag_service()
                    if hasattr(rag, 'delete_embeddings'):
                        rag.delete_embeddings(embedding_ids)
                    elif hasattr(rag, 'collection'):
                        rag.collection.delete(ids=embedding_ids)
                    logger.info(f"[chroma_cleanup] Cleaned {len(embedding_ids)} orphaned embeddings")
                except Exception as cleanup_err:
                    logger.error(f"[chroma_cleanup] Cleanup failed: {cleanup_err}")
            raise

    @staticmethod
    async def _mark_failed(document_id: uuid.UUID, error_message: str) -> None:
        """Mark document as FAILED with error details."""
        from app.core.database import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    doc = await session.get(Document, document_id)
                    if doc:
                        doc.status = "failed"
                        doc.error_message = error_message[:2000]  # Cap length
            logger.info(f"[db] Document {document_id} marked as FAILED")
        except Exception as e:
            logger.error(f"[db] Failed to mark document as failed: {e}")

    # ──────────────────────────────────────────────────────────
    # CRUD Operations
    # ──────────────────────────────────────────────────────────

    @staticmethod
    async def list_user_documents(db: AsyncSession, user_id: str):
        """Get all documents belonging to a user, ordered by upload date."""
        result = await db.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.uploaded_at.desc())
        )
        return result.scalars().all()

    @staticmethod
    async def get_document(
        db: AsyncSession, document_id: uuid.UUID, user_id: str
    ) -> Optional[Document]:
        """Get a single document with ownership check."""
        doc = await db.get(Document, document_id)
        if not doc or str(doc.user_id) != user_id:
            return None
        return doc

    @staticmethod
    async def get_document_status(
        db: AsyncSession, document_id: uuid.UUID, user_id: str
    ) -> Optional[Document]:
        """Get document status (lightweight query for polling)."""
        doc = await db.get(Document, document_id)
        if not doc or str(doc.user_id) != user_id:
            return None
        return doc

    @staticmethod
    async def delete_document(
        db: AsyncSession, document_id: uuid.UUID, user_id: str
    ) -> bool:
        """Delete document, its file, and cascade to clauses."""
        doc = await db.get(Document, document_id)
        if not doc or str(doc.user_id) != user_id:
            return False
        # Remove file from disk
        file_path = Path(settings.UPLOAD_DIR) / doc.filename
        if file_path.exists():
            file_path.unlink()
        await db.delete(doc)
        await db.commit()
        logger.info(f"[db] Document deleted: {document_id}")
        return True
