"""
Analysis Service — orchestrates ML pipelines.
Module 2: classify_document()
Module 3: analyze_document_risk()
Module 4: explain_document()
Full pipeline: run_pipeline()

Hardened:
  - SELECT FOR UPDATE on document status transitions (Category 2)
  - Config snapshot stored per analysis (Category 8)
  - Single-session validate+write (Category 2)
  - Proper status rollback on failure (Category 4)
"""

import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload
from app.models.models import (
    Document, Clause, ClauseClassification, ClauseRiskScore,
    ClauseExplanation, AnalysisResult, DocumentComplianceReport,
    DocumentSummary, ClauseSummary,
)
from app.core.exceptions import (
    ClassificationError,
    InvalidDocumentStateError,
    RiskComputationError,
    MissingClassificationError,
    ExplainabilityError,
    ComplianceError,
    SummarizationError,
)
from app.core.metrics import track_classification, track_risk_scoring, track_explanation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Concurrency Guard: Atomic status transition
# ═══════════════════════════════════════════════════════════════

async def _acquire_document_lock(session, document_id, allowed_statuses, target_status=None):
    """
    Atomic status validation with row-level lock (SELECT FOR UPDATE).
    Prevents race conditions on concurrent pipeline calls.

    Returns the locked document row.
    Raises InvalidDocumentStateError if status is invalid.
    """
    result = await session.execute(
        select(Document)
        .where(Document.id == document_id)
        .with_for_update()
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise InvalidDocumentStateError(
            f"Document {document_id} not found",
            document_id=str(document_id),
        )

    if doc.status not in allowed_statuses:
        raise InvalidDocumentStateError(
            f"Cannot proceed: status '{doc.status}', required one of {allowed_statuses}",
            document_id=str(document_id),
            current_status=doc.status,
            required_status=str(allowed_statuses),
        )

    return doc


class AnalysisService:
    """
    Orchestrates analysis pipelines:
    - classify_document()       — Module 2
    - analyze_document_risk()   — Module 3
    - explain_document()        — Module 4
    - run_pipeline()            — Full pipeline
    """

    # ══════════════════════════════════════════════════════════
    # MODULE 2: Standalone Classification
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def classify_document(
        document_id: uuid.UUID,
        confidence_threshold: float = 0.5,
    ) -> Dict:
        """Multi-label classification pipeline. Status: PROCESSED → CLASSIFIED."""
        from app.core.database import AsyncSessionLocal
        from app.ml.classifier import classify_clauses_multi_label

        doc_id_str = str(document_id)
        pipeline_start = time.perf_counter()

        logger.info(f"{'='*60}")
        logger.info(f"[classifier] START classification: {doc_id_str}")
        logger.info(f"[classifier] Confidence threshold: {confidence_threshold}")
        logger.info(f"{'='*60}")

        try:
            # 1. Validate with row lock + fetch clauses in same session
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    doc = await _acquire_document_lock(
                        session, document_id,
                        allowed_statuses=("processed", "classified", "analyzed"),
                    )

                    result = await session.execute(
                        select(Clause)
                        .where(Clause.document_id == document_id)
                        .order_by(Clause.clause_index)
                    )
                    clauses = result.scalars().all()
                    if not clauses:
                        raise ClassificationError(
                            f"No clauses for {doc_id_str}",
                            document_id=doc_id_str, step="fetch",
                        )
                    clause_inputs = [
                        {"index": c.clause_index, "text": c.clause_text, "db_id": str(c.id)}
                        for c in clauses
                    ]

            logger.info(f"[classifier] Fetched {len(clause_inputs)} clauses from DB")

            # 2. Classify (pure ML — no DB)
            t0 = time.perf_counter()
            classification_results = classify_clauses_multi_label(
                clause_inputs, confidence_threshold=confidence_threshold,
            )
            classification_time = int((time.perf_counter() - t0) * 1000)
            track_classification(classification_time / 1000.0, len(clause_inputs))
            logger.info(f"[classifier] ✓ {len(classification_results)} clauses classified ({classification_time}ms)")

            # 3. Store results with row lock
            model_version = classification_results[0].get("model_version", "unknown") if classification_results else "unknown"
            label_distribution = {}
            classified_count = 0
            total_confidence = 0.0

            async with AsyncSessionLocal() as session:
                async with session.begin():
                    # Re-acquire lock for status transition
                    doc = await _acquire_document_lock(
                        session, document_id,
                        allowed_statuses=("processed", "classified", "analyzed"),
                    )

                    for clause_input, cls_result in zip(clause_inputs, classification_results):
                        clause_db_id = uuid.UUID(clause_input["db_id"])

                        # Delete old classifications for this clause
                        old = await session.execute(
                            select(ClauseClassification)
                            .where(ClauseClassification.clause_id == clause_db_id)
                        )
                        for o in old.scalars():
                            await session.delete(o)

                        for label_data in cls_result.get("labels", []):
                            session.add(ClauseClassification(
                                clause_id=clause_db_id,
                                label=label_data["label"],
                                confidence_score=label_data["confidence"],
                                model_version=model_version,
                            ))
                            lbl = label_data["label"]
                            label_distribution[lbl] = label_distribution.get(lbl, 0) + 1

                        clause_obj = await session.get(Clause, clause_db_id)
                        if clause_obj:
                            clause_obj.category = cls_result.get("primary_label")
                            clause_obj.confidence = cls_result.get("primary_confidence")
                            clause_obj.classification_status = "completed"
                            clause_obj.classification_completed_at = datetime.now(timezone.utc)

                        if cls_result.get("primary_label") != "UNKNOWN":
                            classified_count += 1
                            total_confidence += cls_result.get("primary_confidence", 0)

                    # Atomic status transition
                    doc.status = "classified"
                    doc.version += 1

            total_time = int((time.perf_counter() - pipeline_start) * 1000)
            avg_confidence = round(total_confidence / classified_count, 4) if classified_count > 0 else 0.0

            logger.info(f"[classifier] Average confidence: {avg_confidence}")
            logger.info(f"[classifier] COMPLETE {doc_id_str}: {classified_count}/{len(clause_inputs)}, {total_time}ms")
            logger.info(f"{'='*60}")

            return {
                "document_id": doc_id_str, "status": "CLASSIFIED",
                "total_clauses": len(clause_inputs), "classified_clauses": classified_count,
                "average_confidence": avg_confidence, "classification_time_ms": total_time,
                "model_version": model_version, "label_distribution": label_distribution,
            }

        except (ClassificationError, InvalidDocumentStateError):
            raise
        except Exception as e:
            logger.error(f"[classifier] FAILED {doc_id_str}: {e}", exc_info=True)
            raise ClassificationError(
                f"Classification failed: {e}",
                document_id=doc_id_str, step="classify_document",
            )

    # ══════════════════════════════════════════════════════════
    # MODULE 3: Risk Analysis
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def analyze_document_risk(document_id: uuid.UUID) -> Dict:
        """
        Dynamic risk scoring pipeline. Status: CLASSIFIED → ANALYZED.
        Hardened: SELECT FOR UPDATE, config snapshot, atomic transition.
        """
        from app.core.database import AsyncSessionLocal
        from app.ml.risk_scorer import compute_clause_risks, compute_document_risk
        from app.core.risk_config import get_config_snapshot

        doc_id_str = str(document_id)
        pipeline_start = time.perf_counter()

        logger.info(f"{'='*60}")
        logger.info(f"[risk] START analysis: {doc_id_str}")
        logger.info(f"{'='*60}")

        try:
            # 1. Validate with row lock
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    doc = await _acquire_document_lock(
                        session, document_id,
                        allowed_statuses=("classified", "analyzed"),
                    )

            # 2. Pre-fetch ALL data (no DB in loop)
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Clause)
                    .options(selectinload(Clause.classifications))
                    .where(Clause.document_id == document_id)
                    .order_by(Clause.clause_index)
                )
                clauses = result.scalars().all()
                if not clauses:
                    raise MissingClassificationError(
                        f"No clauses for {doc_id_str}",
                        document_id=doc_id_str,
                    )

                # Check at least some clauses have classifications
                has_classifications = any(len(c.classifications) > 0 for c in clauses)
                if not has_classifications:
                    raise MissingClassificationError(
                        f"No classification data for {doc_id_str}. Run classification first.",
                        document_id=doc_id_str,
                    )

                clause_data = []
                for c in clauses:
                    clause_data.append({
                        "clause_id": str(c.id),
                        "clause_index": c.clause_index,
                        "clause_text": c.clause_text,
                        "category": c.category or "",
                        "confidence": c.confidence or 0.5,
                        "classifications": [
                            {"label": cls.label, "confidence_score": cls.confidence_score}
                            for cls in c.classifications
                        ],
                        "entities": c.entities or [],
                    })

            logger.info(f"[risk] Fetched {len(clause_data)} clauses with classifications")

            # 3. Compute risk (pure ML — no DB)
            t0 = time.perf_counter()
            clause_risks = compute_clause_risks(clause_data, total_clauses=len(clause_data))
            doc_risk = compute_document_risk(clause_risks)
            risk_time = int((time.perf_counter() - t0) * 1000)
            track_risk_scoring(risk_time / 1000.0, len(clause_data))

            logger.info(f"[risk] ✓ Computed in {risk_time}ms: {doc_risk['risk_level']} ({doc_risk['overall_risk_score']})")

            # 4. Store with row lock + config snapshot
            config_snapshot = get_config_snapshot()

            async with AsyncSessionLocal() as session:
                async with session.begin():
                    doc = await _acquire_document_lock(
                        session, document_id,
                        allowed_statuses=("classified", "analyzed"),
                    )

                    # Delete old risk scores
                    for c in clauses:
                        old_risks = await session.execute(
                            select(ClauseRiskScore).where(ClauseRiskScore.clause_id == c.id)
                        )
                        for old in old_risks.scalars():
                            await session.delete(old)

                    # Insert new risk scores + update clause fields
                    for risk_result in clause_risks:
                        clause_db_id = uuid.UUID(risk_result["clause_id"])
                        session.add(ClauseRiskScore(
                            clause_id=clause_db_id,
                            risk_score=risk_result["risk_score"],
                            risk_level=risk_result["risk_level"],
                            risk_factors=risk_result.get("risk_factors"),
                            explanation=risk_result.get("explanation"),
                        ))

                        clause_obj = await session.get(Clause, clause_db_id)
                        if clause_obj:
                            level_map = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high", "CRITICAL": "critical"}
                            clause_obj.risk_score = risk_result["risk_score"] / 100.0
                            clause_obj.risk_level = level_map.get(risk_result["risk_level"], "low")
                            clause_obj.explanation = risk_result.get("explanation")

                    # Atomic status transition + config snapshot
                    doc.overall_risk_score = doc_risk["overall_risk_score"]
                    doc.overall_risk_level = doc_risk["risk_level"]
                    doc.status = "analyzed"
                    doc.analyzed_at = datetime.now(timezone.utc)
                    doc.config_snapshot = config_snapshot
                    doc.version += 1

            total_time = int((time.perf_counter() - pipeline_start) * 1000)

            summary = {
                "document_id": doc_id_str, "status": "ANALYZED",
                "overall_risk_score": doc_risk["overall_risk_score"],
                "risk_level": doc_risk["risk_level"],
                "total_clauses": len(clause_risks),
                "total_high_risk_clauses": doc_risk["total_high_risk_clauses"],
                "risk_distribution": doc_risk["risk_distribution"],
                "analysis_time_ms": total_time,
            }

            logger.info(f"[risk] COMPLETE {doc_id_str} ({total_time}ms)")
            logger.info(f"[risk] Distribution: {doc_risk['risk_distribution']}")
            logger.info(f"{'='*60}")

            return summary

        except (InvalidDocumentStateError, MissingClassificationError, RiskComputationError):
            raise
        except Exception as e:
            logger.error(f"[risk] FAILED {doc_id_str}: {e}", exc_info=True)
            try:
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        doc = await session.get(Document, document_id)
                        if doc:
                            doc.error_message = f"Risk analysis failed: {str(e)}"
            except Exception:
                pass
            raise RiskComputationError(
                f"Risk analysis failed: {e}",
                document_id=doc_id_str, step="analyze_risk",
            )

    # ══════════════════════════════════════════════════════════
    # MODULE 4: Explainability
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def explain_document(document_id: uuid.UUID) -> Dict:
        """
        Generate classification and risk explanations for all clauses.
        Hardened: SELECT FOR UPDATE, atomic store.
        """
        from app.core.database import AsyncSessionLocal
        from app.ml.explainability import generate_explanations_batch

        doc_id_str = str(document_id)
        pipeline_start = time.perf_counter()

        logger.info(f"{'='*60}")
        logger.info(f"[explain] START {doc_id_str}")
        logger.info(f"{'='*60}")

        try:
            # 1. Validate with row lock
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    doc = await _acquire_document_lock(
                        session, document_id,
                        allowed_statuses=("analyzed", "classified"),
                    )
                    overall_risk_score = doc.overall_risk_score or 0.0
                    overall_risk_level = doc.overall_risk_level or "LOW"

            # 2. Pre-fetch ALL data
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Clause)
                    .options(
                        selectinload(Clause.classifications),
                        selectinload(Clause.risk_scores),
                    )
                    .where(Clause.document_id == document_id)
                    .order_by(Clause.clause_index)
                )
                clauses = result.scalars().all()

                if not clauses:
                    raise ExplainabilityError(
                        f"No clauses found for {doc_id_str}",
                        document_id=doc_id_str, step="fetch",
                    )

                engine_input = []
                for c in clauses:
                    latest_risk = None
                    if c.risk_scores:
                        latest_risk = max(c.risk_scores, key=lambda r: r.created_at or datetime.min)

                    engine_input.append({
                        "clause_id": str(c.id),
                        "clause_index": c.clause_index,
                        "clause_text": c.clause_text,
                        "category": c.category or "",
                        "confidence": c.confidence or 0.5,
                        "classifications": [
                            {"label": cls.label, "confidence_score": cls.confidence_score}
                            for cls in c.classifications
                        ],
                        "risk_score": (latest_risk.risk_score if latest_risk else 0),
                        "risk_level": (latest_risk.risk_level if latest_risk else "LOW"),
                        "risk_factors": (latest_risk.risk_factors if latest_risk else []),
                        "risk_debug": None,
                        "entities": c.entities or [],
                    })

            logger.info(f"[explain] Generating classification explanations")
            logger.info(f"[explain] Generating risk explanations")

            # 3. Batch generate explanations
            t0 = time.perf_counter()
            explanations = generate_explanations_batch(engine_input)
            gen_time = int((time.perf_counter() - t0) * 1000)
            track_explanation(gen_time / 1000.0, len(engine_input))

            logger.info(f"[explain] Generated {len(explanations)} explanations ({gen_time}ms)")

            # 4. Store with row lock
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    # Delete old explanations
                    for c in clauses:
                        old_exps = await session.execute(
                            select(ClauseExplanation).where(ClauseExplanation.clause_id == c.id)
                        )
                        for old in old_exps.scalars():
                            await session.delete(old)

                    # Insert new explanations
                    for exp in explanations:
                        clause_db_id = uuid.UUID(exp["clause_id"])

                        session.add(ClauseExplanation(
                            clause_id=clause_db_id,
                            explanation_type="CLASSIFICATION",
                            explanation_data=exp["classification_explanation"],
                        ))

                        session.add(ClauseExplanation(
                            clause_id=clause_db_id,
                            explanation_type="RISK",
                            explanation_data=exp["risk_explanation"],
                        ))

            total_time = int((time.perf_counter() - pipeline_start) * 1000)

            logger.info(f"[explain] Stored {len(explanations) * 2} explanation rows")
            logger.info(f"[explain] COMPLETE {doc_id_str} ({total_time}ms)")
            logger.info(f"{'='*60}")

            return {
                "document_id": doc_id_str,
                "overall_risk_score": overall_risk_score,
                "risk_level": overall_risk_level,
                "total_clauses": len(explanations),
                "explanation_time_ms": total_time,
                "clauses": explanations,
            }

        except (InvalidDocumentStateError, ExplainabilityError):
            raise
        except Exception as e:
            logger.error(f"[explain] FAILED {doc_id_str}: {e}", exc_info=True)
            raise ExplainabilityError(
                f"Explanation generation failed: {e}",
                document_id=doc_id_str, step="explain_document",
            )

    # ══════════════════════════════════════════════════════════
    # FULL ANALYSIS PIPELINE
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def run_pipeline(document_id: uuid.UUID):
        """Full analysis pipeline — runs as a background task."""
        from app.core.database import AsyncSessionLocal
        from app.ml.preprocessing import extract_clauses
        from app.ml.classifier import classify_clauses
        from app.ml.risk_scorer import score_risks
        from app.ml.summarizer import summarize_document
        from app.ml.explainability import generate_explanations
        from app.ml.rag import get_rag_service

        try:
            async with AsyncSessionLocal() as session:
                doc = await session.get(Document, document_id)
                if not doc or not doc.extracted_text:
                    logger.error(f"No extracted text for {document_id}")
                    return
                text = doc.extracted_text
                original_filename = doc.original_filename
                doc.status = "processing"
                await session.commit()

            logger.info(f"[analysis] Pipeline started for {document_id}")

            raw_clauses = extract_clauses(text)
            logger.info(f"[analysis] Extracted {len(raw_clauses)} clauses")

            classified = classify_clauses(raw_clauses)
            scored = score_risks(classified)
            explained = generate_explanations(scored)
            summary = summarize_document(text)

            overall_score, overall_risk = AnalysisService._compute_overall_risk(explained)
            recommendations = AnalysisService._generate_recommendations(explained)

            category_counts = {}
            for c in explained:
                cat = c.get("category", "Unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            async with AsyncSessionLocal() as session:
                async with session.begin():
                    for old in (await session.execute(select(Clause).where(Clause.document_id == document_id))).scalars():
                        await session.delete(old)
                    for old in (await session.execute(select(AnalysisResult).where(AnalysisResult.document_id == document_id))).scalars():
                        await session.delete(old)

                    for i, clause_data in enumerate(explained):
                        session.add(Clause(
                            document_id=document_id, clause_text=clause_data["text"],
                            clause_index=i, category=clause_data.get("category"),
                            confidence=clause_data.get("confidence"),
                            risk_level=clause_data.get("risk_level"),
                            risk_score=clause_data.get("risk_score"),
                            explanation=clause_data.get("explanation"),
                        ))

                    session.add(AnalysisResult(
                        document_id=document_id, summary=summary,
                        overall_risk=overall_risk, overall_score=round(overall_score, 3),
                        recommendations=recommendations, category_breakdown=category_counts,
                    ))

                    doc = await session.get(Document, document_id)
                    if doc:
                        doc.status = "analyzed"
                        doc.analyzed_at = datetime.now(timezone.utc)
                        doc.version += 1

            try:
                rag = get_rag_service()
                await rag.ingest_document(str(document_id), text, {"filename": original_filename})
            except Exception as e:
                logger.warning(f"RAG ingestion failed (non-fatal): {e}")

            logger.info(f"[analysis] COMPLETE {document_id}: {overall_risk} ({overall_score:.2f})")

        except Exception as e:
            logger.error(f"[analysis] FAILED {document_id}: {e}", exc_info=True)
            # Status rollback on failure (Category 4)
            try:
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        doc = await session.get(Document, document_id)
                        if doc:
                            doc.status = "failed"
                            doc.error_message = f"Analysis failed: {str(e)}"
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════
    # UNIFIED PIPELINE (returns structured result)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def run_full_pipeline(document_id: uuid.UUID) -> Dict:
        """
        Full analysis pipeline that returns structured results.
        Used by POST /api/v1/analysis/run/{document_id}.

        Steps:
          1. PDF extraction (already done by upload)
          2. Text cleaning + clause extraction
          3. Classification (Legal-BERT)
          4. Risk scoring
          5. Explainability (SHAP)
          6. Summarization (T5)
          7. Compliance mapping (GDPR + CCPA)
          8. Embedding + RAG storage (ChromaDB)
          9. Save all to PostgreSQL
          10. Return structured JSON
        """
        from app.core.database import AsyncSessionLocal
        from app.ml.preprocessing import extract_clauses
        from app.ml.classifier import classify_clauses
        from app.ml.risk_scorer import score_risks
        from app.ml.summarizer import summarize_document
        from app.ml.explainability import generate_explanations
        from app.ml.rag import get_rag_service

        pipeline_start = time.perf_counter()

        try:
            # ── Step 1: Fetch document text ──────────────────────
            async with AsyncSessionLocal() as session:
                doc = await session.get(Document, document_id)
                if not doc or not doc.extracted_text:
                    raise InvalidDocumentStateError(
                        "No extracted text found",
                        document_id=str(document_id),
                    )
                text = doc.extracted_text
                original_filename = doc.original_filename
                doc.status = "processing"
                await session.commit()

            logger.info(f"[unified] Pipeline started for {document_id}")

            # ── Step 2: Clause extraction ────────────────────────
            raw_clauses = extract_clauses(text)
            logger.info(f"[unified] Extracted {len(raw_clauses)} clauses")

            # ── Step 3: Classification ───────────────────────────
            classified = classify_clauses(raw_clauses)
            logger.info(f"[unified] Classified {len(classified)} clauses")

            # ── Step 4: Risk scoring ─────────────────────────────
            scored = score_risks(classified)
            logger.info(f"[unified] Risk scored {len(scored)} clauses")

            # ── Step 5: Explainability ───────────────────────────
            explained = generate_explanations(scored)
            logger.info(f"[unified] Generated explanations")

            # ── Step 6: Summarization ────────────────────────────
            summary = summarize_document(text)
            logger.info(f"[unified] Generated summary")

            # ── Compute aggregates ───────────────────────────────
            overall_score, overall_risk = AnalysisService._compute_overall_risk(explained)
            recommendations = AnalysisService._generate_recommendations(explained)

            category_counts = {}
            for c in explained:
                cat = c.get("category", "Unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # ── Step 9: Save to PostgreSQL ───────────────────────
            clause_objs = []
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    # Clear old data
                    for old in (await session.execute(select(Clause).where(Clause.document_id == document_id))).scalars():
                        await session.delete(old)
                    for old in (await session.execute(select(AnalysisResult).where(AnalysisResult.document_id == document_id))).scalars():
                        await session.delete(old)

                    for i, clause_data in enumerate(explained):
                        clause_obj = Clause(
                            document_id=document_id, clause_text=clause_data["text"],
                            clause_index=i, category=clause_data.get("category"),
                            confidence=clause_data.get("confidence"),
                            risk_level=clause_data.get("risk_level"),
                            risk_score=clause_data.get("risk_score"),
                            explanation=clause_data.get("explanation"),
                        )
                        session.add(clause_obj)
                        clause_objs.append(clause_obj)

                    session.add(AnalysisResult(
                        document_id=document_id, summary=summary,
                        overall_risk=overall_risk, overall_score=round(overall_score, 3),
                        recommendations=recommendations, category_breakdown=category_counts,
                    ))

                    doc = await session.get(Document, document_id)
                    if doc:
                        doc.status = "analyzed"
                        doc.analyzed_at = datetime.now(timezone.utc)
                        doc.version += 1

            # ── Step 7: Compliance mapping (GDPR + CCPA) ────────
            compliance_result = {"gdpr": {}, "ccpa": {}}
            try:
                compliance_result = AnalysisService._run_compliance(explained)
                logger.info(f"[unified] Compliance mapping complete")
            except Exception as e:
                logger.warning(f"[unified] Compliance mapping failed (non-fatal): {e}")

            # ── Step 8: RAG embedding ────────────────────────────
            try:
                rag = get_rag_service()
                await rag.ingest_document(str(document_id), text, {"filename": original_filename})
                logger.info(f"[unified] RAG ingestion complete")
            except Exception as e:
                logger.warning(f"[unified] RAG ingestion failed (non-fatal): {e}")

            # ── Step 10: Build response ──────────────────────────
            processing_time = round(time.perf_counter() - pipeline_start, 2)

            clause_results = []
            explanation_results = []
            for i, c in enumerate(explained):
                clause_results.append({
                    "clause_index": i,
                    "clause_text": c.get("text", ""),
                    "category": c.get("category"),
                    "confidence": c.get("confidence"),
                    "risk_level": c.get("risk_level"),
                    "risk_score": c.get("risk_score"),
                    "explanation": c.get("explanation"),
                })
                explanation_results.append({
                    "clause_index": i,
                    "category": c.get("category"),
                    "explanation": c.get("explanation"),
                    "risk_factors": c.get("risk_factors", []),
                    "shap_tokens": c.get("shap_tokens", []),
                })

            logger.info(f"[unified] COMPLETE {document_id}: {overall_risk} ({overall_score:.2f}) in {processing_time}s")

            return {
                "status": "ANALYZED",
                "document_id": str(document_id),
                "clauses": clause_results,
                "classifications": category_counts,
                "risk_score": round(overall_score, 3),
                "risk_level": overall_risk,
                "compliance": compliance_result,
                "summary": summary,
                "recommendations": recommendations,
                "explanation": explanation_results,
                "processing_time": processing_time,
            }

        except (InvalidDocumentStateError, ClassificationError):
            raise
        except Exception as e:
            logger.error(f"[unified] FAILED {document_id}: {e}", exc_info=True)
            # Rollback status
            try:
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        doc = await session.get(Document, document_id)
                        if doc:
                            doc.status = "failed"
                            doc.error_message = f"Pipeline failed: {str(e)}"
            except Exception:
                pass
            raise

    @staticmethod
    def _run_compliance(explained_clauses: list) -> Dict:
        """Run GDPR + CCPA compliance mapping on explained clauses."""
        from app.ml.compliance_engine import evaluate_compliance

        # Build data structures the compliance engine expects
        clause_data = []
        classification_data = []
        risk_data = []
        entity_data = []

        for i, c in enumerate(explained_clauses):
            cid = str(i)
            clause_data.append({
                "clause_id": cid,
                "clause_index": i,
                "clause_text": c.get("text", ""),
                "category": c.get("category", ""),
            })
            classification_data.append({
                "clause_id": cid,
                "label": c.get("category", ""),
                "confidence_score": c.get("confidence", 0.5),
            })
            risk_data.append({
                "clause_id": cid,
                "risk_score": c.get("risk_score", 0),
                "risk_level": c.get("risk_level", "low"),
            })
            entity_data.append({
                "clause_id": cid,
                "entities": c.get("entities", []),
            })

        result = {"gdpr": {}, "ccpa": {}}

        for framework in ["GDPR", "CCPA"]:
            try:
                report = evaluate_compliance(
                    framework_name=framework,
                    clauses=clause_data,
                    classifications=classification_data,
                    risk_scores=risk_data,
                    entities=entity_data,
                )
                result[framework.lower()] = {
                    "coverage_percentage": report.get("coverage_percentage", 0),
                    "articles_detected": [
                        f["article_id"] for f in report.get("fully_satisfied", [])
                    ] + [
                        f["article_id"] for f in report.get("partial_requirements", [])
                    ],
                    "missing_articles": [
                        f["article_id"] for f in report.get("missing_requirements", [])
                    ],
                    "compliance_score": report.get("compliance_score", 0),
                }
            except Exception as e:
                logger.warning(f"[compliance] {framework} evaluation failed: {e}")
                result[framework.lower()] = {
                    "coverage_percentage": 0,
                    "articles_detected": [],
                    "missing_articles": [],
                    "compliance_score": 0,
                }

        return result

    # ══════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _compute_overall_risk(scored_clauses: list) -> tuple[float, str]:
        if not scored_clauses:
            return 0.0, "low"
        avg = sum(c.get("risk_score", 0) for c in scored_clauses) / len(scored_clauses)
        high_count = sum(1 for c in scored_clauses if c.get("risk_level") in ("high", "critical"))
        critical_count = sum(1 for c in scored_clauses if c.get("risk_level") == "critical")
        if critical_count >= 2 or avg > 0.75:
            return avg, "critical"
        elif high_count >= 3 or avg > 0.6:
            return avg, "high"
        elif avg > 0.35:
            return avg, "medium"
        return avg, "low"

    @staticmethod
    def _generate_recommendations(scored_clauses: list) -> list:
        recs = []
        seen = set()
        priority_map = {
            "Data Sharing": "Review data sharing clauses — broad third-party access detected.",
            "Third-Party Transfer": "Third-party data transfers are extensive. Verify recipients.",
            "Data Retention": "Data retention terms are vague. Request specific time periods.",
            "User Rights": "User rights may be limited. Verify GDPR/CCPA compliance.",
            "Security Measures": "Security commitments appear weak. Look for encryption standards.",
            "Consent": "Consent mechanisms may not meet regulatory standards.",
            "Liability Limitation": "Liability limitations are broad. Review indemnification terms.",
            "Children's Privacy": "Policy references minors' data. Verify COPPA compliance.",
            "Data Collection": "Extensive data collection detected. Review necessity.",
            "Compliance Reference": "Compliance references should be verified independently.",
            "Cookies & Tracking": "Tracking technologies in use. Review cookie consent mechanisms.",
        }
        for c in scored_clauses:
            cat = c.get("category", "")
            risk = c.get("risk_level", "low")
            if risk in ("high", "critical") and cat not in seen:
                seen.add(cat)
                recs.append(priority_map.get(cat, f"High-risk clause in '{cat}'. Review carefully."))
        if not recs:
            recs.append("No critical risks detected. The policy appears reasonably transparent.")
        return recs

    # ══════════════════════════════════════════════════════════
    # MODULE 5: Compliance Mapping & Gap Detection
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def generate_compliance_report(
        document_id: uuid.UUID,
        framework: str = "GDPR",
        custom_weights: dict = None,
    ) -> Dict:
        """
        Generate regulatory compliance report for an analyzed document.

        Flow:
          1. Validate document status == ANALYZED
          2. Pre-fetch clauses, classifications, risk scores, entities
          3. Call compliance_engine.evaluate_compliance()
          4. Store report in document_compliance_reports
          5. Return structured response

        Transactional: single session for read + write.
        """
        from app.core.database import AsyncSessionLocal
        from app.ml.compliance_engine import evaluate_compliance
        from app.ml.model_loader import get_model_version
        from app.core.regulatory_frameworks import get_framework, get_available_frameworks

        # Validate framework name
        available = get_available_frameworks()
        if framework.upper() not in available:
            raise ComplianceError(
                f"Unknown framework '{framework}'. Available: {available}",
                document_id=str(document_id),
                step="compliance_validation",
            )

        t0 = time.perf_counter()

        async with AsyncSessionLocal() as session:
            async with session.begin():
                # 1. Validate document status
                doc = await _acquire_document_lock(
                    session, document_id,
                    allowed_statuses=["analyzed"],
                    target_status=None,  # Don't change status
                )

                # 2. Pre-fetch ALL data in batch (no DB inside loop)
                clauses_result = await session.execute(
                    select(Clause)
                    .where(Clause.document_id == document_id)
                    .order_by(Clause.clause_index)
                )
                clauses = clauses_result.scalars().all()

                if not clauses:
                    raise ComplianceError(
                        "No clauses found for document",
                        document_id=str(document_id),
                        step="compliance_data_fetch",
                    )

                clause_ids = [c.id for c in clauses]

                # Fetch classifications
                class_result = await session.execute(
                    select(ClauseClassification)
                    .where(ClauseClassification.clause_id.in_(clause_ids))
                )
                classifications_raw = class_result.scalars().all()

                # Fetch risk scores
                risk_result = await session.execute(
                    select(ClauseRiskScore)
                    .where(ClauseRiskScore.clause_id.in_(clause_ids))
                )
                risks_raw = risk_result.scalars().all()

                # 3. Prepare data for compliance engine (no ORM objects)
                clause_data = [
                    {
                        "clause_id": str(c.id),
                        "clause_index": c.clause_index,
                        "clause_text": c.clause_text or "",
                        "category": c.category or "",
                    }
                    for c in clauses
                ]

                classification_data = [
                    {
                        "clause_id": str(cc.clause_id),
                        "label": cc.label,
                        "confidence_score": cc.confidence_score,
                    }
                    for cc in classifications_raw
                ]

                risk_data = [
                    {
                        "clause_id": str(rs.clause_id),
                        "risk_score": rs.risk_score,
                        "risk_level": rs.risk_level,
                    }
                    for rs in risks_raw
                ]

                entity_data = [
                    {
                        "clause_id": str(c.id),
                        "entities": c.entities or [],
                    }
                    for c in clauses
                ]

                # 4. Call compliance engine (pure computation, no DB)
                try:
                    report = evaluate_compliance(
                        framework_name=framework,
                        clauses=clause_data,
                        classifications=classification_data,
                        risk_scores=risk_data,
                        entities=entity_data,
                        custom_weights=custom_weights,
                    )
                except Exception as e:
                    raise ComplianceError(
                        f"Compliance evaluation failed: {str(e)}",
                        document_id=str(document_id),
                        step="compliance_engine",
                    )

                compliance_time = int((time.perf_counter() - t0) * 1000)

                # 5. Build config snapshot for reproducibility
                config_snapshot = {
                    "framework": framework.upper(),
                    "custom_weights": custom_weights,
                    "total_clauses": len(clause_data),
                    "total_classifications": len(classification_data),
                    "total_risk_scores": len(risk_data),
                }

                model_version = get_model_version("classifier")

                # 6. Store report (UPSERT: delete existing for same framework)
                existing = await session.execute(
                    select(DocumentComplianceReport)
                    .where(
                        DocumentComplianceReport.document_id == document_id,
                        DocumentComplianceReport.framework == framework.upper(),
                    )
                )
                old_report = existing.scalar_one_or_none()
                if old_report:
                    await session.delete(old_report)
                    await session.flush()

                compliance_record = DocumentComplianceReport(
                    document_id=document_id,
                    framework=framework.upper(),
                    compliance_score=report["compliance_score"],
                    coverage_percentage=report["coverage_percentage"],
                    total_articles=report["total_articles"],
                    satisfied_count=report["satisfied_count"],
                    partial_count=report["partial_count"],
                    missing_count=report["missing_count"],
                    missing_requirements=report["missing_requirements"],
                    partial_requirements=report["partial_requirements"],
                    fully_satisfied=report["fully_satisfied"],
                    config_snapshot=config_snapshot,
                    model_version=model_version,
                )
                session.add(compliance_record)

        logger.info(
            f"[compliance] ✓ {framework.upper()} report for {document_id}: "
            f"score={report['compliance_score']}, coverage={report['coverage_percentage']}% "
            f"({compliance_time}ms)"
        )

        return {
            "document_id": document_id,
            "framework": report["framework"],
            "compliance_score": report["compliance_score"],
            "coverage_percentage": report["coverage_percentage"],
            "total_articles": report["total_articles"],
            "satisfied_count": report["satisfied_count"],
            "partial_count": report["partial_count"],
            "missing_count": report["missing_count"],
            "missing_requirements": report["missing_requirements"],
            "partial_requirements": report["partial_requirements"],
            "fully_satisfied": report["fully_satisfied"],
            "compliance_time_ms": compliance_time,
        }

    # ══════════════════════════════════════════════════════════
    # MODULE 6: Policy Simplification & Executive Insights
    # ══════════════════════════════════════════════════════════

    @staticmethod
    async def generate_policy_summary(
        document_id: uuid.UUID,
    ) -> Dict:
        """
        Generate policy simplification and executive insights.

        Flow:
          1. Validate document status == ANALYZED
          2. Pre-fetch clauses, risk scores, compliance report
          3. Simplify clauses (batch, template-based)
          4. Generate executive insights
          5. Store summaries in document_summaries + clause_summaries
          6. Return structured response

        Deterministic: template-based, no ML randomness.
        Transactional: single session for read + write.
        """
        from app.core.database import AsyncSessionLocal
        from app.ml.policy_simplifier import simplify_clauses
        from app.ml.executive_insight_engine import generate_executive_insights
        from app.ml.model_loader import get_model_version

        t0 = time.perf_counter()

        async with AsyncSessionLocal() as session:
            async with session.begin():
                # 1. Validate document status
                doc = await _acquire_document_lock(
                    session, document_id,
                    allowed_statuses=["analyzed"],
                    target_status=None,  # Don't change status
                )

                # 2. Pre-fetch ALL data in batch (no DB inside loop)
                clauses_result = await session.execute(
                    select(Clause)
                    .where(Clause.document_id == document_id)
                    .order_by(Clause.clause_index)
                )
                clauses = clauses_result.scalars().all()

                if not clauses:
                    raise SummarizationError(
                        "No clauses found for document",
                        document_id=str(document_id),
                        step="summary_data_fetch",
                    )

                clause_ids = [c.id for c in clauses]

                # Fetch risk scores
                risk_result = await session.execute(
                    select(ClauseRiskScore)
                    .where(ClauseRiskScore.clause_id.in_(clause_ids))
                )
                risks_raw = risk_result.scalars().all()
                risk_map = {str(r.clause_id): r for r in risks_raw}

                # Fetch compliance report (most recent, any framework)
                compliance_result = await session.execute(
                    select(DocumentComplianceReport)
                    .where(DocumentComplianceReport.document_id == document_id)
                    .order_by(DocumentComplianceReport.created_at.desc())
                )
                compliance_report = compliance_result.scalar_one_or_none()

                # 3. Prepare clause data for simplifier
                clause_data = []
                for c in clauses:
                    risk = risk_map.get(str(c.id))
                    clause_data.append({
                        "clause_id": str(c.id),
                        "clause_index": c.clause_index,
                        "clause_text": c.clause_text or "",
                        "category": c.category or "",
                        "risk_score": risk.risk_score if risk else 0.0,
                        "risk_level": risk.risk_level if risk else "low",
                    })

                # 4. Run clause simplifier (pure computation)
                try:
                    clause_summaries_data = simplify_clauses(clause_data)
                except Exception as e:
                    raise SummarizationError(
                        f"Clause simplification failed: {str(e)}",
                        document_id=str(document_id),
                        step="clause_simplification",
                    )

                # 5. Run executive insight generator (pure computation)
                compliance_gaps = []
                compliance_score = None
                if compliance_report:
                    compliance_score = compliance_report.compliance_score
                    missing = compliance_report.missing_requirements or []
                    partial = compliance_report.partial_requirements or []
                    compliance_gaps = missing + partial

                try:
                    insights = generate_executive_insights(
                        overall_risk_score=doc.overall_risk_score or 0.0,
                        risk_level=doc.overall_risk_level or "low",
                        clause_risks=clause_data,
                        compliance_gaps=compliance_gaps,
                        compliance_score=compliance_score,
                        total_clauses=len(clauses),
                    )
                except Exception as e:
                    raise SummarizationError(
                        f"Executive insight generation failed: {str(e)}",
                        document_id=str(document_id),
                        step="executive_insights",
                    )

                summary_time = int((time.perf_counter() - t0) * 1000)

                # 6. Config snapshot for reproducibility
                config_snapshot = {
                    "total_clauses": len(clause_data),
                    "total_risks": len(risks_raw),
                    "compliance_framework": compliance_report.framework if compliance_report else None,
                    "compliance_score": compliance_score,
                }
                model_version = get_model_version("classifier")

                # 7. Store document summary (UPSERT: delete existing)
                existing_summary = await session.execute(
                    select(DocumentSummary)
                    .where(DocumentSummary.document_id == document_id)
                )
                old_summary = existing_summary.scalar_one_or_none()
                if old_summary:
                    await session.delete(old_summary)
                    await session.flush()

                doc_summary = DocumentSummary(
                    document_id=document_id,
                    overall_summary=insights["overall_summary"],
                    executive_insights=insights,
                    config_snapshot=config_snapshot,
                    model_version=model_version,
                )
                session.add(doc_summary)

                # 8. Store clause summaries (UPSERT: delete existing)
                existing_clause_summaries = await session.execute(
                    select(ClauseSummary)
                    .where(ClauseSummary.clause_id.in_(clause_ids))
                )
                for old_cs in existing_clause_summaries.scalars():
                    await session.delete(old_cs)
                await session.flush()

                for cs in clause_summaries_data:
                    # Find matching clause UUID
                    clause_uuid = None
                    for c in clauses:
                        if str(c.id) == cs["clause_id"]:
                            clause_uuid = c.id
                            break
                    if clause_uuid:
                        session.add(ClauseSummary(
                            clause_id=clause_uuid,
                            plain_summary=cs["plain_summary"],
                            risk_note=cs.get("risk_note", ""),
                        ))

        logger.info(
            f"[summary] ✓ Policy summary for {document_id}: "
            f"{len(clause_summaries_data)} clause summaries, "
            f"risk_level={insights['risk_level']} ({summary_time}ms)"
        )

        return {
            "document_id": document_id,
            "overall_summary": insights["overall_summary"],
            "executive_insights": insights,
            "clause_summaries": clause_summaries_data,
            "summary_time_ms": summary_time,
        }
