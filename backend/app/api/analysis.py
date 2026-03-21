"""
Analysis API — Classification, Risk Analysis, Full Analysis, Results, Reports, NER, Chat.
Hardened: rate limiting, timeout-aware calls, proper error mapping.
"""

import uuid
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.core.rate_limiter import rate_limiter
from app.core.exceptions import (
    ClassificationError, InvalidDocumentStateError,
    RiskComputationError, MissingClassificationError,
    ExplainabilityError, ComplianceError, SummarizationError,
)
from app.models.models import Document, Clause, AnalysisResult
from app.services.analysis_service import AnalysisService
from app.services.report_generator import ReportGenerator
from app.schemas.analysis import (
    AnalysisTriggerResponse, AnalysisResultResponse,
    ClauseResponse, ChatRequest, ChatResponse,
    ClassificationResponse, RiskAnalysisResponse,
    ExplainabilityResponse, ComplianceReportResponse,
    PolicySummaryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])

TIMEOUT = settings.ML_INFERENCE_TIMEOUT_SECONDS


# ═══════════════════════════════════════════════════════════════
# MODULE 2: Classification Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/classify/{document_id}", response_model=ClassificationResponse)
async def classify_document(
    document_id: uuid.UUID,
    confidence_threshold: float = Query(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence to include a label (0.0-1.0)",
    ),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate=Depends(rate_limiter.limit("classify", settings.RATE_LIMIT_CLASSIFY, 60)),
):
    """Run multi-label classification. Rate limited: 10/min. Timeout: configurable."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = await asyncio.wait_for(
            AnalysisService.classify_document(document_id, confidence_threshold=confidence_threshold),
            timeout=TIMEOUT,
        )
        return ClassificationResponse(**result)
    except asyncio.TimeoutError:
        logger.error(f"[api] Classification timed out for {document_id} ({TIMEOUT}s)")
        # GPU cleanup on timeout
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        raise HTTPException(status_code=504, detail=f"Classification timed out after {TIMEOUT}s")
    except InvalidDocumentStateError as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {str(e)}")
    except ClassificationError as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# MODULE 3: Risk Analysis Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/risk/{document_id}", response_model=RiskAnalysisResponse)
async def analyze_risk(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate=Depends(rate_limiter.limit("risk", settings.RATE_LIMIT_RISK, 60)),
):
    """Run dynamic risk scoring. Rate limited: 10/min. Timeout: configurable."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = await asyncio.wait_for(
            AnalysisService.analyze_document_risk(document_id),
            timeout=TIMEOUT,
        )
        return RiskAnalysisResponse(**result)
    except asyncio.TimeoutError:
        logger.error(f"[api] Risk analysis timed out for {document_id} ({TIMEOUT}s)")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        raise HTTPException(status_code=504, detail=f"Risk analysis timed out after {TIMEOUT}s")
    except InvalidDocumentStateError as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {str(e)}")
    except MissingClassificationError as e:
        raise HTTPException(status_code=400, detail=f"Missing classification: {str(e)}")
    except RiskComputationError as e:
        raise HTTPException(status_code=500, detail=f"Risk computation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# MODULE 4: Explainability Endpoint
# ═══════════════════════════════════════════════════════════════

@router.get("/explain/{document_id}", response_model=ExplainabilityResponse)
async def explain_document(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate=Depends(rate_limiter.limit("explain", settings.RATE_LIMIT_EXPLAIN, 60)),
):
    """Get structured explanations. Rate limited: 20/min. Timeout: configurable."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = await asyncio.wait_for(
            AnalysisService.explain_document(document_id),
            timeout=TIMEOUT,
        )
        return ExplainabilityResponse(**result)
    except asyncio.TimeoutError:
        logger.error(f"[api] Explanation timed out for {document_id} ({TIMEOUT}s)")
        raise HTTPException(status_code=504, detail=f"Explanation timed out after {TIMEOUT}s")
    except InvalidDocumentStateError as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {str(e)}")
    except ExplainabilityError as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# MODULE 5: Compliance Mapping & Gap Detection
# ═══════════════════════════════════════════════════════════════

@router.post("/compliance/{document_id}", response_model=ComplianceReportResponse)
async def generate_compliance(
    document_id: uuid.UUID,
    framework: str = Query(
        default="GDPR",
        description="Regulatory framework: GDPR, CCPA",
    ),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate=Depends(rate_limiter.limit("compliance", 10, 60)),
):
    """
    Generate regulatory compliance report.

    Maps classified clauses to regulatory articles, detects gaps, computes coverage score.
    Requires status: ANALYZED. Rate limited: 10/min. Timeout: configurable.
    """
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = await asyncio.wait_for(
            AnalysisService.generate_compliance_report(
                document_id, framework=framework,
            ),
            timeout=TIMEOUT,
        )
        return ComplianceReportResponse(**result)
    except asyncio.TimeoutError:
        logger.error(f"[api] Compliance timed out for {document_id} ({TIMEOUT}s)")
        raise HTTPException(status_code=504, detail=f"Compliance analysis timed out after {TIMEOUT}s")
    except InvalidDocumentStateError as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {str(e)}")
    except ComplianceError as e:
        raise HTTPException(status_code=500, detail=f"Compliance failed: {str(e)}")


@router.get("/compliance/frameworks", tags=["Analysis"])
async def list_frameworks():
    """List available regulatory frameworks."""
    from app.core.regulatory_frameworks import get_available_frameworks
    return {"frameworks": get_available_frameworks()}


# ═══════════════════════════════════════════════════════════════
# MODULE 6: Policy Simplification & Executive Insights
# ═══════════════════════════════════════════════════════════════

@router.post("/summarize/{document_id}", response_model=PolicySummaryResponse)
async def summarize_document(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate=Depends(rate_limiter.limit("summarize", 10, 60)),
):
    """
    Generate plain-English policy summary and executive insights.

    Returns clause-level simplifications and executive-level analysis.
    Deterministic, template-based, no hallucination.
    Rate limited: 10/min. Timeout: configurable.
    """
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = await asyncio.wait_for(
            AnalysisService.generate_policy_summary(document_id),
            timeout=TIMEOUT,
        )
        return PolicySummaryResponse(**result)
    except asyncio.TimeoutError:
        logger.error(f"[api] Summary timed out for {document_id} ({TIMEOUT}s)")
        raise HTTPException(status_code=504, detail=f"Summary timed out after {TIMEOUT}s")
    except InvalidDocumentStateError as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {str(e)}")
    except SummarizationError as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")



# ═══════════════════════════════════════════════════════════════
# Full Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

@router.post("/analyze/{document_id}", response_model=AnalysisTriggerResponse)
async def trigger_analysis(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger full ML analysis pipeline (classify → risk → summarize → explain)."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status not in ("extracted", "processed", "classified", "analyzed"):
        raise HTTPException(
            status_code=400,
            detail=f"Document must be processed first. Current status: {doc.status}",
        )

    background_tasks.add_task(AnalysisService.run_pipeline, document_id)

    return AnalysisTriggerResponse(
        document_id=document_id, status="processing",
        message="Analysis pipeline started. This may take 1-3 minutes.",
    )


# ═══════════════════════════════════════════════════════════════
# Unified Pipeline (returns structured result)
# ═══════════════════════════════════════════════════════════════

@router.post("/run/{document_id}")
async def run_full_analysis(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run full analysis pipeline and return structured results.

    Executes all 10 steps: extract → classify → risk → explain →
    summarize → compliance (GDPR+CCPA) → embed → store → return.
    """
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status not in ("extracted", "processed", "classified", "analyzed"):
        raise HTTPException(
            status_code=400,
            detail=f"Document must be processed first. Current status: {doc.status}",
        )

    try:
        result = await asyncio.wait_for(
            AnalysisService.run_full_pipeline(document_id),
            timeout=300,  # 5 minutes for full pipeline
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"[api] Full pipeline timed out for {document_id}")
        raise HTTPException(status_code=504, detail="Pipeline timed out after 5 minutes")
    except Exception as e:
        logger.error(f"[api] Full pipeline failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# Results & Query Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/results/{document_id}", response_model=AnalysisResultResponse)
async def get_analysis_results(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get analysis results with clauses and risk scores."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.document_id == document_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis results. Run analysis first.")

    clauses_result = await db.execute(
        select(Clause).where(Clause.document_id == document_id).order_by(Clause.clause_index)
    )
    clauses = clauses_result.scalars().all()

    return AnalysisResultResponse(
        id=analysis.id, document_id=analysis.document_id,
        summary=analysis.summary, overall_risk=analysis.overall_risk,
        overall_score=analysis.overall_score, recommendations=analysis.recommendations,
        category_breakdown=analysis.category_breakdown,
        clauses=[ClauseResponse.model_validate(c) for c in clauses],
        created_at=analysis.created_at,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_document(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
):
    """Ask questions about documents using RAG. Supports document-scoped retrieval."""
    from app.ml.rag import get_rag_service
    try:
        rag = get_rag_service()
        result = rag.answer_question(
            question=request.query,
            document_id=request.document_id,
        )
        return ChatResponse(
            answer=result.get("answer", "I could not find relevant information."),
            sources=[
                d.get("content", "")[:300] if isinstance(d, dict) else str(d)[:300]
                for d in result.get("source_documents", [])
            ],
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# Compliance & Reports
# ═══════════════════════════════════════════════════════════════

@router.get("/compliance/{document_id}")
async def get_compliance_report(
    document_id: uuid.UUID,
    framework: str = "GDPR",
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate compliance report (GDPR or CCPA)."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    clauses_result = await db.execute(
        select(Clause).where(Clause.document_id == document_id).order_by(Clause.clause_index)
    )
    clauses = clauses_result.scalars().all()
    if not clauses:
        raise HTTPException(status_code=404, detail="No clauses found. Run analysis first.")

    clause_dicts = [
        {"text": c.clause_text, "category": c.category,
         "risk_score": c.risk_score, "risk_level": c.risk_level}
        for c in clauses
    ]
    return ReportGenerator.generate_compliance_report(clause_dicts, framework)


@router.get("/report/{document_id}")
async def get_risk_report(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate risk assessment report."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    analysis = (await db.execute(
        select(AnalysisResult).where(AnalysisResult.document_id == document_id)
    )).scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="No analysis found.")

    clauses = (await db.execute(
        select(Clause).where(Clause.document_id == document_id).order_by(Clause.clause_index)
    )).scalars().all()

    clause_dicts = [
        {"text": c.clause_text, "category": c.category, "risk_score": c.risk_score,
         "risk_level": c.risk_level, "explanation": c.explanation}
        for c in clauses
    ]
    return ReportGenerator.generate_risk_report(
        clause_dicts, analysis.summary or "", analysis.overall_risk or "low", analysis.overall_score or 0,
    )


@router.get("/entities/{document_id}")
async def get_entities(
    document_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Extract named entities from document text."""
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.extracted_text:
        raise HTTPException(status_code=400, detail="Document text not extracted yet.")

    from app.ml.ner import extract_entities
    entities = extract_entities(doc.extracted_text)
    return {"document_id": str(document_id), "entities": entities}
