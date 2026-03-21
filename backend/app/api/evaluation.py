"""
Evaluation API — Research evaluation endpoints.
Module 7: POST /run, GET /{id}, GET /datasets.
Rate limited. Isolated from production flow.
"""

import uuid
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.core.rate_limiter import rate_limiter
from app.services.evaluation_service import EvaluationService
from app.schemas.evaluation import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationListItem,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

TIMEOUT = settings.ML_INFERENCE_TIMEOUT_SECONDS


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/run", response_model=EvaluationRunResponse)
async def run_evaluation(
    request: EvaluationRunRequest,
    user_id: str = Depends(get_current_user),
    _rate=Depends(rate_limiter.limit("evaluation_run", 5, 60)),
):
    """
    Run a full evaluation pipeline on a labeled test dataset.

    Accepts labeled clauses with ground truth labels and risk scores.
    Returns classification metrics, risk metrics, baseline comparison,
    ablation study, statistical tests, and a Markdown research report.

    Rate limited: 5/min. Timeout: configurable.
    Does NOT modify production data.
    """
    if len(request.clauses) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 clauses per evaluation run",
        )

    clause_dicts = [c.model_dump() for c in request.clauses]
    compliance_gt = [c.model_dump() for c in request.compliance_ground_truth]

    try:
        result = await asyncio.wait_for(
            EvaluationService.run_evaluation(
                dataset_name=request.dataset_name,
                clauses=clause_dicts,
                compliance_ground_truth=compliance_gt if compliance_gt else None,
                framework=request.framework,
                run_baseline=request.run_baseline,
                run_ablation=request.run_ablation,
                run_statistical_tests=request.run_statistical_tests,
            ),
            timeout=TIMEOUT,
        )
        return EvaluationRunResponse(**result)
    except asyncio.TimeoutError:
        logger.error(f"[api] Evaluation timed out ({TIMEOUT}s)")
        raise HTTPException(
            status_code=504,
            detail=f"Evaluation timed out after {TIMEOUT}s",
        )
    except Exception as e:
        logger.error(f"[api] Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/{evaluation_id}", response_model=EvaluationRunResponse)
async def get_evaluation(
    evaluation_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    _rate=Depends(rate_limiter.limit("evaluation_get", 30, 60)),
):
    """
    Retrieve a stored evaluation run by ID.
    Rate limited: 30/min.
    """
    result = await EvaluationService.get_evaluation(evaluation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return EvaluationRunResponse(**result)


@router.get("/", response_model=list[EvaluationListItem])
async def list_evaluations(
    limit: int = Query(default=20, le=100, ge=1),
    user_id: str = Depends(get_current_user),
    _rate=Depends(rate_limiter.limit("evaluation_list", 30, 60)),
):
    """
    List recent evaluation runs.
    Rate limited: 30/min.
    """
    return await EvaluationService.list_evaluations(limit=limit)


@router.get("/datasets/available", tags=["Evaluation"])
async def list_available_datasets(
    user_id: str = Depends(get_current_user),
):
    """
    List available evaluation dataset formats.

    Returns the expected input format for evaluation runs.
    """
    return {
        "description": "Submit a labeled dataset via POST /evaluation/run",
        "required_format": {
            "dataset_name": "string — unique name for this evaluation",
            "clauses": [
                {
                    "clause_id": "string — unique clause identifier",
                    "clause_text": "string — the clause text",
                    "true_labels": ["list of ground truth labels"],
                    "true_risk_score": "float 0.0-1.0 (optional)",
                    "true_risk_level": "low|medium|high|critical (optional)",
                }
            ],
            "compliance_ground_truth": [
                {
                    "article_id": "Article 6",
                    "status": "satisfied|partial|missing",
                }
            ],
            "framework": "GDPR or CCPA (default: GDPR)",
            "run_baseline": True,
            "run_ablation": True,
            "run_statistical_tests": True,
        },
        "example_labels": [
            "DATA_COLLECTION", "DATA_SHARING", "USER_RIGHTS",
            "DATA_RETENTION", "SECURITY_MEASURES", "THIRD_PARTY_TRANSFER",
            "COOKIES_TRACKING", "CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE",
            "LIABILITY_LIMITATION", "CONSENT",
        ],
    }
