"""
Evaluation Schemas — Pydantic models for Module 7.
"""

from uuid import UUID
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict


class EvaluationDatasetClause(BaseModel):
    """A labeled test clause in the evaluation dataset."""
    clause_id: str
    clause_text: str
    true_labels: List[str] = []
    true_risk_score: Optional[float] = None
    true_risk_level: Optional[str] = None


class EvaluationComplianceGT(BaseModel):
    """A labeled compliance ground truth entry."""
    article_id: str
    status: str  # "satisfied", "partial", "missing"


class EvaluationRunRequest(BaseModel):
    """Request to run a full evaluation."""
    dataset_name: str
    clauses: List[EvaluationDatasetClause]
    compliance_ground_truth: List[EvaluationComplianceGT] = []
    framework: str = "GDPR"
    run_baseline: bool = True
    run_ablation: bool = True
    run_statistical_tests: bool = True


class EvaluationRunResponse(BaseModel):
    """Response from an evaluation run."""
    id: UUID
    dataset_name: str
    model_version: Optional[str] = None
    classification_metrics: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    compliance_metrics: Optional[Dict[str, Any]] = None
    baseline_comparison: Optional[Dict[str, Any]] = None
    ablation_results: Optional[Dict[str, Any]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    report_markdown: Optional[str] = None
    graph_data: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationListItem(BaseModel):
    """Summary of an evaluation run for listing."""
    id: UUID
    dataset_name: str
    model_version: Optional[str] = None
    f1_macro: Optional[float] = None
    risk_mae: Optional[float] = None
    created_at: Optional[str] = None
