"""
Pydantic schemas for analysis, classification, risk scoring, and explainability.
Module 2: Classification
Module 3: Risk Analysis
Module 4: Explainability
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime


# ─── Classification (Module 2) ───────────────────────────────

class ClassificationLabel(BaseModel):
    label: str
    confidence: float


class ClauseClassificationResult(BaseModel):
    clause_id: Optional[UUID] = None
    clause_index: int
    labels: List[ClassificationLabel] = []
    primary_label: str
    primary_confidence: float

    model_config = ConfigDict(from_attributes=True)


class ClassificationResponse(BaseModel):
    document_id: UUID
    status: str
    total_clauses: int
    classified_clauses: int
    average_confidence: float
    classification_time_ms: int
    model_version: str
    label_distribution: Dict[str, int] = {}


# ─── Risk Scoring (Module 3) ────────────────────────────────

class ClauseRiskResult(BaseModel):
    clause_id: Optional[UUID] = None
    clause_index: int = 0
    risk_score: float
    risk_level: str
    risk_factors: List[str] = []
    explanation: Optional[str] = None


class RiskAnalysisResponse(BaseModel):
    document_id: UUID
    status: str
    overall_risk_score: float
    risk_level: str
    total_clauses: int
    total_high_risk_clauses: int
    risk_distribution: Dict[str, int] = {}
    analysis_time_ms: int


# ─── Explainability (Module 4) ──────────────────────────────

class LabelExplanation(BaseModel):
    """Token-level explanation for a single classification label."""
    label: str
    confidence: float
    top_influential_terms: List[str] = []
    term_count: int = 0
    influence_score: float = 0.0


class ClassificationExplanation(BaseModel):
    """Classification explanation for a clause."""
    clause_id: str
    label_explanations: List[LabelExplanation] = []


class RiskExplanationFactors(BaseModel):
    """Decomposed risk score factors."""
    base_risk: float = 0.0
    entity_bonus: float = 0.0
    pattern_bonus: float = 0.0
    position_factor: float = 1.0


class RiskExplanation(BaseModel):
    """Risk explanation for a clause."""
    clause_id: str
    risk_score: float
    risk_level: str
    explanation: RiskExplanationFactors
    risk_factors: List[str] = []
    entity_types_present: List[str] = []
    justification: str


class ClauseExplainabilityResult(BaseModel):
    """Combined explanation for a single clause."""
    clause_id: str
    clause_index: int = 0
    category: str = ""
    classification_explanation: ClassificationExplanation
    risk_explanation: RiskExplanation


class ExplainabilityResponse(BaseModel):
    """Response from the explain endpoint."""
    document_id: UUID
    overall_risk_score: float
    risk_level: str
    total_clauses: int
    explanation_time_ms: int
    clauses: List[ClauseExplainabilityResult] = []


# ─── Clause & Analysis Results ───────────────────────────────

class ClauseResponse(BaseModel):
    id: UUID
    clause_text: str
    clause_index: int
    category: Optional[str] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    explanation: Optional[str] = None
    embedding_id: Optional[str] = None
    entity_count: Optional[int] = None
    classification_status: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AnalysisResultResponse(BaseModel):
    id: UUID
    document_id: UUID
    summary: Optional[str] = None
    overall_risk: Optional[str] = None
    overall_score: Optional[float] = None
    recommendations: Optional[List[str]] = None
    category_breakdown: Optional[Dict[str, int]] = None
    clauses: List[ClauseResponse] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AnalysisTriggerResponse(BaseModel):
    document_id: UUID
    status: str
    message: str


class ChatRequest(BaseModel):
    query: str
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []


# ─── Compliance (Module 5) ──────────────────────────────────

class ComplianceSupportingClause(BaseModel):
    clause_id: str
    clause_index: int = -1
    risk_score: float = 0.0
    risk_level: str = "unknown"
    matched_labels: List[str] = []

class ComplianceArticleFinding(BaseModel):
    article_id: str
    title: str
    requirement: str
    status: str  # satisfied, partial, missing
    label_coverage: float
    keyword_coverage: float
    entity_coverage: float
    supporting_clauses: List[ComplianceSupportingClause] = []
    explanation: str
    importance_weight: float = 1.0

class ComplianceReportResponse(BaseModel):
    document_id: UUID
    framework: str
    compliance_score: float
    coverage_percentage: float
    total_articles: int
    satisfied_count: int
    partial_count: int
    missing_count: int
    missing_requirements: List[ComplianceArticleFinding] = []
    partial_requirements: List[ComplianceArticleFinding] = []
    fully_satisfied: List[ComplianceArticleFinding] = []

    model_config = ConfigDict(from_attributes=True)


# ─── Policy Simplification (Module 6) ───────────────────────

class ClauseSummaryItem(BaseModel):
    clause_id: str
    clause_index: int = 0
    plain_summary: str
    risk_note: str = ""


class TopRisk(BaseModel):
    category: str
    description: str
    severity: str
    avg_risk_score: float = 0.0


class ExecutiveInsightsSchema(BaseModel):
    overall_summary: str
    risk_level: str
    overall_risk_score: float
    top_risks: List[TopRisk] = []
    compliance_gaps: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    key_statistics: Dict[str, Any] = {}


class PolicySummaryResponse(BaseModel):
    document_id: UUID
    overall_summary: str
    executive_insights: ExecutiveInsightsSchema
    clause_summaries: List[ClauseSummaryItem] = []
    summary_time_ms: int = 0


# ─── Unified Pipeline (Phase 1) ─────────────────────────────

class ComplianceFrameworkResult(BaseModel):
    coverage_percentage: float = 0.0
    articles_detected: List[str] = []
    missing_articles: List[str] = []
    compliance_score: float = 0.0

class UnifiedComplianceResult(BaseModel):
    gdpr: ComplianceFrameworkResult = ComplianceFrameworkResult()
    ccpa: ComplianceFrameworkResult = ComplianceFrameworkResult()

class UnifiedClauseResult(BaseModel):
    clause_index: int
    clause_text: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    explanation: Optional[str] = None

class UnifiedPipelineResponse(BaseModel):
    status: str
    document_id: UUID
    clauses: List[UnifiedClauseResult] = []
    classifications: Dict[str, int] = {}
    risk_score: float = 0.0
    risk_level: str = "low"
    compliance: UnifiedComplianceResult = UnifiedComplianceResult()
    summary: Optional[str] = None
    recommendations: List[str] = []
    explanation: List[Dict[str, Any]] = []
    processing_time: float = 0.0

