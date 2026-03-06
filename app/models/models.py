"""
SQLAlchemy ORM models — mirrors database/init.sql schema.
Module 1: Processing metrics, clause entities, embedding references.
Module 2: ClauseClassification model, classification status on Clause.
Module 3: ClauseRiskScore model, overall_risk_score/level on Document.
Module 4: ClauseExplanation model.
Hardened: UniqueConstraint, version column for optimistic locking.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, BigInteger, Integer, Float, Boolean,
    ForeignKey, CheckConstraint, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Tenant(Base):
    """Module 8: Multi-tenant organization."""
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    subscription_plan = Column(String(50), nullable=False, default="free", server_default="free")
    status = Column(String(20), nullable=False, default="active", server_default="active")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('active','suspended','deleted')",
            name="ck_tenant_status",
        ),
    )

    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    full_name = Column(String(200), nullable=True)
    role = Column(String(20), nullable=False, default="ANALYST", server_default="ANALYST")
    is_active = Column(Boolean, default=True, server_default="true")
    last_login_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "role IN ('ADMIN','ANALYST','VIEWER')",
            name="ck_user_role",
        ),
        Index("idx_user_tenant", "tenant_id"),
    )

    tenant = relationship("Tenant", back_populates="users")
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True)
    filename = Column(Text, nullable=False)
    original_filename = Column(Text, nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    mime_type = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="uploaded", server_default="uploaded")
    extracted_text = Column(Text, nullable=True)
    num_pages = Column(Integer, nullable=True)
    is_scanned = Column(Boolean, default=False, server_default="false")
    source_url = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Optimistic locking (Hardening Category 2)
    version = Column(Integer, nullable=False, default=1, server_default="1")

    # Module 1: Processing metrics
    processing_time_ms = Column(Integer, nullable=True)
    raw_text_length = Column(Integer, nullable=True)
    total_clauses = Column(Integer, default=0, server_default="0")
    total_entities = Column(Integer, default=0, server_default="0")

    # Module 3: Document-level risk
    overall_risk_score = Column(Float, nullable=True)       # 0-100
    overall_risk_level = Column(String(20), nullable=True)  # LOW/MEDIUM/HIGH/CRITICAL

    # Research reproducibility (Hardening Category 8)
    config_snapshot = Column(JSONB, nullable=True)

    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    extracted_at = Column(TIMESTAMP(timezone=True), nullable=True)
    processed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    analyzed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('uploaded','processing','extracted','processed','classified','analyzed','failed')",
            name="ck_document_status",
        ),
        Index("idx_document_tenant", "tenant_id"),
        Index("idx_document_tenant_created", "tenant_id", "uploaded_at"),
    )

    owner = relationship("User", back_populates="documents")
    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan")
    analysis = relationship("AnalysisResult", back_populates="document", uselist=False, cascade="all, delete-orphan")
    compliance_reports = relationship("DocumentComplianceReport", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("DocumentSummary", back_populates="document", uselist=False, cascade="all, delete-orphan")


class Clause(Base):
    __tablename__ = "clauses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    clause_text = Column(Text, nullable=False)
    clause_index = Column(Integer, nullable=False)
    category = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)
    risk_score = Column(Float, nullable=True)
    explanation = Column(Text, nullable=True)

    # Module 1: Clause intelligence
    embedding_id = Column(String(255), nullable=True)
    entity_count = Column(Integer, default=0, server_default="0")
    entities = Column(JSONB, nullable=True)

    # Module 2: Classification
    classification_status = Column(String(20), nullable=True)
    classification_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('low','medium','high','critical')",
            name="ck_clause_risk_level",
        ),
        UniqueConstraint("document_id", "clause_index", name="uq_clause_doc_index"),
        Index("idx_clauses_position", "document_id", "clause_index"),
    )

    document = relationship("Document", back_populates="clauses")
    classifications = relationship("ClauseClassification", back_populates="clause", cascade="all, delete-orphan")
    risk_scores = relationship("ClauseRiskScore", back_populates="clause", cascade="all, delete-orphan")
    explanations = relationship("ClauseExplanation", back_populates="clause", cascade="all, delete-orphan")
    summary = relationship("ClauseSummary", back_populates="clause", uselist=False, cascade="all, delete-orphan")


class ClauseClassification(Base):
    """Module 2: Multi-label classification results per clause."""
    __tablename__ = "clause_classifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String(100), nullable=False, default="nlpaueb/legal-bert-base-uncased")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("clause_id", "label", name="uq_classification_clause_label"),
        Index("idx_classifications_clause", "clause_id"),
        Index("idx_classifications_label", "label"),
    )

    clause = relationship("Clause", back_populates="classifications")


class ClauseRiskScore(Base):
    """Module 3: Risk scoring results per clause."""
    __tablename__ = "clause_risk_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False)
    risk_score = Column(Float, nullable=False)           # 0-100
    risk_level = Column(String(20), nullable=False)      # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors = Column(JSONB, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("clause_id", name="uq_risk_clause"),
        Index("idx_risk_scores_clause", "clause_id"),
        Index("idx_risk_scores_level", "risk_level"),
    )

    clause = relationship("Clause", back_populates="risk_scores")


class ClauseExplanation(Base):
    """Module 4: Stores classification and risk explanations per clause."""
    __tablename__ = "clause_explanations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False)
    explanation_type = Column(String(20), nullable=False)
    explanation_data = Column(JSONB, nullable=False)
    model_version = Column(String(100), nullable=True)
    generated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("clause_id", "explanation_type", name="uq_explanation_clause_type"),
        Index("idx_explanations_clause", "clause_id"),
        Index("idx_explanations_type", "explanation_type"),
    )

    clause = relationship("Clause", back_populates="explanations")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False)
    summary = Column(Text, nullable=True)
    overall_risk = Column(String(20), nullable=True)
    overall_score = Column(Float, nullable=True)
    recommendations = Column(JSONB, nullable=True)
    category_breakdown = Column(JSONB, nullable=True)
    analysis_metadata = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "overall_risk IN ('low','medium','high','critical')",
            name="ck_analysis_overall_risk",
        ),
    )

    document = relationship("Document", back_populates="analysis")


class DocumentComplianceReport(Base):
    """Module 5: Compliance report per document per framework."""
    __tablename__ = "document_compliance_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    framework = Column(String(50), nullable=False)
    compliance_score = Column(Float, nullable=False)
    coverage_percentage = Column(Float, nullable=False)
    total_articles = Column(Integer, nullable=False, default=0)
    satisfied_count = Column(Integer, nullable=False, default=0)
    partial_count = Column(Integer, nullable=False, default=0)
    missing_count = Column(Integer, nullable=False, default=0)
    missing_requirements = Column(JSONB, nullable=True)
    partial_requirements = Column(JSONB, nullable=True)
    fully_satisfied = Column(JSONB, nullable=True)
    config_snapshot = Column(JSONB, nullable=True)
    model_version = Column(String(100), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("document_id", "framework", name="uq_compliance_doc_framework"),
        Index("idx_compliance_document", "document_id"),
        Index("idx_compliance_framework", "framework"),
    )

    document = relationship("Document", back_populates="compliance_reports")


class DocumentSummary(Base):
    """Module 6: Document-level summary and executive insights."""
    __tablename__ = "document_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    overall_summary = Column(Text, nullable=False)
    executive_insights = Column(JSONB, nullable=True)
    config_snapshot = Column(JSONB, nullable=True)
    model_version = Column(String(100), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("document_id", name="uq_summary_document"),
        Index("idx_summary_document", "document_id"),
    )

    document = relationship("Document", back_populates="summaries")


class ClauseSummary(Base):
    """Module 6: Per-clause plain-English summary."""
    __tablename__ = "clause_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clause_id = Column(UUID(as_uuid=True), ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False)
    plain_summary = Column(Text, nullable=False)
    risk_note = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("clause_id", name="uq_clause_summary"),
        Index("idx_clause_summary_clause", "clause_id"),
    )

    clause = relationship("Clause", back_populates="summary")


class EvaluationRun(Base):
    """Module 7: Research evaluation run results."""
    __tablename__ = "evaluation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True)
    dataset_name = Column(String(200), nullable=False)
    model_version = Column(String(100), nullable=True)
    config_snapshot = Column(JSONB, nullable=True)
    classification_metrics = Column(JSONB, nullable=True)
    risk_metrics = Column(JSONB, nullable=True)
    compliance_metrics = Column(JSONB, nullable=True)
    baseline_comparison = Column(JSONB, nullable=True)
    ablation_results = Column(JSONB, nullable=True)
    statistical_tests = Column(JSONB, nullable=True)
    report_markdown = Column(Text, nullable=True)
    graph_data = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_evaluation_dataset", "dataset_name"),
        Index("idx_evaluation_created", "created_at"),
        Index("idx_evaluation_tenant", "tenant_id"),
    )


class AuditLog(Base):
    """Module 8: Audit trail for all critical operations."""
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_audit_tenant", "tenant_id"),
        Index("idx_audit_tenant_action", "tenant_id", "action"),
        Index("idx_audit_created", "created_at"),
    )


