-- ═══════════════════════════════════════════════════════════
-- Intelligent Policy Analyzer — Full Database Schema
-- PostgreSQL 14+
-- Hardened: UNIQUE constraints, version column, ON CONFLICT support
-- ═══════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Tenants (Module 8) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS tenants (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                VARCHAR(200) NOT NULL,
    subscription_plan   VARCHAR(50) NOT NULL DEFAULT 'free',
    status              VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT ck_tenant_status CHECK (status IN ('active','suspended','deleted'))
);

-- ─── Users ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    email           VARCHAR(255) UNIQUE NOT NULL,
    username        VARCHAR(100) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name       VARCHAR(200),
    role            VARCHAR(20) NOT NULL DEFAULT 'ANALYST',
    is_active       BOOLEAN DEFAULT TRUE,
    last_login_at   TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT ck_user_role CHECK (role IN ('ADMIN','ANALYST','VIEWER'))
);

-- ─── Documents ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id           UUID REFERENCES users(id) ON DELETE CASCADE,
    tenant_id         UUID REFERENCES tenants(id) ON DELETE CASCADE,
    filename          TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_type         VARCHAR(10) NOT NULL,
    file_size_bytes   BIGINT NOT NULL,
    mime_type         VARCHAR(100) NOT NULL,
    status            VARCHAR(20) NOT NULL DEFAULT 'uploaded'
                      CHECK (status IN ('uploaded','processing','extracted','processed','classified','analyzed','failed')),
    extracted_text    TEXT,
    num_pages         INTEGER,
    is_scanned        BOOLEAN DEFAULT FALSE,
    source_url        TEXT,
    error_message     TEXT,

    -- Optimistic locking (Category 2: Concurrency)
    version           INTEGER NOT NULL DEFAULT 1,

    -- Module 1: Processing metrics
    processing_time_ms  INTEGER,
    raw_text_length     INTEGER,
    total_clauses       INTEGER DEFAULT 0,
    total_entities      INTEGER DEFAULT 0,

    -- Module 3: Document-level risk
    overall_risk_score  FLOAT,
    overall_risk_level  VARCHAR(20),

    -- Research reproducibility (Category 8)
    config_snapshot     JSONB,              -- risk_config at time of analysis

    uploaded_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    extracted_at      TIMESTAMP WITH TIME ZONE,
    processed_at      TIMESTAMP WITH TIME ZONE,
    analyzed_at       TIMESTAMP WITH TIME ZONE
);

-- ─── Clauses ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clauses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_text     TEXT NOT NULL,
    clause_index    INTEGER NOT NULL,
    category        VARCHAR(50),
    confidence      FLOAT,
    risk_level      VARCHAR(20)
                    CHECK (risk_level IN ('low','medium','high','critical')),
    risk_score      FLOAT,
    explanation     TEXT,

    -- Module 1: Clause intelligence
    embedding_id    VARCHAR(255),
    entity_count    INTEGER DEFAULT 0,
    entities        JSONB,

    -- Module 2: Classification
    classification_status       VARCHAR(20),
    classification_completed_at TIMESTAMP WITH TIME ZONE,

    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Prevent duplicate clause indexes per document
    UNIQUE (document_id, clause_index)
);

-- ─── Analysis Results ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID UNIQUE NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    summary         TEXT,
    overall_risk    VARCHAR(20) CHECK (overall_risk IN ('low','medium','high','critical')),
    overall_score   FLOAT,
    recommendations JSONB,
    category_breakdown JSONB,
    analysis_metadata JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─── Clause Classifications (Module 2) ──────────────────────
CREATE TABLE IF NOT EXISTS clause_classifications (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id         UUID NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    label             VARCHAR(50) NOT NULL,
    confidence_score  FLOAT NOT NULL,
    model_version     VARCHAR(100) NOT NULL DEFAULT 'facebook/bart-large-mnli',
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Category 1: Prevent duplicate label per clause
    UNIQUE (clause_id, label)
);

-- ─── Clause Risk Scores (Module 3) ──────────────────────────
CREATE TABLE IF NOT EXISTS clause_risk_scores (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id       UUID NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    risk_score      FLOAT NOT NULL,
    risk_level      VARCHAR(20) NOT NULL,
    risk_factors    JSONB,
    explanation     TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Category 1: One risk score per clause (latest wins)
    UNIQUE (clause_id)
);

-- ─── Clause Explanations (Module 4) ─────────────────────────
CREATE TABLE IF NOT EXISTS clause_explanations (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id         UUID NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    explanation_type  VARCHAR(20) NOT NULL,
    explanation_data  JSONB NOT NULL,
    model_version     VARCHAR(100),
    generated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Category 1: One explanation per type per clause
    UNIQUE (clause_id, explanation_type)
);

-- ─── Document Compliance Reports (Module 5) ─────────────────
CREATE TABLE IF NOT EXISTS document_compliance_reports (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id           UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    framework             VARCHAR(50) NOT NULL,
    compliance_score      FLOAT NOT NULL,
    coverage_percentage   FLOAT NOT NULL,
    total_articles        INTEGER NOT NULL DEFAULT 0,
    satisfied_count       INTEGER NOT NULL DEFAULT 0,
    partial_count         INTEGER NOT NULL DEFAULT 0,
    missing_count         INTEGER NOT NULL DEFAULT 0,
    missing_requirements  JSONB,
    partial_requirements  JSONB,
    fully_satisfied       JSONB,
    config_snapshot       JSONB,
    model_version         VARCHAR(100),
    created_at            TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE (document_id, framework)
);

-- ─── Document Summaries (Module 6) ──────────────────────────
CREATE TABLE IF NOT EXISTS document_summaries (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    overall_summary     TEXT NOT NULL,
    executive_insights  JSONB,
    config_snapshot     JSONB,
    model_version       VARCHAR(100),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE (document_id)
);

-- ─── Clause Summaries (Module 6) ────────────────────────────
CREATE TABLE IF NOT EXISTS clause_summaries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id       UUID NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    plain_summary   TEXT NOT NULL,
    risk_note       TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE (clause_id)
);

-- ─── Indexes ────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_documents_user         ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status       ON documents(status);
CREATE INDEX IF NOT EXISTS idx_clauses_document       ON clauses(document_id);
CREATE INDEX IF NOT EXISTS idx_clauses_position       ON clauses(document_id, clause_index);
CREATE INDEX IF NOT EXISTS idx_clauses_category       ON clauses(category);
CREATE INDEX IF NOT EXISTS idx_clauses_risk           ON clauses(risk_level);
CREATE INDEX IF NOT EXISTS idx_clauses_embedding      ON clauses(embedding_id);
CREATE INDEX IF NOT EXISTS idx_analysis_document      ON analysis_results(document_id);
CREATE INDEX IF NOT EXISTS idx_classifications_clause ON clause_classifications(clause_id);
CREATE INDEX IF NOT EXISTS idx_classifications_label  ON clause_classifications(label);
CREATE INDEX IF NOT EXISTS idx_risk_scores_clause     ON clause_risk_scores(clause_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_level      ON clause_risk_scores(risk_level);
CREATE INDEX IF NOT EXISTS idx_explanations_clause    ON clause_explanations(clause_id);
CREATE INDEX IF NOT EXISTS idx_explanations_type      ON clause_explanations(explanation_type);
CREATE INDEX IF NOT EXISTS idx_compliance_document    ON document_compliance_reports(document_id);
CREATE INDEX IF NOT EXISTS idx_compliance_framework   ON document_compliance_reports(framework);
CREATE INDEX IF NOT EXISTS idx_summary_document       ON document_summaries(document_id);
CREATE INDEX IF NOT EXISTS idx_clause_summary_clause  ON clause_summaries(clause_id);

-- ─── Evaluation Runs (Module 7) ─────────────────────────────
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id                 UUID REFERENCES tenants(id) ON DELETE CASCADE,
    dataset_name              VARCHAR(200) NOT NULL,
    model_version             VARCHAR(100),
    config_snapshot           JSONB,
    classification_metrics    JSONB,
    risk_metrics              JSONB,
    compliance_metrics        JSONB,
    baseline_comparison       JSONB,
    ablation_results          JSONB,
    statistical_tests         JSONB,
    report_markdown           TEXT,
    graph_data                JSONB,
    created_at                TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluation_dataset  ON evaluation_runs(dataset_name);
CREATE INDEX IF NOT EXISTS idx_evaluation_created  ON evaluation_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_evaluation_tenant   ON evaluation_runs(tenant_id);

-- ─── Audit Logs (Module 8) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    user_id         UUID,
    action          VARCHAR(50) NOT NULL,
    resource_type   VARCHAR(50) NOT NULL,
    resource_id     UUID,
    metadata_json   JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─── Module 8 Indexes ───────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_user_tenant              ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_document_tenant          ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_document_tenant_created  ON documents(tenant_id, uploaded_at);
CREATE INDEX IF NOT EXISTS idx_audit_tenant             ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_action      ON audit_logs(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_audit_created            ON audit_logs(created_at);

