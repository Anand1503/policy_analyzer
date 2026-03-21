"""Initial schema — full hardened schema

Revision ID: 001_initial
Revises: None
Create Date: 2026-02-22

This is a baseline migration capturing the complete schema
with all hardening constraints (UNIQUE, version, config_snapshot).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # Users
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('hashed_password', sa.Text(), nullable=False),
        sa.Column('full_name', sa.String(200), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username'),
    )

    # Documents
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('original_filename', sa.Text(), nullable=False),
        sa.Column('file_type', sa.String(10), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('mime_type', sa.String(100), nullable=False),
        sa.Column('status', sa.String(20), server_default='uploaded', nullable=False),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('num_pages', sa.Integer(), nullable=True),
        sa.Column('is_scanned', sa.Boolean(), server_default='false'),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('raw_text_length', sa.Integer(), nullable=True),
        sa.Column('total_clauses', sa.Integer(), server_default='0'),
        sa.Column('total_entities', sa.Integer(), server_default='0'),
        sa.Column('overall_risk_score', sa.Float(), nullable=True),
        sa.Column('overall_risk_level', sa.String(20), nullable=True),
        sa.Column('config_snapshot', postgresql.JSONB(), nullable=True),
        sa.Column('uploaded_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.Column('extracted_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('analyzed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('uploaded','processing','extracted','processed','classified','analyzed','failed')", name='ck_document_status'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Clauses
    op.create_table('clauses',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('clause_text', sa.Text(), nullable=False),
        sa.Column('clause_index', sa.Integer(), nullable=False),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('risk_level', sa.String(20), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('embedding_id', sa.String(255), nullable=True),
        sa.Column('entity_count', sa.Integer(), server_default='0'),
        sa.Column('entities', postgresql.JSONB(), nullable=True),
        sa.Column('classification_status', sa.String(20), nullable=True),
        sa.Column('classification_completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.CheckConstraint("risk_level IN ('low','medium','high','critical')", name='ck_clause_risk_level'),
        sa.UniqueConstraint('document_id', 'clause_index', name='uq_clause_doc_index'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Analysis Results
    op.create_table('analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('overall_risk', sa.String(20), nullable=True),
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('recommendations', postgresql.JSONB(), nullable=True),
        sa.Column('category_breakdown', postgresql.JSONB(), nullable=True),
        sa.Column('analysis_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.CheckConstraint("overall_risk IN ('low','medium','high','critical')", name='ck_analysis_overall_risk'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id'),
    )

    # Clause Classifications
    op.create_table('clause_classifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('clause_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('label', sa.String(50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(100), server_default='nlpaueb/legal-bert-base-uncased', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.UniqueConstraint('clause_id', 'label', name='uq_classification_clause_label'),
        sa.ForeignKeyConstraint(['clause_id'], ['clauses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Clause Risk Scores
    op.create_table('clause_risk_scores',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('clause_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('risk_level', sa.String(20), nullable=False),
        sa.Column('risk_factors', postgresql.JSONB(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.UniqueConstraint('clause_id', name='uq_risk_clause'),
        sa.ForeignKeyConstraint(['clause_id'], ['clauses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Clause Explanations
    op.create_table('clause_explanations',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('clause_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('explanation_type', sa.String(20), nullable=False),
        sa.Column('explanation_data', postgresql.JSONB(), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.Column('generated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.UniqueConstraint('clause_id', 'explanation_type', name='uq_explanation_clause_type'),
        sa.ForeignKeyConstraint(['clause_id'], ['clauses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Indexes
    op.create_index('idx_documents_user', 'documents', ['user_id'])
    op.create_index('idx_documents_status', 'documents', ['status'])
    op.create_index('idx_clauses_document', 'clauses', ['document_id'])
    op.create_index('idx_clauses_category', 'clauses', ['category'])
    op.create_index('idx_clauses_risk', 'clauses', ['risk_level'])
    op.create_index('idx_clauses_embedding', 'clauses', ['embedding_id'])
    op.create_index('idx_classifications_clause', 'clause_classifications', ['clause_id'])
    op.create_index('idx_classifications_label', 'clause_classifications', ['label'])
    op.create_index('idx_risk_scores_clause', 'clause_risk_scores', ['clause_id'])
    op.create_index('idx_risk_scores_level', 'clause_risk_scores', ['risk_level'])
    op.create_index('idx_explanations_clause', 'clause_explanations', ['clause_id'])
    op.create_index('idx_explanations_type', 'clause_explanations', ['explanation_type'])


def downgrade() -> None:
    op.drop_table('clause_explanations')
    op.drop_table('clause_risk_scores')
    op.drop_table('clause_classifications')
    op.drop_table('analysis_results')
    op.drop_table('clauses')
    op.drop_table('documents')
    op.drop_table('users')
