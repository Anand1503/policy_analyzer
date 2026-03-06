"""Add document_compliance_reports table

Revision ID: 002_compliance
Revises: 001_initial
Create Date: 2026-02-22

Module 5: Regulatory Compliance Mapping & Gap Detection.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '002_compliance'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('document_compliance_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('framework', sa.String(50), nullable=False),
        sa.Column('compliance_score', sa.Float(), nullable=False),
        sa.Column('coverage_percentage', sa.Float(), nullable=False),
        sa.Column('total_articles', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('satisfied_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('partial_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('missing_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('missing_requirements', postgresql.JSONB(), nullable=True),
        sa.Column('partial_requirements', postgresql.JSONB(), nullable=True),
        sa.Column('fully_satisfied', postgresql.JSONB(), nullable=True),
        sa.Column('config_snapshot', postgresql.JSONB(), nullable=True),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.UniqueConstraint('document_id', 'framework', name='uq_compliance_doc_framework'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_compliance_document', 'document_compliance_reports', ['document_id'])
    op.create_index('idx_compliance_framework', 'document_compliance_reports', ['framework'])


def downgrade() -> None:
    op.drop_table('document_compliance_reports')
