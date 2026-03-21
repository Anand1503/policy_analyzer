"""Add document_summaries and clause_summaries tables

Revision ID: 003_summaries
Revises: 002_compliance
Create Date: 2026-02-22

Module 6: Intelligent Policy Simplification & Executive Insight Engine.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '003_summaries'
down_revision: Union[str, None] = '002_compliance'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Document-level summaries
    op.create_table('document_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('overall_summary', sa.Text(), nullable=False),
        sa.Column('executive_insights', postgresql.JSONB(), nullable=True),
        sa.Column('config_snapshot', postgresql.JSONB(), nullable=True),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.UniqueConstraint('document_id', name='uq_summary_document'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_summary_document', 'document_summaries', ['document_id'])

    # Per-clause summaries
    op.create_table('clause_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('clause_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('plain_summary', sa.Text(), nullable=False),
        sa.Column('risk_note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.UniqueConstraint('clause_id', name='uq_clause_summary'),
        sa.ForeignKeyConstraint(['clause_id'], ['clauses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_clause_summary_clause', 'clause_summaries', ['clause_id'])


def downgrade() -> None:
    op.drop_table('clause_summaries')
    op.drop_table('document_summaries')
