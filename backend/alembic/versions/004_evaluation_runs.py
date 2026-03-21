"""Add evaluation_runs table

Revision ID: 004_evaluation
Revises: 003_summaries
Create Date: 2026-02-22

Module 7: Analytics, Benchmarking & Research Validation Engine.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '004_evaluation'
down_revision: Union[str, None] = '003_summaries'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('evaluation_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('dataset_name', sa.String(200), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.Column('config_snapshot', postgresql.JSONB(), nullable=True),
        sa.Column('classification_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('risk_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('compliance_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('baseline_comparison', postgresql.JSONB(), nullable=True),
        sa.Column('ablation_results', postgresql.JSONB(), nullable=True),
        sa.Column('statistical_tests', postgresql.JSONB(), nullable=True),
        sa.Column('report_markdown', sa.Text(), nullable=True),
        sa.Column('graph_data', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_evaluation_dataset', 'evaluation_runs', ['dataset_name'])
    op.create_index('idx_evaluation_created', 'evaluation_runs', ['created_at'])


def downgrade() -> None:
    op.drop_table('evaluation_runs')
