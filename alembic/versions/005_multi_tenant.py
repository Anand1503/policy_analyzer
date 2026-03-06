"""Add multi-tenant RBAC tables and columns

Revision ID: 005_multi_tenant
Revises: 004_evaluation
Create Date: 2026-02-22

Module 8: Multi-Tenant SaaS & Role-Based Access Control.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '005_multi_tenant'
down_revision: Union[str, None] = '004_evaluation'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Tenants table
    op.create_table('tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('subscription_plan', sa.String(50), server_default='free', nullable=False),
        sa.Column('status', sa.String(20), server_default='active', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.CheckConstraint("status IN ('active','suspended','deleted')", name='ck_tenant_status'),
        sa.PrimaryKeyConstraint('id'),
    )

    # 2. Enhance users
    op.add_column('users', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('users', sa.Column('role', sa.String(20), server_default='ANALYST', nullable=False))
    op.add_column('users', sa.Column('last_login_at', sa.TIMESTAMP(timezone=True), nullable=True))
    op.create_foreign_key('fk_user_tenant', 'users', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_check_constraint('ck_user_role', 'users', "role IN ('ADMIN','ANALYST','VIEWER')")
    op.create_index('idx_user_tenant', 'users', ['tenant_id'])

    # 3. Add tenant_id to documents
    op.add_column('documents', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_document_tenant', 'documents', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_index('idx_document_tenant', 'documents', ['tenant_id'])
    op.create_index('idx_document_tenant_created', 'documents', ['tenant_id', 'uploaded_at'])

    # 4. Add tenant_id to evaluation_runs
    op.add_column('evaluation_runs', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_evaluation_tenant', 'evaluation_runs', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')
    op.create_index('idx_evaluation_tenant', 'evaluation_runs', ['tenant_id'])

    # 5. Audit log table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('resource_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_audit_tenant', 'audit_logs', ['tenant_id'])
    op.create_index('idx_audit_tenant_action', 'audit_logs', ['tenant_id', 'action'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])


def downgrade() -> None:
    op.drop_table('audit_logs')

    op.drop_index('idx_evaluation_tenant', 'evaluation_runs')
    op.drop_constraint('fk_evaluation_tenant', 'evaluation_runs', type_='foreignkey')
    op.drop_column('evaluation_runs', 'tenant_id')

    op.drop_index('idx_document_tenant_created', 'documents')
    op.drop_index('idx_document_tenant', 'documents')
    op.drop_constraint('fk_document_tenant', 'documents', type_='foreignkey')
    op.drop_column('documents', 'tenant_id')

    op.drop_index('idx_user_tenant', 'users')
    op.drop_constraint('ck_user_role', 'users', type_='check')
    op.drop_constraint('fk_user_tenant', 'users', type_='foreignkey')
    op.drop_column('users', 'last_login_at')
    op.drop_column('users', 'role')
    op.drop_column('users', 'tenant_id')

    op.drop_table('tenants')
