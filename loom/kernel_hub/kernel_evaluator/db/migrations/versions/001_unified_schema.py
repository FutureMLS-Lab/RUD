"""Unified schema for kernel_evaluator.

This migration creates the new schema from scratch. If migrating from
turbo-gemm's existing schema, run the data migration script separately.

Revision ID: 001
Revises:
Create Date: 2026-05-13
"""
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create eval_runs table
    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            reference_plugin TEXT NOT NULL,
            spec_json JSONB NOT NULL,
            function_name TEXT NOT NULL,
            scalar_args JSONB NOT NULL,
            gpu TEXT NOT NULL DEFAULT 'h100',
            overhead_pct FLOAT NOT NULL DEFAULT 30.0,
            model TEXT,
            sm_version TEXT NOT NULL DEFAULT 'sm_90a',
            cuda_version TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            ended_at TIMESTAMPTZ,
            baseline_source TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_reference_plugin ON eval_runs(reference_plugin)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_function_name ON eval_runs(function_name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_scalar_args ON eval_runs USING GIN (scalar_args)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_spec_json ON eval_runs USING GIN (spec_json)")

    # Create kernel_library table
    op.execute("""
        CREATE TABLE IF NOT EXISTS kernel_library (
            id SERIAL PRIMARY KEY,
            job_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL REFERENCES eval_runs(run_id),
            agent_index INTEGER NOT NULL DEFAULT 0,
            kernel_us FLOAT NOT NULL,
            baseline_us FLOAT NOT NULL,
            kernel_source TEXT NOT NULL,
            postprocessed_source TEXT,
            python_registration TEXT,
            gpu TEXT NOT NULL DEFAULT 'h100',
            achieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            valid BOOLEAN NOT NULL DEFAULT TRUE,
            invalidation_reason TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_kernel_library_valid ON kernel_library(valid) WHERE valid = TRUE")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kernel_library")
    op.execute("DROP TABLE IF EXISTS eval_runs")
