from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE eval_runs ALTER COLUMN overhead_pct SET DEFAULT 30.0")


def downgrade() -> None:
    op.execute("ALTER TABLE eval_runs ALTER COLUMN overhead_pct SET DEFAULT 0.0")
