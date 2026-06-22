from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS plugin TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS target TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS shape_json JSONB NOT NULL DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS task_slug TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS run_contract JSONB NOT NULL DEFAULT '{}'::jsonb")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_plugin ON eval_runs(plugin)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_target ON eval_runs(target)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_task_slug ON eval_runs(task_slug)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_run_contract ON eval_runs USING GIN (run_contract)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_eval_runs_run_contract")
    op.execute("DROP INDEX IF EXISTS ix_eval_runs_task_slug")
    op.execute("DROP INDEX IF EXISTS ix_eval_runs_target")
    op.execute("DROP INDEX IF EXISTS ix_eval_runs_plugin")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS run_contract")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS task_slug")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS shape_json")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS target")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS plugin")
