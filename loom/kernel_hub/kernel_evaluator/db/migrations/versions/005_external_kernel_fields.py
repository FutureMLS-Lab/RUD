from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE kernel_library ALTER COLUMN run_id DROP NOT NULL")
    op.execute("ALTER TABLE kernel_library ADD COLUMN IF NOT EXISTS plugin TEXT")
    op.execute("ALTER TABLE kernel_library ADD COLUMN IF NOT EXISTS spec_json JSONB")
    op.execute("ALTER TABLE kernel_library ADD COLUMN IF NOT EXISTS function_name TEXT")
    op.execute("ALTER TABLE kernel_library ADD COLUMN IF NOT EXISTS scalar_args JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE kernel_library DROP COLUMN IF EXISTS scalar_args")
    op.execute("ALTER TABLE kernel_library DROP COLUMN IF EXISTS function_name")
    op.execute("ALTER TABLE kernel_library DROP COLUMN IF EXISTS spec_json")
    op.execute("ALTER TABLE kernel_library DROP COLUMN IF EXISTS plugin")
