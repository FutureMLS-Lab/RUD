from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            revoked_at TIMESTAMPTZ,
            last_used_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_api_keys_key_hash ON api_keys(key_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_api_keys_role ON api_keys(role)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS api_keys")
