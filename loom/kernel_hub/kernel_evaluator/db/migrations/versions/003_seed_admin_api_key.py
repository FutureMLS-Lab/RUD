from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO api_keys (key_id, key_hash, role)
        VALUES (
            'bootstrap-admin',
            'ac9689e2272427085e35b9d3e3e8bed88cb3434828b43b86fc0596cad4c6e270',
            'admin'
        )
        ON CONFLICT (key_id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM api_keys WHERE key_id = 'bootstrap-admin'")
