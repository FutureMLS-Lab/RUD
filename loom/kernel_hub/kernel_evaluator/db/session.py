import os

from sqlalchemy import create_engine

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10, pool_pre_ping=True)


def create_tables() -> None:
    """Run Alembic migrations to create/update tables."""
    from alembic import command
    from alembic.config import Config

    alembic_ini = os.path.join(os.path.dirname(__file__), "..", "alembic.ini")
    cfg = Config(os.path.abspath(alembic_ini))
    command.upgrade(cfg, "head")
