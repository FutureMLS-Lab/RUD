from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel, Column
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB


JSON_COLUMN = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EvalRun(SQLModel, table=True):
    """A generation run (formerly RunCluster) that evaluates kernels for a specific spec."""
    __tablename__ = "eval_runs"

    run_id: str = Field(primary_key=True)
    plugin: str = Field(default="", index=True)
    target: str = Field(default="", index=True)
    shape_json: dict = Field(default_factory=dict, sa_column=Column(JSON_COLUMN, nullable=False))
    task_slug: str = Field(default="", index=True)
    reference_plugin: str = Field(index=True)
    spec_json: dict = Field(sa_column=Column(JSON_COLUMN))
    function_name: str = Field(index=True)
    scalar_args: dict = Field(sa_column=Column(JSON_COLUMN))
    run_contract: dict = Field(default_factory=dict, sa_column=Column(JSON_COLUMN, nullable=False))
    gpu: str = Field(default="h100")
    overhead_pct: float = Field(default=30.0)
    model: Optional[str] = None
    sm_version: str = Field(default="sm_90a")
    cuda_version: Optional[str] = None
    started_at: datetime = Field(default_factory=_now)
    ended_at: Optional[datetime] = None
    baseline_source: Optional[str] = None


class KernelLibrary(SQLModel, table=True):
    """A kernel submission with benchmark results."""
    __tablename__ = "kernel_library"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(unique=True)
    run_id: Optional[str] = Field(default=None, foreign_key="eval_runs.run_id")
    agent_index: int = Field(default=0)
    kernel_us: float
    baseline_us: float
    kernel_source: str
    postprocessed_source: Optional[str] = None
    python_registration: Optional[str] = None
    gpu: str = Field(default="h100")
    achieved_at: datetime = Field(default_factory=_now)
    valid: bool = Field(default=True)
    invalidation_reason: Optional[str] = None

    # Fields for external kernels (when run_id is None)
    plugin: Optional[str] = None
    spec_json: Optional[dict] = Field(default=None, sa_column=Column(JSON_COLUMN, nullable=True))
    function_name: Optional[str] = None
    scalar_args: Optional[dict] = Field(default=None, sa_column=Column(JSON_COLUMN, nullable=True))


class ApiKey(SQLModel, table=True):
    __tablename__ = "api_keys"

    key_id: str = Field(primary_key=True)
    key_hash: str = Field(unique=True, index=True)
    role: str = Field(index=True)
    created_at: datetime = Field(default_factory=_now)
    revoked_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
