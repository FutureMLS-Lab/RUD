"""Database operations for kernel_evaluator.

Each function opens and closes its own Session so they are safe to call
from any thread.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Session, select

from kernel_evaluator.db.models import ApiKey, EvalRun, KernelLibrary
from kernel_evaluator.db.session import engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# EvalRun operations
# ---------------------------------------------------------------------------

def create_eval_run(
    run_id: str,
    reference_plugin: str,
    spec_json: dict,
    function_name: str,
    scalar_args: dict,
    plugin: str = "",
    target: str = "",
    shape_json: dict | None = None,
    task_slug: str = "",
    run_contract: dict | None = None,
    gpu: str = "h100",
    model: Optional[str] = None,
    sm_version: str = "sm_90a",
    cuda_version: Optional[str] = None,
    baseline_source: Optional[str] = None,
) -> EvalRun:
    """Create a new EvalRun."""
    with Session(engine) as session:
        existing = session.get(EvalRun, run_id)
        if existing is not None:
            return existing

        run = EvalRun(
            run_id=run_id,
            plugin=plugin,
            target=target,
            shape_json={} if shape_json is None else shape_json,
            task_slug=task_slug,
            reference_plugin=reference_plugin,
            spec_json=spec_json,
            function_name=function_name,
            scalar_args=scalar_args,
            run_contract={} if run_contract is None else run_contract,
            gpu=gpu,
            model=model,
            sm_version=sm_version,
            cuda_version=cuda_version,
            baseline_source=baseline_source,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run


def create_eval_run_from_contract(
    run_contract: dict,
    gpu: str = "h100",
    model: Optional[str] = None,
    sm_version: str = "sm_90a",
    cuda_version: Optional[str] = None,
    baseline_source: Optional[str] = None,
) -> EvalRun:
    first_shape = run_contract["benchmark_shapes"][0]
    spec_json = first_shape["spec"]
    return create_eval_run(
        run_id=run_contract["run_id"],
        plugin=run_contract["plugin"],
        target=run_contract["target"],
        shape_json=first_shape["shape"],
        task_slug=first_shape["task_slug"],
        reference_plugin=first_shape["reference_plugin"],
        spec_json=spec_json,
        function_name=spec_json["function_name"],
        scalar_args=first_shape["scalars"],
        run_contract=run_contract,
        gpu=gpu,
        model=model,
        sm_version=sm_version,
        cuda_version=cuda_version,
        baseline_source=baseline_source,
    )


def get_eval_run(run_id: str) -> Optional[EvalRun]:
    """Get an EvalRun by run_id."""
    with Session(engine) as session:
        return session.get(EvalRun, run_id)


def finish_eval_run(run_id: str) -> None:
    """Mark an EvalRun as finished."""
    with Session(engine) as session:
        run = session.get(EvalRun, run_id)
        if run:
            run.ended_at = _now()
            session.add(run)
            session.commit()


def list_eval_runs(reference_plugin: Optional[str] = None) -> list[EvalRun]:
    """List EvalRuns, optionally filtered by reference_plugin."""
    with Session(engine) as session:
        stmt = select(EvalRun).order_by(EvalRun.started_at.desc())
        if reference_plugin is not None:
            stmt = stmt.where(EvalRun.reference_plugin == reference_plugin)
        return list(session.exec(stmt).all())


def create_api_key_record(key_hash: str, role: str) -> ApiKey:
    with Session(engine) as session:
        key = ApiKey(
            key_id=uuid4().hex,
            key_hash=key_hash,
            role=role,
        )
        session.add(key)
        session.commit()
        session.refresh(key)
        return key


def get_active_api_key_by_hash(key_hash: str) -> Optional[ApiKey]:
    with Session(engine) as session:
        stmt = (
            select(ApiKey)
            .where(ApiKey.key_hash == key_hash)
            .where(ApiKey.revoked_at.is_(None))
            .limit(1)
        )
        return session.exec(stmt).first()


def mark_api_key_used(key_id: str) -> None:
    with Session(engine) as session:
        key = session.exec(select(ApiKey).where(ApiKey.key_id == key_id).limit(1)).first()
        if key is not None:
            key.last_used_at = _now()
            session.add(key)
            session.commit()


def revoke_api_key_record(key_id: str) -> bool:
    with Session(engine) as session:
        key = session.exec(select(ApiKey).where(ApiKey.key_id == key_id).limit(1)).first()
        if key is None:
            return False
        key.revoked_at = _now()
        session.add(key)
        session.commit()
        return True


# ---------------------------------------------------------------------------
# KernelLibrary operations
# ---------------------------------------------------------------------------

def insert_kernel(
    job_id: str,
    run_id: str,
    kernel_source: str,
    kernel_us: float,
    baseline_us: float,
    agent_index: int = 0,
    gpu: str = "h100",
) -> KernelLibrary:
    """Insert a new kernel submission."""
    with Session(engine) as session:
        kernel = KernelLibrary(
            job_id=job_id,
            run_id=run_id,
            agent_index=agent_index,
            kernel_source=kernel_source,
            kernel_us=kernel_us,
            baseline_us=baseline_us,
            gpu=gpu,
        )
        session.add(kernel)
        session.commit()
        session.refresh(kernel)
        return kernel


def get_kernel(kernel_id: int) -> Optional[KernelLibrary]:
    """Get a kernel by id."""
    with Session(engine) as session:
        return session.get(KernelLibrary, kernel_id)


def update_kernel_postprocessing(
    kernel_id: int,
    postprocessed_source: Optional[str] = None,
    python_registration: Optional[str] = None,
) -> bool:
    """Update postprocessed_source and/or python_registration for a kernel."""
    with Session(engine) as session:
        kernel = session.get(KernelLibrary, kernel_id)
        if not kernel:
            return False

        if postprocessed_source is not None:
            kernel.postprocessed_source = postprocessed_source
        if python_registration is not None:
            kernel.python_registration = python_registration

        session.add(kernel)
        session.commit()
        return True


def invalidate_kernel(kernel_id: int, reason: Optional[str] = None) -> bool:
    """Mark a kernel as invalid. Admin use only."""
    with Session(engine) as session:
        kernel = session.get(KernelLibrary, kernel_id)
        if kernel:
            kernel.valid = False
            kernel.invalidation_reason = reason
            session.add(kernel)
            session.commit()
            return True
        return False


# ---------------------------------------------------------------------------
# Inliner query operations
# ---------------------------------------------------------------------------

def get_best_kernel_for_spec_json(
    spec_json: dict,
    gpu: str = "h100",
) -> Optional[tuple[KernelLibrary, Optional[EvalRun]]]:
    """Query the best kernel matching spec_json (PRIMARY lookup method).

    Searches both run-associated kernels (via EvalRun.spec_json) and
    external kernels (via KernelLibrary.spec_json).
    """
    with Session(engine) as session:
        # Query run-associated kernels
        run_stmt = (
            select(KernelLibrary, EvalRun)
            .join(EvalRun, KernelLibrary.run_id == EvalRun.run_id)
            .where(EvalRun.spec_json == spec_json)
            .where(EvalRun.gpu == gpu)
            .where(KernelLibrary.valid == True)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
            .limit(1)
        )
        run_result = session.exec(run_stmt).first()

        # Query external kernels (no run_id, spec on kernel directly)
        ext_stmt = (
            select(KernelLibrary)
            .where(KernelLibrary.run_id.is_(None))
            .where(KernelLibrary.spec_json == spec_json)
            .where(KernelLibrary.gpu == gpu)
            .where(KernelLibrary.valid == True)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
            .limit(1)
        )
        ext_result = session.exec(ext_stmt).first()

        # Return the one with higher speedup
        if run_result and ext_result:
            run_speedup = run_result[0].baseline_us / run_result[0].kernel_us
            ext_speedup = ext_result.baseline_us / ext_result.kernel_us
            if ext_speedup > run_speedup:
                return (ext_result, None)
            return run_result
        if ext_result:
            return (ext_result, None)
        return run_result


def get_best_kernel_by_function_and_scalars(
    function_name: str,
    scalar_args: dict,
    gpu: str = "h100",
) -> Optional[tuple[KernelLibrary, EvalRun]]:
    """Convenience lookup by function_name + scalar_args."""
    with Session(engine) as session:
        stmt = (
            select(KernelLibrary, EvalRun)
            .join(EvalRun, KernelLibrary.run_id == EvalRun.run_id)
            .where(EvalRun.function_name == function_name)
            .where(EvalRun.gpu == gpu)
            .where(KernelLibrary.valid == True)
            .where(cast(EvalRun.scalar_args, JSONB).op("@>")(cast(scalar_args, JSONB)))
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
            .limit(1)
        )
        result = session.exec(stmt).first()
        return result if result else None


def get_best_kernel_for_run(run_id: str) -> Optional[tuple[KernelLibrary, EvalRun]]:
    """Get the best kernel with its EvalRun for a specific run."""
    with Session(engine) as session:
        stmt = (
            select(KernelLibrary, EvalRun)
            .join(EvalRun, KernelLibrary.run_id == EvalRun.run_id)
            .where(KernelLibrary.run_id == run_id)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
            .limit(1)
        )
        return session.exec(stmt).first()


# Alias for leaderboard API
get_run_best = get_best_kernel_for_run


def get_kernels_for_run(run_id: str) -> list[KernelLibrary]:
    """Get all kernels for a run, ordered by speedup descending."""
    with Session(engine) as session:
        stmt = (
            select(KernelLibrary)
            .where(KernelLibrary.run_id == run_id)
            .where(KernelLibrary.valid == True)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
        )
        return list(session.exec(stmt).all())


def get_kernel_by_job_id(job_id: str) -> Optional[KernelLibrary]:
    """Get a kernel by its job_id."""
    with Session(engine) as session:
        stmt = select(KernelLibrary).where(KernelLibrary.job_id == job_id)
        return session.exec(stmt).first()


def get_all_agent_bests(run_id: str) -> list[KernelLibrary]:
    """Best kernel per agent_index for a given run."""
    with Session(engine) as session:
        stmt = (
            select(KernelLibrary)
            .where(KernelLibrary.run_id == run_id)
            .where(KernelLibrary.valid == True)
            .distinct(KernelLibrary.agent_index)
            .order_by(
                KernelLibrary.agent_index,
                (KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc(),
            )
        )
        return list(session.exec(stmt).all())


def insert_external_kernel(
    kernel_source: str,
    kernel_us: float,
    baseline_us: float,
    plugin: str,
    spec_json: dict,
    function_name: str,
    scalar_args: Optional[dict] = None,
    gpu: str = "h100",
    postprocessed_source: Optional[str] = None,
    python_registration: Optional[str] = None,
) -> KernelLibrary:
    """Insert an external kernel (not generated by the evaluation system).

    External kernels are standalone (no run_id) and store spec info directly.
    Uses job_id prefix 'ext-' to distinguish from system-generated kernels.
    """
    job_id = f"ext-{uuid4().hex}"
    with Session(engine) as session:
        kernel = KernelLibrary(
            job_id=job_id,
            run_id=None,
            agent_index=-1,
            kernel_source=kernel_source,
            kernel_us=kernel_us,
            baseline_us=baseline_us,
            gpu=gpu,
            postprocessed_source=postprocessed_source,
            python_registration=python_registration,
            plugin=plugin,
            spec_json=spec_json,
            function_name=function_name,
            scalar_args=scalar_args or {},
        )
        session.add(kernel)
        session.commit()
        session.refresh(kernel)
        return kernel


def list_external_kernels(
    plugin: Optional[str] = None,
    function_name: Optional[str] = None,
) -> list[KernelLibrary]:
    """List external kernels, optionally filtered by plugin or function_name."""
    with Session(engine) as session:
        stmt = (
            select(KernelLibrary)
            .where(KernelLibrary.run_id.is_(None))
            .where(KernelLibrary.valid == True)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
        )
        if plugin is not None:
            stmt = stmt.where(KernelLibrary.plugin == plugin)
        if function_name is not None:
            stmt = stmt.where(KernelLibrary.function_name == function_name)
        return list(session.exec(stmt).all())


def get_global_winners() -> list[tuple[KernelLibrary, EvalRun]]:
    """Best kernel per spec_json across all runs, filtered to speedup > 1.0."""
    with Session(engine) as session:
        stmt = (
            select(KernelLibrary, EvalRun)
            .join(EvalRun, KernelLibrary.run_id == EvalRun.run_id)
            .where(KernelLibrary.valid == True)
            .where(KernelLibrary.baseline_us / KernelLibrary.kernel_us > 1.0)
            .distinct(EvalRun.spec_json)
            .order_by(
                EvalRun.spec_json,
                (KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc(),
            )
        )
        return list(session.exec(stmt).all())


def get_best_similar_kernel(
    function_name: str,
    scalar_args: dict,
    gpu: str = "h100",
) -> Optional[tuple[KernelLibrary, EvalRun, str]]:
    """Find best similar kernel for starter code.

    Matching strategy:
    1. Try exact match on function_name + scalar_args
    2. Fall back to function_name only

    Returns:
        Tuple of (kernel, run, match_type) where match_type is
        "exact" or "function_only", or None if no match.
    """
    with Session(engine) as session:
        # Try exact match first
        exact_stmt = (
            select(KernelLibrary, EvalRun)
            .join(EvalRun, KernelLibrary.run_id == EvalRun.run_id)
            .where(EvalRun.function_name == function_name)
            .where(EvalRun.scalar_args == scalar_args)
            .where(EvalRun.gpu == gpu)
            .where(KernelLibrary.valid == True)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
            .limit(1)
        )
        exact_result = session.exec(exact_stmt).first()
        if exact_result:
            return (exact_result[0], exact_result[1], "exact")

        # Fall back to function_name only
        func_stmt = (
            select(KernelLibrary, EvalRun)
            .join(EvalRun, KernelLibrary.run_id == EvalRun.run_id)
            .where(EvalRun.function_name == function_name)
            .where(EvalRun.gpu == gpu)
            .where(KernelLibrary.valid == True)
            .order_by((KernelLibrary.baseline_us / KernelLibrary.kernel_us).desc())
            .limit(1)
        )
        func_result = session.exec(func_stmt).first()
        if func_result:
            return (func_result[0], func_result[1], "function_only")

        return None
