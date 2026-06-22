from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from kernel_evaluator.db import ops as db_ops
from kernel_evaluator.services.api_keys import ApiPrincipal, ApiRole, require_api_key

scaffold_router = APIRouter(prefix="/scaffold", tags=["scaffold"])


# --- Starter code ---

@scaffold_router.get("/starter")
async def get_starter(run_id: str):
    """Get best similar kernel for starter code (best-similar mode)."""
    run = db_ops.get_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    result = db_ops.get_best_similar_kernel(
        function_name=run.function_name,
        scalar_args=run.scalar_args,
        gpu=run.gpu,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="no similar kernel found")

    kernel, matched_run, match_type = result
    speedup = kernel.baseline_us / kernel.kernel_us if kernel.kernel_us > 0 else 0

    return {
        "source": kernel.kernel_source,
        "match_type": match_type,
        "matched_function": matched_run.function_name,
        "matched_scalars": matched_run.scalar_args if match_type == "exact" else None,
        "speedup": speedup,
        "kernel_id": kernel.id,
        "job_id": kernel.job_id,
    }


# --- Leaderboard ---
def _overhead_pct(run_id: str, principal: ApiPrincipal) -> float:
    if principal.role != ApiRole.USER:
        return 0.0
    run = db_ops.get_eval_run(run_id)
    return run.overhead_pct if run else 0.0


@scaffold_router.get("/best")
async def get_best(run_id: str, principal: ApiPrincipal = Depends(require_api_key)):
    result = db_ops.get_run_best(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no best kernel yet")
    kernel, run = result
    kernel_us = kernel.kernel_us * (1 + _overhead_pct(run_id, principal) / 100)
    return {
        "kernel_us": kernel_us,
        "baseline_us": kernel.baseline_us,
        "speedup": kernel.baseline_us / kernel_us,
        "job_id": kernel.job_id,
        "agent_index": kernel.agent_index,
        "source": kernel.kernel_source,
    }


@scaffold_router.get("/archive")
async def get_archive(run_id: str, principal: ApiPrincipal = Depends(require_api_key)):
    entries = db_ops.get_kernels_for_run(run_id)
    factor = 1 + _overhead_pct(run_id, principal) / 100
    return {
        "run_id": run_id,
        "entries": [
            {
                "job_id": e.job_id,
                "agent_index": e.agent_index,
                "kernel_us": e.kernel_us * factor,
                "baseline_us": e.baseline_us,
                "speedup": e.baseline_us / (e.kernel_us * factor),
                "achieved_at": e.achieved_at.isoformat(),
            }
            for e in entries
        ],
    }


@scaffold_router.get("/kernel-source/{job_id}", response_class=PlainTextResponse)
async def get_kernel_source(job_id: str):
    entry = db_ops.get_kernel_by_job_id(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="job not found in archive")
    return entry.kernel_source


@scaffold_router.get("/agent-bests")
async def get_agent_bests(run_id: str, principal: ApiPrincipal = Depends(require_api_key)):
    entries = db_ops.get_all_agent_bests(run_id)
    factor = 1 + _overhead_pct(run_id, principal) / 100
    return {
        "run_id": run_id,
        "agent_bests": [
            {
                "agent_index": e.agent_index,
                "kernel_us": e.kernel_us * factor,
                "baseline_us": e.baseline_us,
                "speedup": e.baseline_us / (e.kernel_us * factor),
                "job_id": e.job_id,
            }
            for e in entries
        ],
    }
