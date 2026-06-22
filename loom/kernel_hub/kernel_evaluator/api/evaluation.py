from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kernel_evaluator.db.models import EvalRun
from kernel_evaluator.db.ops import (
    create_eval_run_from_contract,
    finish_eval_run,
    get_best_kernel_for_run,
    get_eval_run,
    list_eval_runs,
)
from kernel_evaluator.services.artifact_requests import RequestedArtifactKind, parse_requested_artifacts
from kernel_evaluator.services.api_keys import (
    ApiPrincipal,
    ApiRole,
    check_in_flight_submission_limit,
    require_admin_key,
    require_api_key,
    require_submission_key,
)
from kernel_evaluator.services.evaluation.types import CandidateKind
from kernel_evaluator.services.jobs import EvaluationJob, EvaluationJobState
from kernel_evaluator.services.queue_state import queue_state
from kernel_evaluator.services.plugins import make_plugin_run
from kernel_evaluator.services.run_contracts import BenchmarkPolicyModel, validate_run_contract
from kernel_evaluator.services.workers import start_workers

evaluation_router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _user_job_summary(summary: dict, run_id: str) -> dict:
    if "candidate_us" not in summary:
        return summary
    run = get_eval_run(run_id)
    overhead_pct = run.overhead_pct if run else 0.0
    candidate_us = summary["candidate_us"] * (1 + overhead_pct / 100)
    return {
        **{k: v for k, v in summary.items() if k not in ("shape_results", "repeats")},
        "candidate_us": candidate_us,
        "speedup": summary["baseline_us"] / candidate_us,
    }


class EvaluationRunRequest(BaseModel):
    plugin: str
    target: str
    shapes: list[dict]
    target_speedup: Optional[float] = None
    cuda_arch: str = "90a"
    benchmark_policy: Optional[BenchmarkPolicyModel] = None


class EvaluationSubmitRequest(BaseModel):
    source_text: str
    artifacts: list[str] = Field(default_factory=list)
    agent_index: Optional[int] = None


def _job_or_404(job_id: str) -> EvaluationJob:
    try:
        return queue_state.jobs[job_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


def _visible_job_or_404(job_id: str, principal: ApiPrincipal) -> EvaluationJob:
    job = _job_or_404(job_id)
    if principal.role == ApiRole.ADMIN:
        return job
    if job.owner_key_id != principal.key_id:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _run_from_record(record: EvalRun) -> dict:
    return validate_run_contract(record.run_contract)


def _run_or_404(run_id: str) -> dict:
    record = get_eval_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_from_record(record)


def _artifact_or_404(job: EvaluationJob, artifact_kind: str):
    try:
        kind = RequestedArtifactKind(artifact_kind)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    try:
        artifact = job.produced_artifacts[kind]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    if not artifact.path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return artifact


def _ensure_workers_started() -> None:
    if not queue_state.started:
        start_workers(queue_state)


def _in_flight_jobs_for_key(key_id: str) -> int:
    return sum(
        1
        for job in queue_state.jobs.values()
        if job.owner_key_id == key_id and not job.terminal()
    )


@evaluation_router.post("/runs")
def create_evaluation_run(req: EvaluationRunRequest, _principal: ApiPrincipal = Depends(require_admin_key)):
    try:
        raw_run = make_plugin_run(req.plugin, req.target, req.shapes, req.cuda_arch)
        if req.target_speedup is not None:
            raw_run["target_speedup"] = req.target_speedup
        if req.benchmark_policy is not None:
            raw_run["benchmark_policy"] = {
                **raw_run.get("benchmark_policy", {}),
                **req.benchmark_policy.model_dump(exclude_unset=True, mode="json"),
            }
        run = validate_run_contract(raw_run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stored = create_eval_run_from_contract(run)
    return _run_from_record(stored)


@evaluation_router.get("/runs")
def list_evaluation_runs(
    reference_plugin: Optional[str] = None,
    _principal: ApiPrincipal = Depends(require_api_key),
):
    runs = list_eval_runs(reference_plugin)
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "reference_plugin": r.reference_plugin,
                "function_name": r.function_name,
                "gpu": r.gpu,
                "model": r.model,
                "started_at": r.started_at.isoformat(),
                "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            }
            for r in runs
        ]
    }


@evaluation_router.get("/runs/{run_id}")
def get_evaluation_run(run_id: str, principal: ApiPrincipal = Depends(require_api_key)):
    run = _run_or_404(run_id)
    run["jobs"] = [
        job.summary()
        for job in queue_state.jobs.values()
        if job.run_id == run_id
        and (principal.role == ApiRole.ADMIN or job.owner_key_id == principal.key_id)
    ]
    return run


@evaluation_router.post("/runs/{run_id}/finish")
def finish_evaluation_run(run_id: str, _principal: ApiPrincipal = Depends(require_api_key)):
    record = get_eval_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    finish_eval_run(run_id)
    return {"ok": True}


@evaluation_router.get("/runs/{run_id}/best-kernel")
def get_run_best_kernel(run_id: str, _principal: ApiPrincipal = Depends(require_api_key)):
    record = get_eval_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    result = get_best_kernel_for_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no best kernel yet")
    kernel, run = result
    return {
        "id": kernel.id,
        "job_id": kernel.job_id,
        "run_id": kernel.run_id,
        "agent_index": kernel.agent_index,
        "kernel_us": kernel.kernel_us,
        "baseline_us": kernel.baseline_us,
        "speedup": kernel.baseline_us / kernel.kernel_us if kernel.kernel_us > 0 else 0,
        "kernel_source": kernel.kernel_source,
        "postprocessed_source": kernel.postprocessed_source,
        "python_registration": kernel.python_registration,
        "gpu": kernel.gpu,
        "achieved_at": kernel.achieved_at.isoformat() if kernel.achieved_at else None,
        "valid": kernel.valid,
        "invalidation_reason": kernel.invalidation_reason,
        # From EvalRun
        "function_name": run.function_name,
        "scalar_args": run.scalar_args,
        "spec_json": run.spec_json,
        "plugin": run.plugin,
    }


@evaluation_router.post("/runs/{run_id}/jobs")
async def submit_evaluation_job(
    run_id: str,
    req: EvaluationSubmitRequest,
    principal: ApiPrincipal = Depends(require_submission_key),
):
    run = _run_or_404(run_id)
    try:
        requested_artifacts = parse_requested_artifacts(req.artifacts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _ensure_workers_started()
    check_in_flight_submission_limit(principal, _in_flight_jobs_for_key(principal.key_id))
    job_id = uuid4().hex
    job = EvaluationJob(
        job_id=job_id,
        run_id=run_id,
        candidate_kind=CandidateKind(run["candidate_kind"]),
        source_text=req.source_text,
        benchmark_shapes=run["benchmark_shapes"],
        owner_key_id=principal.key_id,
        requested_artifacts=requested_artifacts,
        entrypoint=run["entrypoint"],
        benchmark_policy=run["benchmark_policy"],
        cuda_arch=run["cuda_arch"],
        agent_index=req.agent_index,
    )
    queue_state.jobs[job_id] = job
    await queue_state.compile_queue.put(job)
    return job.summary()


@evaluation_router.get("/jobs/{job_id}")
def get_evaluation_job(job_id: str, principal: ApiPrincipal = Depends(require_api_key)):
    job = _visible_job_or_404(job_id, principal)
    summary = job.summary()
    return _user_job_summary(summary, job.run_id) if principal.role == ApiRole.USER else summary


@evaluation_router.get("/jobs/{job_id}/result")
def get_evaluation_result(job_id: str, principal: ApiPrincipal = Depends(require_api_key)):
    job = _visible_job_or_404(job_id, principal)
    if job.state != EvaluationJobState.COMPLETED:
        raise HTTPException(status_code=409, detail=job.summary())
    summary = job.summary()
    return _user_job_summary(summary, job.run_id) if principal.role == ApiRole.USER else summary


@evaluation_router.get("/jobs/{job_id}/artifacts/{artifact_kind}")
def get_evaluation_artifact(
    job_id: str,
    artifact_kind: str,
    principal: ApiPrincipal = Depends(require_api_key),
):
    job = _visible_job_or_404(job_id, principal)
    artifact = _artifact_or_404(job, artifact_kind)
    return FileResponse(artifact.path, media_type=artifact.content_type, filename=artifact.path.name)
