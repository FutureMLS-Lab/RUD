import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import torch

from kernel_evaluator.services.artifact_requests import ProducedArtifact, RequestedArtifactKind, artifact_content_type
from kernel_evaluator.services.compile_subprocess import PACKAGE_MANIFEST
from kernel_evaluator.services.evaluation import (
    Artifact,
    ArtifactKind,
    BenchmarkController,
    BenchmarkPolicy,
    BuildContext,
    CandidateSubmission,
    ReferenceTask,
    default_registry,
)
from kernel_evaluator.services.evaluation.task_factory import reference_task_from_spec
from kernel_evaluator.services.evaluation.types import BenchmarkRepeat, BenchmarkResult, CandidateKind, TimingMode
from kernel_evaluator.services.evaluation.types import ShapeBenchmarkResult
from kernel_evaluator.services.jobs import EvaluationJob


PUBLIC_ARTIFACT_KINDS = {
    ArtifactKind.CUBIN: RequestedArtifactKind.CUBIN,
    ArtifactKind.PTX: RequestedArtifactKind.PTX,
    ArtifactKind.RESOURCE_USAGE: RequestedArtifactKind.RESOURCE_USAGE,
}
PROFILE_ARTIFACT_KINDS = {
    RequestedArtifactKind.NCU_REPORT,
    RequestedArtifactKind.NCU_SUMMARY,
    RequestedArtifactKind.ROCPROF_REPORT,
    RequestedArtifactKind.ROCPROF_SUMMARY,
}

COMPILE_TIMEOUT_S = float(os.environ["KERNEL_EVALUATOR_COMPILE_TIMEOUT_S"])
BENCHMARK_TIMEOUT_S = float(os.environ["KERNEL_EVALUATOR_BENCHMARK_TIMEOUT_S"])
PROFILE_TIMEOUT_S = BENCHMARK_TIMEOUT_S


class ExecutionBackend(Protocol):
    def compile(self, job: EvaluationJob) -> None:
        ...

    def benchmark(self, job: EvaluationJob, gpu_id: int) -> None:
        ...

    def profile(self, job: EvaluationJob, gpu_id: int) -> None:
        ...


def _artifact_source(job: EvaluationJob) -> Path:
    if job.artifact_dir is None:
        raise RuntimeError("job artifact_dir is not set")
    if job.candidate_kind == CandidateKind.CUDA_SOURCE:
        suffix = ".cu"
    elif job.candidate_kind == CandidateKind.HIP_SOURCE:
        suffix = ".cpp"
    else:
        suffix = ".py"
    target = job.artifact_dir / f"source{suffix}"
    target.write_text(job.source_text, encoding="utf-8")
    return target


def _reference_task_from_shape(shape_contract: dict) -> ReferenceTask:
    task = reference_task_from_spec(
        shape_contract["spec"],
        shape_contract["scalars"],
        shape_contract["dtype"],
        shape_contract["task_slug"],
    )
    return task


def _benchmark_policy(job: EvaluationJob) -> BenchmarkPolicy:
    policy = job.benchmark_policy
    return BenchmarkPolicy(
        timing_mode=TimingMode(policy["timing_mode"]),
        warmup=int(policy["warmup"]),
        iterations=int(policy["iterations"]),
        graph_calls=int(policy["graph_calls"]),
        repeats=int(policy["repeats"]),
        sleep_s=float(policy.get("sleep_s", 10.0)),
        clear_outputs_after_prepare=bool(policy["clear_outputs_after_prepare"]),
    )


def _submission(job: EvaluationJob, task: ReferenceTask, source: Path) -> CandidateSubmission:
    artifacts = (Artifact(ArtifactKind.SOURCE, source),)
    metadata = {"scalars": job.benchmark_shapes[0]["scalars"]}
    return CandidateSubmission(
        kind=job.candidate_kind,
        abi=task.abi,
        artifacts=artifacts,
        source=str(source) if job.candidate_kind in (CandidateKind.CUDA_SOURCE, CandidateKind.HIP_SOURCE) else None,
        entrypoint=job.entrypoint,
        metadata=metadata,
    )


def _artifact_path(root: Path, rel_path: str) -> Path:
    path = (root / rel_path).resolve()
    path.relative_to(root.resolve())
    return path


def _register_artifact(job: EvaluationJob, kind: RequestedArtifactKind, path: Path) -> None:
    if kind in job.requested_artifacts:
        job.produced_artifacts[kind] = ProducedArtifact(kind=kind, path=path, content_type=artifact_content_type(kind))


def _register_built_artifacts(job: EvaluationJob, artifacts: tuple[Artifact, ...]) -> None:
    for artifact in artifacts:
        if artifact.kind in PUBLIC_ARTIFACT_KINDS:
            _register_artifact(job, PUBLIC_ARTIFACT_KINDS[artifact.kind], artifact.path)


def _register_package_artifacts(job: EvaluationJob) -> None:
    if job.artifact_dir is None:
        raise RuntimeError("job artifact_dir is not set")
    manifest = json.loads((job.artifact_dir / PACKAGE_MANIFEST).read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        kind = ArtifactKind(item["kind"])
        if kind in PUBLIC_ARTIFACT_KINDS:
            _register_artifact(job, PUBLIC_ARTIFACT_KINDS[kind], _artifact_path(job.artifact_dir, item["path"]))


def _register_profile_artifacts(job: EvaluationJob, artifacts: list[dict]) -> None:
    if job.artifact_dir is None:
        raise RuntimeError("job artifact_dir is not set")
    for item in artifacts:
        kind = RequestedArtifactKind(item["kind"])
        if kind in PROFILE_ARTIFACT_KINDS:
            _register_artifact(job, kind, _artifact_path(job.artifact_dir, item["path"]))


def _run_subprocess(cmd: list[str], env: dict[str, str], timeout_s: float, phase: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        output = (stderr + stdout).strip()
        message = f"{phase} subprocess timed out after {timeout_s:.1f}s"
        if output:
            message = f"{message}\n{output}"
        raise RuntimeError(message) from exc


class LocalExecutionBackend:
    def compile(self, job: EvaluationJob) -> None:
        if job.artifact_dir is None:
            raise RuntimeError("job artifact_dir is not set")
        source = _artifact_source(job)
        task = _reference_task_from_shape(job.benchmark_shapes[0])
        submission = _submission(job, task, source)
        context = BuildContext(work_dir=job.artifact_dir, cuda_arch=job.cuda_arch)
        job.built_candidate = BenchmarkController(
            registry=default_registry(),
            policy=BenchmarkPolicy(),
        ).build(submission, context)
        _register_built_artifacts(job, job.built_candidate.artifacts)

    def benchmark(self, job: EvaluationJob, gpu_id: int) -> None:
        if job.built_candidate is None:
            raise RuntimeError("job has no compiled candidate")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        controller = BenchmarkController(registry=default_registry(), policy=_benchmark_policy(job))
        shape_results = []
        for shape_contract in job.benchmark_shapes:
            task = _reference_task_from_shape(shape_contract)
            result = controller.benchmark(task, job.built_candidate)
            shape_results.append(
                ShapeBenchmarkResult(
                    shape=shape_contract["shape"],
                    task_slug=shape_contract["task_slug"],
                    repeats=result.repeats,
                )
            )
        job.result = BenchmarkResult(shape_results=tuple(shape_results))

    def profile(self, job: EvaluationJob, gpu_id: int) -> None:
        del job, gpu_id


class SubprocessExecutionBackend:
    def compile(self, job: EvaluationJob) -> None:
        if job.artifact_dir is None:
            raise RuntimeError("job artifact_dir is not set")
        _artifact_source(job)
        job_json = job.artifact_dir / "compile_job.json"
        result_json = job.artifact_dir / "compile_result.json"
        job_json.write_text(json.dumps(job.subprocess_payload(), indent=2, sort_keys=True), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "kernel_evaluator.services.compile_subprocess",
            "--job-json",
            str(job_json),
            "--result-json",
            str(result_json),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{job.artifact_dir}{os.pathsep}{env['PYTHONPATH']}" if "PYTHONPATH" in env else str(job.artifact_dir)
        completed = _run_subprocess(cmd, env, COMPILE_TIMEOUT_S, "compile")
        if not result_json.exists():
            output = (completed.stderr + completed.stdout).strip()
            raise RuntimeError(output or f"compile subprocess failed with exit code {completed.returncode}")
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        if not payload["ok"]:
            raise RuntimeError(payload["error"])
        _register_package_artifacts(job)

    def benchmark(self, job: EvaluationJob, gpu_id: int) -> None:
        if job.artifact_dir is None:
            raise RuntimeError("job artifact_dir is not set")
        job_json = job.artifact_dir / "job.json"
        result_json = job.artifact_dir / "result.json"
        job_json.write_text(json.dumps(job.subprocess_payload(), indent=2, sort_keys=True), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "kernel_evaluator.services.benchmark_subprocess",
            "--job-json",
            str(job_json),
            "--result-json",
            str(result_json),
            "--gpu-id",
            str(gpu_id),
        ]
        env = os.environ.copy()
        if job.assigned_gpu_token is None:
            raise RuntimeError("job assigned_gpu_token is not set")
        env["CUDA_VISIBLE_DEVICES"] = job.assigned_gpu_token
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env["PYTHONPATH"] = f"{job.artifact_dir}{os.pathsep}{env['PYTHONPATH']}" if "PYTHONPATH" in env else str(job.artifact_dir)
        cmd[-1] = "0"
        completed = _run_subprocess(cmd, env, BENCHMARK_TIMEOUT_S, "benchmark")
        if not result_json.exists():
            output = (completed.stderr + completed.stdout).strip()
            raise RuntimeError(output or f"benchmark subprocess failed with exit code {completed.returncode}")
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        if not payload["ok"]:
            raise RuntimeError(payload["error"])
        shape_results = tuple(
            ShapeBenchmarkResult(
                shape=shape_result["shape"],
                task_slug=shape_result["task_slug"],
                repeats=tuple(
                    BenchmarkRepeat(
                        seed=repeat["seed"],
                        baseline_us=repeat["baseline_us"],
                        candidate_us=repeat["candidate_us"],
                        correct=repeat["correct"],
                    )
                    for repeat in shape_result["repeats"]
                ),
            )
            for shape_result in payload["result"]["shape_results"]
        )
        job.result = BenchmarkResult(shape_results=shape_results)

    def profile(self, job: EvaluationJob, gpu_id: int) -> None:
        if job.artifact_dir is None:
            raise RuntimeError("job artifact_dir is not set")
        job_json = job.artifact_dir / "profile_job.json"
        result_json = job.artifact_dir / "profile_result.json"
        job_json.write_text(json.dumps(job.subprocess_payload(), indent=2, sort_keys=True), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "kernel_evaluator.services.profile_subprocess",
            "--job-json",
            str(job_json),
            "--result-json",
            str(result_json),
            "--gpu-id",
            str(gpu_id),
            "--timeout-s",
            str(PROFILE_TIMEOUT_S),
        ]
        env = os.environ.copy()
        if job.assigned_gpu_token is None:
            raise RuntimeError("job assigned_gpu_token is not set")
        env["CUDA_VISIBLE_DEVICES"] = job.assigned_gpu_token
        env["PYTHONPATH"] = f"{job.artifact_dir}{os.pathsep}{env['PYTHONPATH']}" if "PYTHONPATH" in env else str(job.artifact_dir)
        cmd[cmd.index("--gpu-id") + 1] = "0"
        completed = _run_subprocess(cmd, env, PROFILE_TIMEOUT_S, "profile")
        if not result_json.exists():
            output = (completed.stderr + completed.stdout).strip()
            raise RuntimeError(output or f"profile subprocess failed with exit code {completed.returncode}")
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        if not payload["ok"]:
            raise RuntimeError(payload["error"])
        _register_profile_artifacts(job, payload["result"]["artifacts"])
