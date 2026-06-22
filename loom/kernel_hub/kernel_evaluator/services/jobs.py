from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any

from kernel_evaluator.services.artifact_requests import ProducedArtifact, RequestedArtifactKind
from kernel_evaluator.services.evaluation.types import BenchmarkResult, BuiltCandidate, CandidateKind


class EvaluationJobState(StrEnum):
    QUEUED_COMPILE = "queued_compile"
    COMPILING = "compiling"
    COMPILE_FAILED = "compile_failed"
    QUEUED_BENCHMARK = "queued_benchmark"
    BENCHMARKING = "benchmarking"
    BENCHMARK_FAILED = "benchmark_failed"
    COMPLETED = "completed"


TERMINAL_STATES = {
    EvaluationJobState.COMPILE_FAILED,
    EvaluationJobState.BENCHMARK_FAILED,
    EvaluationJobState.COMPLETED,
}


@dataclass
class EvaluationJob:
    job_id: str
    candidate_kind: CandidateKind
    source_text: str
    benchmark_shapes: list[dict[str, Any]]
    run_id: str = ""
    owner_key_id: str | None = None
    requested_artifacts: tuple[RequestedArtifactKind, ...] = ()
    produced_artifacts: dict[RequestedArtifactKind, ProducedArtifact] = field(default_factory=dict)
    entrypoint: str | None = None
    benchmark_policy: dict[str, Any] = field(default_factory=dict)
    cuda_arch: str = "90a"
    agent_index: int = 0
    state: EvaluationJobState = EvaluationJobState.QUEUED_COMPILE
    created_at: float = field(default_factory=monotonic)
    updated_at: float = field(default_factory=monotonic)
    artifact_dir: Path | None = None
    assigned_gpu_id: int | None = None
    assigned_gpu_token: str | None = None
    compile_error: str | None = None
    benchmark_error: str | None = None
    profile_error: str | None = None
    built_candidate: BuiltCandidate | None = field(default=None, repr=False)
    result: BenchmarkResult | None = None

    def set_state(self, state: EvaluationJobState) -> None:
        self.state = state
        self.updated_at = monotonic()

    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def requests_profile(self) -> bool:
        return any(
            kind in self.requested_artifacts
            for kind in (
                RequestedArtifactKind.NCU_REPORT,
                RequestedArtifactKind.NCU_SUMMARY,
                RequestedArtifactKind.ROCPROF_REPORT,
                RequestedArtifactKind.ROCPROF_SUMMARY,
            )
        )

    def summary(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "state": str(self.state),
            "candidate_kind": str(self.candidate_kind),
            "owner_key_id": self.owner_key_id,
            "agent_index": self.agent_index,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir is not None else None,
            "artifacts": [str(kind) for kind in self.produced_artifacts],
            "benchmark_shapes": self.benchmark_shapes,
            "assigned_gpu_id": self.assigned_gpu_id,
            "assigned_gpu_token": self.assigned_gpu_token,
            "compile_error": self.compile_error,
            "benchmark_error": self.benchmark_error,
            "profile_error": self.profile_error,
        }
        if self.result is not None:
            data["correct"] = self.result.correct
            data["baseline_us"] = self.result.baseline_us
            data["candidate_us"] = self.result.candidate_us
            data["speedup"] = self.result.speedup
            data["repeats"] = [
                {
                    "seed": repeat.seed,
                    "baseline_us": repeat.baseline_us,
                    "candidate_us": repeat.candidate_us,
                    "correct": repeat.correct,
                }
                for repeat in self.result.repeats
            ]
            data["shape_results"] = [
                {
                    "shape": dict(shape_result.shape),
                    "task_slug": shape_result.task_slug,
                    "correct": shape_result.correct,
                    "baseline_us": shape_result.baseline_us,
                    "candidate_us": shape_result.candidate_us,
                    "speedup": shape_result.speedup,
                    "repeats": [
                        {
                            "seed": repeat.seed,
                            "baseline_us": repeat.baseline_us,
                            "candidate_us": repeat.candidate_us,
                            "correct": repeat.correct,
                        }
                        for repeat in shape_result.repeats
                    ],
                }
                for shape_result in self.result.shape_results
            ]
        return data

    def subprocess_payload(self) -> dict[str, Any]:
        if self.artifact_dir is None:
            raise RuntimeError("job artifact_dir is not set")
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "candidate_kind": str(self.candidate_kind),
            "source_text": self.source_text,
            "benchmark_shapes": self.benchmark_shapes,
            "requested_artifacts": [str(kind) for kind in self.requested_artifacts],
            "entrypoint": self.entrypoint,
            "benchmark_policy": self.benchmark_policy,
            "cuda_arch": self.cuda_arch,
            "artifact_dir": str(self.artifact_dir),
        }
