import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import torch

from kernel_evaluator.services.jobs import EvaluationJob


def _default_num_gpus() -> int:
    visible = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _default_gpu_tokens() -> list[str]:
    visible = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return [str(index) for index in range(_default_num_gpus())]


@dataclass
class EvaluationQueueState:
    compile_workers: int = 20
    num_gpus: int = field(default_factory=_default_num_gpus)
    gpu_tokens: list[str] = field(default_factory=_default_gpu_tokens)
    artifact_root: Path = Path("/tmp/kernel_evaluator")
    terminal_job_ttl_s: float = 3600.0
    cleanup_interval_s: float = 60.0
    jobs: dict[str, EvaluationJob] = field(default_factory=dict)
    compile_queue: asyncio.Queue | None = None
    benchmark_queues: list[asyncio.Queue] = field(default_factory=list)
    gpu_busy: list[bool] = field(default_factory=list)
    compile_executor: ThreadPoolExecutor | None = None
    benchmark_executor: ThreadPoolExecutor | None = None
    worker_tasks: list[asyncio.Task] = field(default_factory=list)
    started: bool = False

    def configure(
        self,
        compile_workers: int | None = None,
        num_gpus: int | None = None,
        gpu_tokens: list[str] | None = None,
        artifact_root: Path | None = None,
        terminal_job_ttl_s: float | None = None,
        cleanup_interval_s: float | None = None,
    ) -> None:
        if compile_workers is not None:
            self.compile_workers = compile_workers
        if num_gpus is not None:
            self.num_gpus = num_gpus
            if gpu_tokens is None:
                self.gpu_tokens = [str(index) for index in range(num_gpus)]
        if gpu_tokens is not None:
            if len(gpu_tokens) != self.num_gpus:
                raise ValueError("gpu_tokens length must match num_gpus")
            self.gpu_tokens = gpu_tokens
        if artifact_root is not None:
            self.artifact_root = artifact_root
        if terminal_job_ttl_s is not None:
            self.terminal_job_ttl_s = terminal_job_ttl_s
        if cleanup_interval_s is not None:
            self.cleanup_interval_s = cleanup_interval_s

    def initialize(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.compile_queue = asyncio.Queue()
        self.benchmark_queues = [asyncio.Queue() for _ in range(self.num_gpus)]
        self.gpu_busy = [False] * self.num_gpus
        self.compile_executor = ThreadPoolExecutor(max_workers=self.compile_workers)
        self.benchmark_executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        self.started = True

    def require_started(self) -> None:
        if not self.started or self.compile_queue is None:
            raise RuntimeError("evaluation queue state is not started")

    def pick_gpu(self) -> int:
        return min(
            range(self.num_gpus),
            key=lambda index: self.benchmark_queues[index].qsize() + (1 if self.gpu_busy[index] else 0),
        )

    def visible_gpu_token(self, gpu_id: int) -> str:
        return self.gpu_tokens[gpu_id]

    def cleanup_expired_jobs(self, now: float | None = None) -> int:
        current_time = monotonic() if now is None else now
        expired_job_ids = [
            job_id
            for job_id, job in self.jobs.items()
            if job.terminal() and current_time - job.updated_at >= self.terminal_job_ttl_s
        ]
        for job_id in expired_job_ids:
            job = self.jobs.pop(job_id)
            if job.artifact_dir is not None:
                shutil.rmtree(job.artifact_dir, ignore_errors=True)
        return len(expired_job_ids)

    def shutdown(self) -> None:
        for task in self.worker_tasks:
            task.cancel()
        if self.compile_executor is not None:
            self.compile_executor.shutdown(wait=False, cancel_futures=True)
        if self.benchmark_executor is not None:
            self.benchmark_executor.shutdown(wait=False, cancel_futures=True)
        self.started = False


queue_state = EvaluationQueueState()
