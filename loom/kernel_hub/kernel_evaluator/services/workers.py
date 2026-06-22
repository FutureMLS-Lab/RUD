import asyncio
import tempfile
from pathlib import Path

from kernel_evaluator.services.execution import ExecutionBackend, SubprocessExecutionBackend
from kernel_evaluator.services.jobs import EvaluationJob, EvaluationJobState
from kernel_evaluator.services.queue_state import EvaluationQueueState, queue_state


async def compile_worker(state: EvaluationQueueState, backend: ExecutionBackend) -> None:
    state.require_started()
    loop = asyncio.get_running_loop()
    while True:
        job: EvaluationJob = await state.compile_queue.get()
        try:
            job.set_state(EvaluationJobState.COMPILING)
            job.artifact_dir = Path(tempfile.mkdtemp(
                prefix=f"job_{job.job_id}_",
                dir=state.artifact_root,
            ))
            await loop.run_in_executor(state.compile_executor, backend.compile, job)
            gpu_id = state.pick_gpu()
            job.assigned_gpu_id = gpu_id
            job.assigned_gpu_token = state.visible_gpu_token(gpu_id)
            job.set_state(EvaluationJobState.QUEUED_BENCHMARK)
            await state.benchmark_queues[gpu_id].put(job)
        except Exception as exc:
            job.compile_error = str(exc)
            job.set_state(EvaluationJobState.COMPILE_FAILED)
        finally:
            state.compile_queue.task_done()


def _persist_completed_kernel(job: EvaluationJob) -> None:
    if job.result is None:
        return
    if not job.result.correct:
        return
    if job.run_id == "":
        return
    from kernel_evaluator.db.ops import insert_kernel

    insert_kernel(
        job_id=job.job_id,
        run_id=job.run_id,
        kernel_source=job.source_text,
        kernel_us=job.result.candidate_us,
        baseline_us=job.result.baseline_us,
        agent_index=job.agent_index,
    )


async def benchmark_worker(state: EvaluationQueueState, backend: ExecutionBackend, gpu_id: int) -> None:
    state.require_started()
    loop = asyncio.get_running_loop()
    queue = state.benchmark_queues[gpu_id]
    while True:
        job: EvaluationJob = await queue.get()
        state.gpu_busy[gpu_id] = True
        try:
            job.assigned_gpu_id = gpu_id
            job.assigned_gpu_token = state.visible_gpu_token(gpu_id)
            job.set_state(EvaluationJobState.BENCHMARKING)
            await loop.run_in_executor(state.benchmark_executor, backend.benchmark, job, gpu_id)
            if job.result is not None and job.result.correct and job.requests_profile():
                try:
                    await loop.run_in_executor(state.benchmark_executor, backend.profile, job, gpu_id)
                except Exception as exc:
                    job.profile_error = str(exc)
            _persist_completed_kernel(job)
            job.set_state(EvaluationJobState.COMPLETED)
        except Exception as exc:
            job.benchmark_error = str(exc)
            job.set_state(EvaluationJobState.BENCHMARK_FAILED)
        finally:
            state.gpu_busy[gpu_id] = False
            queue.task_done()


async def cleanup_worker(state: EvaluationQueueState) -> None:
    state.require_started()
    while True:
        await asyncio.sleep(state.cleanup_interval_s)
        state.cleanup_expired_jobs()


def start_workers(state: EvaluationQueueState = queue_state, backend: ExecutionBackend | None = None) -> None:
    if state.started:
        return
    selected_backend = backend if backend is not None else SubprocessExecutionBackend()
    state.initialize()
    for _ in range(state.compile_workers):
        state.worker_tasks.append(asyncio.create_task(compile_worker(state, selected_backend)))
    for gpu_id in range(state.num_gpus):
        state.worker_tasks.append(asyncio.create_task(benchmark_worker(state, selected_backend, gpu_id)))
    state.worker_tasks.append(asyncio.create_task(cleanup_worker(state)))


def stop_workers(state: EvaluationQueueState = queue_state) -> None:
    state.shutdown()
