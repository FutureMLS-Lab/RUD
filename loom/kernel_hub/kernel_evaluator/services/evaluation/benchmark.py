import time
from dataclasses import dataclass

import torch

from kernel_evaluator.services.evaluation.registry import BuilderRegistry
from kernel_evaluator.services.evaluation.types import (
    BenchmarkPolicy,
    BenchmarkRepeat,
    BenchmarkResult,
    BuildContext,
    BuiltCandidate,
    CandidateSubmission,
    ExecutionInputs,
    ReferenceTask,
    TimingMode,
)


def _synchronize() -> None:
    torch.cuda.synchronize()


def _median_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    _synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for i in range(iterations):
        starts[i].record()
        fn()
        ends[i].record()
    _synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) * 1000 for i in range(iterations))
    return times[len(times) // 2]


def _graph_median_us(fn, warmup: int, iterations: int, graph_calls: int) -> float:
    stream = torch.cuda.Stream()
    _synchronize()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(graph_calls):
                fn()
        stream.synchronize()
        for _ in range(warmup):
            graph.replay()
        stream.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for i in range(iterations):
            starts[i].record(stream)
            graph.replay()
            ends[i].record(stream)
        stream.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) * 1000 / graph_calls for i in range(iterations))
    return times[len(times) // 2]


_FLUSH_BYTES = 256 * 1024 * 1024
_flush_buf: torch.Tensor | None = None


def _get_flush_buf() -> torch.Tensor:
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    return _flush_buf


def _flushed_median_us(fn, warmup: int, iterations: int) -> float:
    buf = _get_flush_buf()
    for _ in range(warmup):
        buf.zero_()
        _synchronize()
        fn()
    _synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for i in range(iterations):
        buf.zero_()
        _synchronize()
        starts[i].record()
        fn()
        ends[i].record()
    _synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) * 1000 for i in range(iterations))
    return times[len(times) // 2]


def _clear_outputs(inputs: ExecutionInputs, output_names: tuple[str, ...]) -> None:
    for name in output_names:
        inputs.tensors[name].zero_()


def _outputs_close(
    inputs: ExecutionInputs,
    expected,
    output_names: tuple[str, ...],
    tolerances: tuple[float, float],
) -> bool:
    rtol, atol = tolerances
    return all(
        torch.allclose(inputs.tensors[name], expected[name], rtol=rtol, atol=atol)
        for name in output_names
    )


@dataclass
class BenchmarkController:
    registry: BuilderRegistry
    policy: BenchmarkPolicy

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        builder = self.registry.require(submission.kind)
        return builder.build(submission, context)

    def benchmark(self, task: ReferenceTask, candidate: BuiltCandidate) -> BenchmarkResult:
        executor = candidate.executor_factory()
        try:
            repeats = []
            for index in range(self.policy.repeats):
                seed = self.policy.seed + index
                baseline_inputs = task.make_inputs(seed)
                baseline_us = self._benchmark_reference(task, baseline_inputs)
                candidate_inputs = task.make_inputs(seed)
                candidate_us = self._benchmark_candidate(task, executor, candidate_inputs)
                correct = self._check_correctness(task, executor, seed + 10_000)
                repeats.append(
                    BenchmarkRepeat(
                        seed=seed,
                        baseline_us=baseline_us,
                        candidate_us=candidate_us,
                        correct=correct,
                    )
                )
                if self.policy.sleep_s > 0 and index + 1 < self.policy.repeats:
                    time.sleep(self.policy.sleep_s)
            return BenchmarkResult(tuple(repeats))
        finally:
            executor.close()

    def _time(self, fn) -> float:
        if self.policy.timing_mode == TimingMode.GRAPHED:
            return _graph_median_us(fn, self.policy.warmup, self.policy.iterations, self.policy.graph_calls)
        if self.policy.timing_mode == TimingMode.FLUSHED:
            return _flushed_median_us(fn, self.policy.warmup, self.policy.iterations)
        return _median_us(fn, self.policy.warmup, self.policy.iterations)

    def _benchmark_reference(self, task: ReferenceTask, inputs: ExecutionInputs) -> float:
        if task.benchmark_reference is not None:
            return self._time(task.benchmark_reference(inputs))
        return self._time(lambda: task.reference(inputs))

    def _benchmark_candidate(self, task: ReferenceTask, executor, inputs: ExecutionInputs) -> float:
        executor.prepare(inputs)
        if self.policy.clear_outputs_after_prepare:
            _clear_outputs(inputs, task.abi.output_names)
        return self._time(executor.launch)

    def _check_correctness(self, task: ReferenceTask, executor, seed: int) -> bool:
        reference_inputs = task.make_inputs(seed)
        expected = task.reference(reference_inputs)
        check_inputs = task.make_inputs(seed)
        executor.prepare(check_inputs)
        if self.policy.clear_outputs_after_prepare:
            _clear_outputs(check_inputs, task.abi.output_names)
        executor.launch()
        _synchronize()
        return _outputs_close(check_inputs, expected, task.abi.output_names, task.tolerances)
