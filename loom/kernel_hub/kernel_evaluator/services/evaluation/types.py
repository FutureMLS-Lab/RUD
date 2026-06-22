from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import math
from pathlib import Path
from typing import Any, Protocol


class TimingMode(StrEnum):
    GRAPHED = "graphed"
    STANDARD = "standard"
    FLUSHED = "flushed"


class CandidateKind(StrEnum):
    CUDA_SOURCE = "cuda_source"
    HIP_SOURCE = "hip_source"
    EXTERNAL_PTX_SO = "external_ptx_so"
    CUTEDSL_AOT = "cutedsl_aot"
    TRITON_CALLABLE = "triton_callable"
    PYTHON_CALLABLE = "python_callable"


class ArtifactKind(StrEnum):
    SOURCE = "source"
    PTX = "ptx"
    CUBIN = "cubin"
    SHARED_OBJECT = "shared_object"
    HEADER = "header"
    OBJECT = "object"
    MANIFEST = "manifest"
    LOG = "log"
    RESOURCE_USAGE = "resource_usage"


class TensorAccess(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "readwrite"


class ScalarType(StrEnum):
    INT = "int"
    LONG_LONG = "long long"
    FLOAT = "float"
    DOUBLE = "double"
    BOOL = "bool"


@dataclass(frozen=True)
class Artifact:
    kind: ArtifactKind
    path: Path
    checksum: str | None = None


@dataclass(frozen=True)
class TensorArg:
    name: str
    access: TensorAccess
    dtype: str | None = None
    shape: tuple[str | int, ...] = ()
    layout: str | None = None


@dataclass(frozen=True)
class ScalarArg:
    name: str
    dtype: ScalarType
    value: int | float | bool | None = None


@dataclass(frozen=True)
class KernelABI:
    function_name: str
    tensor_args: tuple[TensorArg, ...]
    scalar_args: tuple[ScalarArg, ...]
    output_names: tuple[str, ...]
    globals_size_fn: str | None = None
    make_globals_fn: str | None = None
    grid_dims_fn: str | None = None
    block_dim_fn: str | None = None
    shmem_bytes_fn: str | None = None
    kernel_symbol: str | None = None
    cluster_shape: tuple[int, int, int] = (1, 1, 1)
    max_barrier_slots: int = -1
    run_fn: str | None = None


@dataclass(frozen=True)
class ExecutionInputs:
    tensors: Mapping[str, Any]
    scalars: Mapping[str, int | float | bool]
    output_names: tuple[str, ...]

    @property
    def outputs(self) -> tuple[Any, ...]:
        return tuple(self.tensors[name] for name in self.output_names)


@dataclass(frozen=True)
class ReferenceTask:
    slug: str
    abi: KernelABI
    make_inputs: Callable[[int], ExecutionInputs]
    reference: Callable[[ExecutionInputs], Mapping[str, Any]]
    tolerances: tuple[float, float]
    benchmark_reference: Callable[[ExecutionInputs], Callable[[], None]] | None = None


@dataclass(frozen=True)
class CandidateSubmission:
    kind: CandidateKind
    abi: KernelABI
    artifacts: tuple[Artifact, ...] = ()
    source: str | None = None
    entrypoint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildContext:
    work_dir: Path
    cuda_arch: str
    include_dirs: tuple[Path, ...] = ()
    library_dirs: tuple[Path, ...] = ()
    extra_flags: tuple[str, ...] = ()


class CandidateExecutor(Protocol):
    def prepare(self, inputs: ExecutionInputs) -> None:
        ...

    def launch(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class BuiltCandidate:
    kind: CandidateKind
    abi: KernelABI
    artifacts: tuple[Artifact, ...]
    executor_factory: Callable[[], CandidateExecutor]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkPolicy:
    warmup: int = 10
    iterations: int = 50
    graph_calls: int = 100
    timing_mode: TimingMode = TimingMode.GRAPHED
    repeats: int = 3
    seed: int = 0
    sleep_s: float = 10.0
    clear_outputs_after_prepare: bool = True


@dataclass(frozen=True)
class BenchmarkRepeat:
    seed: int
    baseline_us: float
    candidate_us: float
    correct: bool


def _geomean(values: Sequence[float]) -> float:
    if len(values) == 0:
        raise ValueError("geomean requires at least one value")
    return math.prod(values) ** (1.0 / len(values))


@dataclass(frozen=True)
class ShapeBenchmarkResult:
    shape: Mapping[str, Any]
    task_slug: str
    repeats: tuple[BenchmarkRepeat, ...]

    @property
    def correct(self) -> bool:
        return all(repeat.correct for repeat in self.repeats)

    @property
    def baseline_us(self) -> float:
        return min(repeat.baseline_us for repeat in self.repeats)

    @property
    def candidate_us(self) -> float:
        return max(repeat.candidate_us for repeat in self.repeats)

    @property
    def speedup(self) -> float:
        return self.baseline_us / self.candidate_us


@dataclass(frozen=True, init=False)
class BenchmarkResult:
    repeats: tuple[BenchmarkRepeat, ...]
    shape_results: tuple[ShapeBenchmarkResult, ...]

    def __init__(
        self,
        repeats: tuple[BenchmarkRepeat, ...] | None = None,
        shape_results: tuple[ShapeBenchmarkResult, ...] = (),
    ):
        if repeats is None and len(shape_results) == 0:
            raise ValueError("BenchmarkResult requires repeats or shape_results")
        if repeats is not None and len(shape_results) != 0:
            raise ValueError("BenchmarkResult cannot mix repeats and shape_results")
        if repeats is None:
            repeats = _aggregate_shape_repeats(shape_results)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "shape_results", shape_results)

    @property
    def correct(self) -> bool:
        return all(repeat.correct for repeat in self.repeats)

    @property
    def baseline_us(self) -> float:
        return min(repeat.baseline_us for repeat in self.repeats)

    @property
    def candidate_us(self) -> float:
        return max(repeat.candidate_us for repeat in self.repeats)

    @property
    def speedup(self) -> float:
        return self.baseline_us / self.candidate_us


def _aggregate_shape_repeats(shape_results: tuple[ShapeBenchmarkResult, ...]) -> tuple[BenchmarkRepeat, ...]:
    if len(shape_results) == 0:
        raise ValueError("shape_results must not be empty")
    repeat_count = len(shape_results[0].repeats)
    if repeat_count == 0:
        raise ValueError("shape_results repeats must not be empty")
    for shape_result in shape_results:
        if len(shape_result.repeats) != repeat_count:
            raise ValueError("all shape_results must have the same repeat count")
    aggregate_repeats = []
    for repeat_index in range(repeat_count):
        repeats = tuple(shape_result.repeats[repeat_index] for shape_result in shape_results)
        aggregate_repeats.append(
            BenchmarkRepeat(
                seed=repeats[0].seed,
                baseline_us=_geomean(tuple(repeat.baseline_us for repeat in repeats)),
                candidate_us=_geomean(tuple(repeat.candidate_us for repeat in repeats)),
                correct=all(repeat.correct for repeat in repeats),
            )
        )
    return tuple(aggregate_repeats)

class CandidateBuilder(Protocol):
    kind: CandidateKind

    def build(self, submission: CandidateSubmission, context: BuildContext) -> BuiltCandidate:
        ...
