from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError, field_validator

from kernel_evaluator.services.evaluation.specs import (
    inject_scalar_values,
    normalize_legacy_scalar_args,
    parse_spec,
    require_cubin_abi,
    require_helper_abi,
)
from kernel_evaluator.services.evaluation.types import CandidateKind, TimingMode

CUDA_ABI_FIELDS = {
    "globals_size_fn",
    "make_globals_fn",
    "grid_dims_fn",
    "block_dim_fn",
    "shmem_bytes_fn",
    "kernel_symbol",
    "max_barrier_slots",
    "run_fn",
    "cluster_shape",
}

CALLABLE_KINDS = {
    CandidateKind.HIP_SOURCE,
    CandidateKind.CUTEDSL_AOT,
    CandidateKind.TRITON_CALLABLE,
    CandidateKind.PYTHON_CALLABLE,
}

DEFAULT_BENCHMARK_POLICY = {
    "timing_mode": "standard",
    "warmup": 10,
    "iterations": 50,
    "graph_calls": 100,
    "repeats": 3,
    "sleep_s": 10.0,
    "clear_outputs_after_prepare": True,
}


class BenchmarkPolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timing_mode: TimingMode = TimingMode.GRAPHED
    warmup: StrictInt = Field(DEFAULT_BENCHMARK_POLICY["warmup"], ge=0)
    iterations: StrictInt = Field(DEFAULT_BENCHMARK_POLICY["iterations"], ge=1)
    graph_calls: StrictInt = Field(DEFAULT_BENCHMARK_POLICY["graph_calls"], ge=1)
    repeats: StrictInt = Field(DEFAULT_BENCHMARK_POLICY["repeats"], ge=1)
    sleep_s: float = DEFAULT_BENCHMARK_POLICY["sleep_s"]
    clear_outputs_after_prepare: StrictBool = DEFAULT_BENCHMARK_POLICY["clear_outputs_after_prepare"]

    @field_validator("sleep_s", mode="before")
    @classmethod
    def _non_negative_number(cls, value: object) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0.0:
            raise ValueError("must be a number >= 0.0")
        return float(value)


class BenchmarkShapeModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shape: dict[str, Any]
    task_slug: StrictStr
    reference_plugin: StrictStr
    spec: dict[str, Any]
    scalars: dict[str, Any]
    dtype: StrictStr

    @field_validator("task_slug", "reference_plugin", "dtype")
    @classmethod
    def _non_empty_str(cls, value: str) -> str:
        if value == "":
            raise ValueError("must be a non-empty string")
        return value


class RunContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: StrictStr
    plugin: StrictStr
    target: StrictStr
    candidate_kind: CandidateKind
    entrypoint: StrictStr | None = None
    benchmark_shapes: list[BenchmarkShapeModel] = Field(min_length=1)
    benchmark_policy: BenchmarkPolicyModel = Field(default_factory=BenchmarkPolicyModel)
    instructions: StrictStr
    target_speedup: float | None = None
    cuda_arch: StrictStr = "90a"

    @field_validator("run_id", "plugin", "target", "instructions", "cuda_arch")
    @classmethod
    def _non_empty_str(cls, value: str) -> str:
        if value == "":
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("entrypoint")
    @classmethod
    def _entrypoint(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("must be null or a non-empty string")
        return value


def _validate_target_contract(kind: CandidateKind, spec: Mapping[str, Any], entrypoint: str | None) -> None:
    parsed = parse_spec(spec)
    if kind == CandidateKind.CUDA_SOURCE:
        require_cubin_abi(parsed.abi)
        if entrypoint is not None:
            raise ValueError("cuda_source runs must use entrypoint null")
        return
    if kind == CandidateKind.EXTERNAL_PTX_SO:
        require_helper_abi(parsed.abi)
        if entrypoint is not None:
            raise ValueError("external_ptx_so runs must use entrypoint null")
        return
    if kind in CALLABLE_KINDS:
        if entrypoint is None:
            raise ValueError(f"{kind} runs require an entrypoint")
        cuda_fields = sorted(name for name in CUDA_ABI_FIELDS if name in spec)
        if cuda_fields:
            raise ValueError(f"{kind} spec includes CUDA-only ABI fields: {cuda_fields}")
        return
    raise ValueError(f"unsupported run candidate kind: {kind}")


def _validate_benchmark_shape(
    entry: BenchmarkShapeModel,
    kind: CandidateKind,
    entrypoint: str | None,
    owner: str,
) -> dict[str, Any]:
    shape = dict(entry.shape)
    task_slug = entry.task_slug
    reference_plugin = entry.reference_plugin
    spec = normalize_legacy_scalar_args(dict(entry.spec))
    scalars = dict(entry.scalars)
    dtype = entry.dtype
    parsed = parse_spec(spec)
    spec = inject_scalar_values(spec, scalars)
    if parsed.reference_plugin != reference_plugin:
        raise ValueError(f"{owner}.reference_plugin must match spec.reference_plugin")
    _validate_target_contract(kind, spec, entrypoint)
    return {
        "shape": shape,
        "task_slug": task_slug,
        "reference_plugin": reference_plugin,
        "spec": spec,
        "scalars": scalars,
        "dtype": dtype,
    }


def validate_run_contract(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("run must be an object")
    run = raw["run"] if "run" in raw else raw
    try:
        model = RunContractModel.model_validate(run)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    if model.candidate_kind == CandidateKind.HIP_SOURCE and model.benchmark_policy.timing_mode == TimingMode.GRAPHED:
        raise ValueError("hip target does not support graphed timing; use standard or flushed")
    benchmark_shapes = tuple(
        _validate_benchmark_shape(shape, model.candidate_kind, model.entrypoint, f"run.benchmark_shapes[{index}]")
        for index, shape in enumerate(model.benchmark_shapes)
    )
    reference_plugins = {shape["reference_plugin"] for shape in benchmark_shapes}
    if len(reference_plugins) != 1:
        raise ValueError("run.benchmark_shapes must use a single reference_plugin")
    return {
        "run_id": model.run_id,
        "plugin": model.plugin,
        "target": model.target,
        "candidate_kind": str(model.candidate_kind),
        "entrypoint": model.entrypoint,
        "shapes": [dict(shape["shape"]) for shape in benchmark_shapes],
        "benchmark_shapes": [dict(shape) for shape in benchmark_shapes],
        "benchmark_policy": model.benchmark_policy.model_dump(mode="json"),
        "instructions": model.instructions,
        "target_speedup": model.target_speedup,
        "cuda_arch": model.cuda_arch,
    }
