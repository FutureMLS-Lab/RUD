from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import uuid
from typing import Any

from kernel_evaluator.services.evaluation.types import ExecutionInputs


@dataclass
class ReferencePlugin:
    make_inputs: Callable[[int], ExecutionInputs]
    reference: Callable[[ExecutionInputs], Mapping[str, Any]]
    tolerances: tuple[float, float]
    output_names: tuple[str, ...]
    benchmark_reference: Callable[[ExecutionInputs], Callable[[], None]] | None = None


@dataclass(frozen=True)
class OperationContract:
    plugin: str
    shape: Mapping[str, Any]
    task_slug: str
    reference_plugin: str
    spec: Mapping[str, Any]
    scalars: Mapping[str, Any]
    dtype: str
    instructions: str


@dataclass(frozen=True)
class KernelEvalPlugin:
    name: str
    reference_factory: Callable[[str, Mapping[str, Any]], ReferencePlugin]
    contract_factory: Callable[[Mapping[str, Any]], OperationContract] | None = None


_REFERENCE_FACTORIES: dict[str, Callable] = {}
_CONTRACT_FACTORIES: dict[str, Callable] = {}


def register_plugin(plugin: KernelEvalPlugin) -> None:
    if plugin.name == "":
        raise ValueError("plugin name must be non-empty")
    if plugin.name in _REFERENCE_FACTORIES:
        raise ValueError(f"duplicate plugin registration: {plugin.name}")
    _REFERENCE_FACTORIES[plugin.name] = plugin.reference_factory
    if plugin.contract_factory is not None:
        _CONTRACT_FACTORIES[plugin.name] = plugin.contract_factory


def make_reference_plugin(name: str, dtype: str, spec: Mapping[str, Any]) -> ReferencePlugin:
    if name not in _REFERENCE_FACTORIES:
        raise ValueError(f"Unknown reference plugin '{name}'. Available: {list(_REFERENCE_FACTORIES)}")
    return _REFERENCE_FACTORIES[name](dtype, spec)


def _normalize_shapes(shapes: object) -> list[Mapping[str, Any]]:
    if not isinstance(shapes, list) or len(shapes) == 0:
        raise ValueError("shapes must be a non-empty list")
    normalized = []
    for shape in shapes:
        if not isinstance(shape, Mapping):
            raise ValueError("each shape must be an object")
        normalized.append(shape)
    return normalized


def _suite_run_id(target: str, plugin: str, shapes: list[Mapping[str, Any]]) -> str:
    payload = {"plugin": plugin, "target": target, "shapes": [dict(shape) for shape in shapes]}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    unique = uuid.uuid4().hex[:6]
    return f"{target}_{plugin.replace('.', '_')}_suite_{digest}_{unique}"


def _suite_instructions(name: str, shapes: list[Mapping[str, Any]], target_instruction: str) -> str:
    shapes_json = json.dumps([dict(shape) for shape in shapes], sort_keys=True, separators=(",", ":"))
    return (
        f"Optimize {name} across {len(shapes)} shape(s): {shapes_json}. "
        "One submitted candidate is built once and benchmarked against every shape. "
        f"{target_instruction}"
    )


def make_plugin_run(name: str, target: str, shapes: object, cuda_arch: str = "90a") -> dict:
    if name not in _CONTRACT_FACTORIES:
        raise ValueError(f"Plugin '{name}' does not support run initialization. Available: {list(_CONTRACT_FACTORIES)}")
    from kernel_evaluator.services.plugins.targets import build_target_contract
    import time

    normalized_shapes = _normalize_shapes(shapes)
    target_contract = build_target_contract(target)
    operations = [_CONTRACT_FACTORIES[name](shape) for shape in normalized_shapes]
    benchmark_shapes = []
    for operation in operations:
        benchmark_shapes.append(
            {
                "shape": dict(operation.shape),
                "task_slug": operation.task_slug,
                "reference_plugin": operation.reference_plugin,
                "spec": {**dict(operation.spec), **target_contract.spec_overrides},
                "scalars": dict(operation.scalars),
                "dtype": operation.dtype,
            }
        )
    return {
        "run_id": _suite_run_id(target, name, normalized_shapes),
        "plugin": name,
        "target": target,
        "candidate_kind": str(target_contract.candidate_kind),
        "entrypoint": target_contract.entrypoint,
        "shapes": [dict(shape) for shape in normalized_shapes],
        "benchmark_shapes": benchmark_shapes,
        "instructions": _suite_instructions(name, normalized_shapes, target_contract.instruction_contract),
        "cuda_arch": cuda_arch,
    }


from kernel_evaluator.services.plugins import add_aiter, aiter_moe_up_gemm, cuda_int4_matmul, fa3_paged_decode, fp8_gemm, linear, mla_decode_fp8, rms_norm, sdpa, sparse_attention_fwd  # noqa: E402

for _module in (add_aiter, aiter_moe_up_gemm, cuda_int4_matmul, fa3_paged_decode, fp8_gemm, linear, mla_decode_fp8, rms_norm, sdpa, sparse_attention_fwd):
    register_plugin(_module.PLUGIN)


def _load_external_plugins() -> None:
    """Register custom plugins from dirs listed in
    ``KERNEL_EVALUATOR_EXTRA_PLUGINS_PATH`` (os.pathsep-separated). Each
    ``*.py`` that exports ``PLUGIN`` is loaded and registered. This lets a task
    work dir supply a per-kernel plugin without editing this file or rebuilding
    the package — the harness installs from tkcc, the evaluator is customized
    in the work dir. Built-in names win; failures are logged and skipped."""
    import importlib.util
    import os
    import sys
    from pathlib import Path

    raw = os.environ.get("KERNEL_EVALUATOR_EXTRA_PLUGINS_PATH", "").strip()
    if not raw:
        return
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        directory = Path(entry)
        if not entry or not directory.is_dir():
            continue
        for f in sorted(directory.glob("*.py")):
            if f.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"_ext_plugin_{f.stem}", f)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                plugin = getattr(module, "PLUGIN", None)
                if plugin is not None and plugin.name not in _REFERENCE_FACTORIES:
                    register_plugin(plugin)
                    print(f"[kernel_evaluator] loaded external plugin: {plugin.name} ({f})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[kernel_evaluator] skipped external plugin {f}: {exc}", flush=True)


_load_external_plugins()
