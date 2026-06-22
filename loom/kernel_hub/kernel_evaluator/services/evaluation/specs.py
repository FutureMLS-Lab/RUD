from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kernel_evaluator.services.evaluation.types import KernelABI, ScalarArg, ScalarType, TensorAccess, TensorArg

_ACCESS_MAP = {
    "read": TensorAccess.READ,
    "write": TensorAccess.WRITE,
    "readwrite": TensorAccess.READ_WRITE,
}

_SCALAR_TYPE_MAP = {
    "int": ScalarType.INT,
    "long long": ScalarType.LONG_LONG,
    "float": ScalarType.FLOAT,
    "double": ScalarType.DOUBLE,
    "bool": ScalarType.BOOL,
}

_DTYPES = {
    "bf16",
    "fp8",
    "fp16",
    "fp32",
    "fp64",
    "int8",
    "int32",
    "int64",
    "uint8",
    "bool",
}


@dataclass(frozen=True)
class NormalizedSpec:
    raw: Mapping[str, Any]
    abi: KernelABI
    reference_plugin: str
    tolerances: tuple[float, float] | None


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_sequence(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, owner: str) -> str:
    if key not in mapping:
        raise ValueError(f"{owner} missing required field '{key}'")
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{owner}.{key} must be a non-empty string")
    return value


def _optional_str(mapping: Mapping[str, Any], key: str, owner: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{owner}.{key} must be a non-empty string")
    return value


def _optional_int(mapping: Mapping[str, Any], key: str, owner: str, default: int) -> int:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, int):
        raise ValueError(f"{owner}.{key} must be an int")
    return value


def _parse_shape(value: object, owner: str) -> tuple[str | int, ...]:
    if value is None:
        return ()
    shape = _require_sequence(value, f"{owner}.shape")
    parsed = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int):
            parsed.append(dim)
        elif isinstance(dim, str) and dim != "":
            parsed.append(dim)
        else:
            raise ValueError(f"{owner}.shape[{index}] must be an int or non-empty string")
    return tuple(parsed)


def _parse_tensor_args(spec: Mapping[str, Any]) -> tuple[TensorArg, ...]:
    tensor_specs = _require_sequence(spec["tensor_args"], "tensor_args")
    names: set[str] = set()
    tensors = []
    for index, item in enumerate(tensor_specs):
        owner = f"tensor_args[{index}]"
        tensor = _require_mapping(item, owner)
        name = _require_str(tensor, "name", owner)
        if name in names:
            raise ValueError(f"duplicate tensor arg '{name}'")
        names.add(name)
        role = _require_str(tensor, "role", owner)
        if role not in _ACCESS_MAP:
            raise ValueError(f"{owner}.role must be one of {sorted(_ACCESS_MAP)}")
        dtype = _optional_str(tensor, "dtype", owner)
        if dtype is not None and dtype not in _DTYPES:
            raise ValueError(f"{owner}.dtype must be one of {sorted(_DTYPES)}")
        shape = _parse_shape(tensor["shape"], owner) if "shape" in tensor else ()
        layout = _optional_str(tensor, "layout", owner)
        tensors.append(TensorArg(name, _ACCESS_MAP[role], dtype, shape, layout))
    if not any(tensor.access in (TensorAccess.WRITE, TensorAccess.READ_WRITE) for tensor in tensors):
        raise ValueError("tensor_args must include at least one write or readwrite tensor")
    return tuple(tensors)


def _parse_scalar_args(spec: Mapping[str, Any]) -> tuple[ScalarArg, ...]:
    if "scalar_args" not in spec:
        return ()
    scalar_specs = _require_sequence(spec["scalar_args"], "scalar_args")
    scalars = []
    names: set[str] = set()
    for index, item in enumerate(scalar_specs):
        owner = f"scalar_args[{index}]"
        meta = _require_mapping(item, owner)
        name = _require_str(meta, "name", owner)
        if name in names:
            raise ValueError(f"duplicate scalar arg '{name}'")
        names.add(name)
        type_name = _require_str(meta, "type", owner)
        if type_name not in _SCALAR_TYPE_MAP:
            raise ValueError(f"{owner}.type must be one of {sorted(_SCALAR_TYPE_MAP)}")
        value = meta["value"] if "value" in meta else None
        scalars.append(ScalarArg(name, _SCALAR_TYPE_MAP[type_name], value))
    return tuple(scalars)


def normalize_legacy_scalar_args(spec: Mapping[str, Any]) -> dict[str, Any]:
    if "scalar_args" not in spec or isinstance(spec["scalar_args"], list):
        return dict(spec)
    scalar_specs = _require_mapping(spec["scalar_args"], "scalar_args")
    normalized = dict(spec)
    normalized["scalar_args"] = [
        {"name": name, **dict(_require_mapping(meta, f"scalar_args.{name}"))}
        for name, meta in scalar_specs.items()
    ]
    return normalized


def _helper(spec: Mapping[str, Any], name: str) -> str | None:
    if "abi_helpers" in spec:
        helpers = _require_mapping(spec["abi_helpers"], "abi_helpers")
        return _optional_str(helpers, name, "abi_helpers")
    return _optional_str(spec, name, "spec")


def _parse_cluster_shape(spec: Mapping[str, Any]) -> tuple[int, int, int]:
    if "cluster_shape" not in spec:
        return (1, 1, 1)
    value = _require_sequence(spec["cluster_shape"], "cluster_shape")
    if len(value) != 3:
        raise ValueError("cluster_shape must have three ints")
    parsed = []
    for index, dim in enumerate(value):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"cluster_shape[{index}] must be a positive int")
        parsed.append(dim)
    return tuple(parsed)


def parse_kernel_abi(spec: Mapping[str, Any]) -> KernelABI:
    _require_str(spec, "function_name", "spec")
    if "tensor_args" not in spec:
        raise ValueError("spec missing required field 'tensor_args'")
    tensors = _parse_tensor_args(spec)
    scalars = _parse_scalar_args(spec)
    output_names = tuple(tensor.name for tensor in tensors if tensor.access in (TensorAccess.WRITE, TensorAccess.READ_WRITE))
    return KernelABI(
        function_name=spec["function_name"],
        tensor_args=tensors,
        scalar_args=scalars,
        output_names=output_names,
        globals_size_fn=_helper(spec, "globals_size_fn"),
        make_globals_fn=_helper(spec, "make_globals_fn"),
        grid_dims_fn=_helper(spec, "grid_dims_fn"),
        block_dim_fn=_helper(spec, "block_dim_fn"),
        shmem_bytes_fn=_helper(spec, "shmem_bytes_fn"),
        kernel_symbol=_optional_str(spec, "kernel_symbol", "spec"),
        cluster_shape=_parse_cluster_shape(spec),
        max_barrier_slots=_optional_int(spec, "max_barrier_slots", "spec", -1),
        run_fn=_optional_str(spec, "run_fn", "spec"),
    )


def parse_spec(spec: Mapping[str, Any]) -> NormalizedSpec:
    if "reference_plugin" not in spec:
        raise ValueError("spec missing required field 'reference_plugin'")
    reference_plugin = _require_str(spec, "reference_plugin", "spec")
    abi = parse_kernel_abi(spec)
    tolerances = None
    if "tolerances" in spec:
        tolerances_spec = _require_mapping(spec["tolerances"], "tolerances")
        if "rtol" not in tolerances_spec or "atol" not in tolerances_spec:
            raise ValueError("tolerances must include rtol and atol")
        tolerances = (float(tolerances_spec["rtol"]), float(tolerances_spec["atol"]))
    elif "rtol" in spec or "atol" in spec:
        if "rtol" not in spec or "atol" not in spec:
            raise ValueError("spec must include both rtol and atol")
        tolerances = (float(spec["rtol"]), float(spec["atol"]))
    return NormalizedSpec(spec, abi, reference_plugin, tolerances)


def inject_scalar_values(spec: Mapping[str, Any], scalars: Mapping[str, int | float | bool]) -> dict[str, Any]:
    if "scalar_args" not in spec:
        if scalars:
            raise ValueError("scalar values provided for spec without scalar_args")
        return dict(spec)
    scalar_args = _parse_scalar_args(spec)
    names = [scalar.name for scalar in scalar_args]
    missing = [name for name in names if name not in scalars]
    if missing:
        raise ValueError(f"missing scalar values: {missing}")
    extra = [name for name in scalars if name not in names]
    if extra:
        raise ValueError(f"unknown scalar values: {extra}")
    patched = dict(spec)
    patched["scalar_args"] = [
        {"name": scalar.name, "type": str(scalar.dtype), "value": scalars[scalar.name]}
        for scalar in scalar_args
    ]
    return patched


def require_helper_abi(abi: KernelABI) -> None:
    missing = []
    for name, value in (
        ("globals_size_fn", abi.globals_size_fn),
        ("make_globals_fn", abi.make_globals_fn),
        ("grid_dims_fn", abi.grid_dims_fn),
        ("block_dim_fn", abi.block_dim_fn),
        ("shmem_bytes_fn", abi.shmem_bytes_fn),
    ):
        if value is None:
            missing.append(name)
    if missing:
        raise ValueError(f"helper ABI missing fields: {missing}")


def require_cubin_abi(abi: KernelABI) -> None:
    require_helper_abi(abi)
    if abi.kernel_symbol is None:
        raise ValueError("cubin packages require kernel_symbol")
