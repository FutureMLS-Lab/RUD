import json
import os
import urllib.error
import urllib.request
from dataclasses import fields

from kernel_evaluator.inliner.artifact_builder import get_artifact_paths, find_kernel_symbol
from kernel_evaluator.inliner.types import ExternalReplacementKernelSpec, CScalarType

_TYPE_MAP = {
    "int": CScalarType.INT,
    "long long": CScalarType.LONG_LONG,
    "float": CScalarType.FLOAT,
    "double": CScalarType.DOUBLE,
    "bool": CScalarType.BOOL,
}

_SPEC_FIELDS = {f.name for f in fields(ExternalReplacementKernelSpec)}


def _get_api_url() -> str:
    port = os.environ.get("KERNEL_EVALUATOR_PORT", "8000")
    return os.environ.get("KERNEL_EVALUATOR_API", f"http://localhost:{port}")


def _get_api_key() -> str:
    return os.environ.get("KERNEL_EVALUATOR_ADMIN_API_KEY", "")


def _api_get(path: str) -> dict:
    url = f"{_get_api_url()}{path}"
    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _scalar_arg_types(scalar_args: dict | list) -> dict:
    if isinstance(scalar_args, list):
        return {arg["name"]: _TYPE_MAP[arg["type"]] for arg in scalar_args}
    return {
        name: _TYPE_MAP[meta["type"] if isinstance(meta, dict) else meta]
        for name, meta in scalar_args.items()
    }


def _api_response_to_spec(data: dict, spec_json: dict) -> ExternalReplacementKernelSpec:
    """Convert API response to ExternalReplacementKernelSpec, building artifacts."""
    source = data.get("postprocessed_source") or data["kernel_source"]
    function_name = data["function_name"]
    so_path, ptx_path, _ = get_artifact_paths(source, function_name)

    kernel_symbol = find_kernel_symbol(ptx_path)
    if not kernel_symbol:
        raise RuntimeError(f"Could not find kernel symbol in {ptx_path}")

    # Exclude fields we set explicitly
    exclude = {"ptx_path", "library_path", "kernel_symbol", "scalar_args"}
    return ExternalReplacementKernelSpec(
        **{k: v for k, v in spec_json.items() if k in _SPEC_FIELDS and k not in exclude},
        ptx_path=ptx_path,
        library_path=so_path,
        kernel_symbol=kernel_symbol,
        scalar_args=_scalar_arg_types(spec_json.get("scalar_args", {})),
    )


def get_kernel_spec_for_spec_json(spec_json: dict, gpu: str = "h100") -> ExternalReplacementKernelSpec:
    """Get kernel spec by full spec_json via API."""
    import urllib.parse
    spec_json_str = json.dumps(spec_json)
    path = f"/kernels/best?spec_json={urllib.parse.quote(spec_json_str)}&gpu={gpu}"
    try:
        data = _api_get(path)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise KeyError(f"No kernel for spec_json={spec_json}, gpu={gpu}")
        raise

    # Use spec_json from response (may have more fields than input)
    response_spec = data.get("spec_json", spec_json)
    return _api_response_to_spec(data, response_spec)


def get_kernel_spec_by_function_and_scalars(function_name: str, scalar_args: dict, gpu: str = "h100") -> ExternalReplacementKernelSpec:
    """Get kernel spec by function name and scalar args via API."""
    import urllib.parse
    scalar_args_json = json.dumps(scalar_args)
    path = f"/kernels/best?function_name={urllib.parse.quote(function_name)}&scalar_args={urllib.parse.quote(scalar_args_json)}&gpu={gpu}"
    try:
        data = _api_get(path)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise KeyError(f"No kernel for {function_name}, {scalar_args}, gpu={gpu}")
        raise

    # Use full spec_json from API response (has scalar_args with type info)
    spec_json = data.get("spec_json") or {"function_name": data["function_name"], "scalar_args": data.get("scalar_args", {})}
    return _api_response_to_spec(data, spec_json)
