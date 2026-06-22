from dataclasses import dataclass

DEFAULT_RUNTIME_ENV = "kernel-evaluator"

_RUNTIME_ENVS = {DEFAULT_RUNTIME_ENV}

_BUILD_PROFILES = {
    "cuda_sm90a_default",
    "hip_pybind",
    "prebuilt",
    "python_entrypoint",
    "cutedsl_entrypoint",
    "triton_entrypoint",
}


@dataclass(frozen=True)
class RuntimePolicy:
    env: str
    build_profile: str
    import_roots: tuple[str, ...] = ()


def validate_runtime_policy(policy: RuntimePolicy) -> RuntimePolicy:
    if policy.env not in _RUNTIME_ENVS:
        raise ValueError(f"unknown runtime env: {policy.env}")
    if policy.build_profile not in _BUILD_PROFILES:
        raise ValueError(f"unknown build profile: {policy.build_profile}")
    for root in policy.import_roots:
        if root.startswith("/") or ".." in root.split("/"):
            raise ValueError(f"invalid import root: {root}")
    return policy
