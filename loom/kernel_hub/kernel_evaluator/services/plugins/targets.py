from dataclasses import dataclass

from kernel_evaluator.services.evaluation.types import CandidateKind

CUDA_ABI_HELPERS = {
    "globals_size_fn": "globals_size",
    "make_globals_fn": "make_globals",
    "grid_dims_fn": "grid_dims",
    "block_dim_fn": "block_dim",
    "shmem_bytes_fn": "shmem_bytes",
    "kernel_symbol": "kernel",
}


@dataclass(frozen=True)
class TargetContract:
    candidate_kind: CandidateKind
    entrypoint: str | None
    spec_overrides: dict
    instruction_contract: str


def build_target_contract(target: str) -> TargetContract:
    if target == "cuda":
        return TargetContract(
            candidate_kind=CandidateKind.CUDA_SOURCE,
            entrypoint=None,
            spec_overrides=CUDA_ABI_HELPERS,
            instruction_contract="Export kernel plus globals_size, make_globals, grid_dims, block_dim, and shmem_bytes.",
        )
    if target == "cutedsl":
        return TargetContract(
            candidate_kind=CandidateKind.CUTEDSL_AOT,
            entrypoint="source:prepare",
            spec_overrides={},
            instruction_contract=(
                "Export exactly def prepare(inputs). inputs.tensors and inputs.scalars contain the task values. "
                "prepare may compile and bind tensors, but must not write outputs. "
                "Return a zero-argument launch function that performs the computation."
            ),
        )
    if target == "hip":
        return TargetContract(
            candidate_kind=CandidateKind.HIP_SOURCE,
            entrypoint="candidate:dispatch",
            spec_overrides={},
            instruction_contract=(
                "Submit C++ HIP/pybind11 source that builds a module named candidate. "
                "Export dispatch and accept the task tensors in spec order."
            ),
        )
    if target == "triton":
        return TargetContract(
            candidate_kind=CandidateKind.TRITON_CALLABLE,
            entrypoint="source:prepare",
            spec_overrides={},
            instruction_contract=(
                "Export exactly def prepare(inputs). inputs.tensors and inputs.scalars contain the task values. "
                "prepare may bind tensors, precompute grids/strides, allocate scratch buffers, and warm up "
                "Triton autotune, but must not write outputs. "
                "Return a zero-argument launch function that performs the kernel launches. "
                "Only work inside that closure is timed; setup in prepare is free."
            ),
        )
    raise ValueError(f"unsupported target: {target}")
