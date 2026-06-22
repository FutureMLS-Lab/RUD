# Python registration template for ABt GEMM kernel.
# Paths are placeholders - actual paths come from kernel_evaluator artifacts.

import triton
import triton.language as tl
import torch
from torch.library import triton_op, wrap_triton
from kernel_evaluator.inliner import ExternalReplacementKernelSpec, CScalarType

# Artifact paths (filled in by kernel_evaluator)
LIBRARY_PATH = "/path/to/kernel.so"
PTX_PATH = "/path/to/kernel.ptx"
KERNEL_SYMBOL = "kernel_entry"

SPEC = ExternalReplacementKernelSpec(
    library_path=LIBRARY_PATH,
    ptx_path=PTX_PATH,
    function_name="tk_gemm",
    tensor_args=["a", "bt", "c"],
    scalar_args={"dim_m": CScalarType.INT, "dim_n": CScalarType.INT, "dim_k": CScalarType.INT},
    kernel_symbol=KERNEL_SYMBOL,
    globals_size_fn="tk_gemm_globals_size",
    make_globals_fn="tk_gemm_make_globals",
    grid_dims_fn="tk_gemm_grid_dims",
    block_dim_fn="tk_gemm_block_dim",
    shmem_bytes_fn="tk_gemm_shmem_bytes",
    pointer_arg_roles={"a": "read", "bt": "read", "c": "write"},
    max_barrier_slots=5,
    num_tma_descriptors_fn="tk_gemm_num_tma_descriptors",
    describe_tma_descriptors_fn="tk_gemm_describe_tma_descriptors",
)


@triton.jit
def _mm_tk_stub(
    a_ptr, bt_ptr, c_ptr,
    dim_m: tl.constexpr, dim_n: tl.constexpr, dim_k: tl.constexpr,
):
    tl.store(c_ptr, tl.load(a_ptr))


@triton_op("tkcc::mm_tk", mutates_args=("c",))
def tk_mm(a: torch.Tensor, bt: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N, _ = bt.shape
    wrap_triton(_mm_tk_stub)[(1, 1, 1)](
        a, bt, c,
        dim_m=M, dim_n=N, dim_k=K,
        num_warps=1, num_stages=1,
    )
    return c


# Registration - connects the stub to the external kernel spec
STUB_NAME = "_mm_tk_stub"
