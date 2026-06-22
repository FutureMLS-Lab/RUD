"""SM100 Sparse Attention Forward (Prefill) plugin.

This plugin implements the sparse attention forward kernel for SM100 (Blackwell)
using the MM-Sparse-Attention CuTeDSL implementation. It supports:
- CSR sparse metadata (k2q_row_ptr, k2q_q_indices)
- Variable length Q/K/V sequences via cu_seqlens
- Grouped-query attention (GQA) with qhead_per_kv in {1, 2, 4, 8, 16}
- Head dimension D=128 only (SM100 constraint)
- Block KV size 128

The kernel is used as both correctness oracle and timing baseline.

Requirements:
    The MM-Sparse-Attention package must be installed in the Docker image.
    The package root (containing interface.py, sparse_index_utils.py, and src/)
    must be in the Python path. Add to Dockerfile:

        ENV PYTHONPATH="${PYTHONPATH}:/path/to/MM-Sparse-Attention"

    Or install as a package if setup.py is provided.
"""

import torch

from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import scalar_values
from kernel_evaluator.services.evaluation.types import ExecutionInputs

PLUGIN_NAME = "sparse_attention.fwd"

SUPPORTED_TOPK = (4, 8, 16, 32)
HEAD_DIM = 128
BLK_KV = 128


def _generate_q2k_indices(
    batch: int,
    total_q: int,
    total_k: int,
    head_kv: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Generate random sparse attention patterns (q2k_indices).

    Each Q token selects topK KV blocks. Values are batch-local KV block indices.
    Invalid patterns (out of range for a batch) are clamped to valid range.
    """
    q2k_indices = torch.zeros(head_kv, total_q, topk, dtype=torch.int32, device=device)

    cu_seqlens_q_cpu = cu_seqlens_q.cpu()
    cu_seqlens_k_cpu = cu_seqlens_k.cpu()

    for b in range(batch):
        q_start = int(cu_seqlens_q_cpu[b].item())
        q_end = int(cu_seqlens_q_cpu[b + 1].item())
        k_len = int(cu_seqlens_k_cpu[b + 1].item()) - int(cu_seqlens_k_cpu[b].item())
        num_kv_blocks = (k_len + BLK_KV - 1) // BLK_KV

        if q_end > q_start and num_kv_blocks > 0:
            num_q_in_batch = q_end - q_start
            for h in range(head_kv):
                if num_kv_blocks >= topk:
                    indices = torch.randint(
                        0, num_kv_blocks, (num_q_in_batch, topk),
                        dtype=torch.int32, device=device, generator=generator
                    )
                else:
                    indices = torch.randint(
                        0, num_kv_blocks, (num_q_in_batch, topk),
                        dtype=torch.int32, device=device, generator=generator
                    )
                q2k_indices[h, q_start:q_end, :] = indices

    return q2k_indices


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    batch = scalars["batch"]
    total_q = scalars["total_q"]
    total_k = scalars["total_k"]
    head_q = scalars["head_q"]
    head_kv = scalars["head_kv"]
    topk = scalars["topk"]
    max_seqlen_q = scalars["max_seqlen_q"]
    max_seqlen_k = scalars["max_seqlen_k"]
    causal = scalars.get("causal", False)

    tensor_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    tolerance = (2e-2, 2e-2)

    def make_inputs(seed: int) -> ExecutionInputs:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)

        cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")

        q_per_batch = total_q // batch
        k_per_batch = total_k // batch
        for b in range(batch):
            cu_seqlens_q[b + 1] = cu_seqlens_q[b] + q_per_batch
            cu_seqlens_k[b + 1] = cu_seqlens_k[b] + k_per_batch
        cu_seqlens_q[-1] = total_q
        cu_seqlens_k[-1] = total_k

        q = torch.randn(
            total_q, head_q, HEAD_DIM,
            dtype=tensor_dtype, device="cuda", generator=generator
        )
        k = torch.randn(
            total_k, head_kv, HEAD_DIM,
            dtype=tensor_dtype, device="cuda", generator=generator
        )
        v = torch.randn(
            total_k, head_kv, HEAD_DIM,
            dtype=tensor_dtype, device="cuda", generator=generator
        )
        o = torch.empty(total_q, head_q, HEAD_DIM, dtype=tensor_dtype, device="cuda")

        q2k_indices = _generate_q2k_indices(
            batch, total_q, total_k, head_kv, topk,
            cu_seqlens_q, cu_seqlens_k, generator, "cuda"
        )

        return ExecutionInputs(
            tensors={
                "q": q,
                "k": k,
                "v": v,
                "o": o,
                "q2k_indices": q2k_indices,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
            },
            scalars={
                "batch": batch,
                "total_q": total_q,
                "total_k": total_k,
                "head_q": head_q,
                "head_kv": head_kv,
                "topk": topk,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "causal": causal,
            },
            output_names=("o",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        from sparse_index_utils import build_k2q_csr
        from interface import sparse_atten_func

        q = inputs.tensors["q"]
        k = inputs.tensors["k"]
        v = inputs.tensors["v"]
        q2k_indices = inputs.tensors["q2k_indices"]
        cu_seqlens_q = inputs.tensors["cu_seqlens_q"]
        cu_seqlens_k = inputs.tensors["cu_seqlens_k"]

        k2q_row_ptr, k2q_q_indices = build_k2q_csr(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            kv_block_size=BLK_KV,
            total_k=int(k.shape[0]),
            max_seqlen_k=inputs.scalars["max_seqlen_k"],
            max_seqlen_q=inputs.scalars["max_seqlen_q"],
            qhead_per_kv=inputs.scalars["head_q"] // inputs.scalars["head_kv"],
        )

        o = sparse_atten_func(
            q=q,
            k=k,
            v=v,
            k2q_row_ptr=k2q_row_ptr,
            k2q_q_indices=k2q_q_indices,
            topK=inputs.scalars["topk"],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=inputs.scalars["max_seqlen_q"],
            max_seqlen_k=inputs.scalars["max_seqlen_k"],
            blk_kv=BLK_KV,
            causal=inputs.scalars["causal"],
            return_softmax_lse=False,
        )
        return {"o": o}

    def benchmark_reference(inputs: ExecutionInputs):
        from sparse_index_utils import build_k2q_csr
        from interface import sparse_atten_func

        q = inputs.tensors["q"]
        k = inputs.tensors["k"]
        v = inputs.tensors["v"]
        q2k_indices = inputs.tensors["q2k_indices"]
        cu_seqlens_q = inputs.tensors["cu_seqlens_q"]
        cu_seqlens_k = inputs.tensors["cu_seqlens_k"]

        k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            kv_block_size=BLK_KV,
            total_k=int(k.shape[0]),
            max_seqlen_k=inputs.scalars["max_seqlen_k"],
            max_seqlen_q=inputs.scalars["max_seqlen_q"],
            qhead_per_kv=inputs.scalars["head_q"] // inputs.scalars["head_kv"],
            return_schedule=True,
        )

        def call():
            sparse_atten_func(
                q=q,
                k=k,
                v=v,
                k2q_row_ptr=k2q_row_ptr,
                k2q_q_indices=k2q_q_indices,
                topK=inputs.scalars["topk"],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=inputs.scalars["max_seqlen_q"],
                max_seqlen_k=inputs.scalars["max_seqlen_k"],
                blk_kv=BLK_KV,
                causal=inputs.scalars["causal"],
                return_softmax_lse=False,
                schedule=schedule,
            )

        return call

    return ReferencePlugin(make_inputs, reference, tolerance, ("o",), benchmark_reference)


def make_operation_contract(shape: dict) -> OperationContract:
    batch = int(shape["batch"])
    total_q = int(shape["total_q"])
    total_k = int(shape["total_k"])
    head_q = int(shape["head_q"])
    head_kv = int(shape["head_kv"])
    topk = int(shape["topk"])
    max_seqlen_q = int(shape.get("max_seqlen_q", total_q // batch))
    max_seqlen_k = int(shape.get("max_seqlen_k", total_k // batch))
    causal = bool(shape.get("causal", False))
    dtype = shape.get("dtype", "bf16")

    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topk must be one of {SUPPORTED_TOPK}, got {topk}")
    if head_q % head_kv != 0:
        raise ValueError(f"head_q ({head_q}) must be divisible by head_kv ({head_kv})")
    qhead_per_kv = head_q // head_kv
    if qhead_per_kv not in (1, 2, 4, 8, 16):
        raise ValueError(f"qhead_per_kv must be in {{1,2,4,8,16}}, got {qhead_per_kv}")

    task_slug = (
        f"sparse_attn_fwd_{dtype}_b{batch}_tq{total_q}_tk{total_k}"
        f"_hq{head_q}_hkv{head_kv}_topk{topk}"
        f"{'_causal' if causal else ''}"
    )

    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "q", "dtype": dtype, "shape": [total_q, head_q, HEAD_DIM], "role": "read"},
            {"name": "k", "dtype": dtype, "shape": [total_k, head_kv, HEAD_DIM], "role": "read"},
            {"name": "v", "dtype": dtype, "shape": [total_k, head_kv, HEAD_DIM], "role": "read"},
            {"name": "o", "dtype": dtype, "shape": [total_q, head_q, HEAD_DIM], "role": "write"},
            {"name": "q2k_indices", "dtype": "int32", "shape": [head_kv, total_q, topk], "role": "read"},
            {"name": "cu_seqlens_q", "dtype": "int32", "shape": [batch + 1], "role": "read"},
            {"name": "cu_seqlens_k", "dtype": "int32", "shape": [batch + 1], "role": "read"},
        ],
        "scalar_args": [
            {"name": "batch", "type": "int"},
            {"name": "total_q", "type": "int"},
            {"name": "total_k", "type": "int"},
            {"name": "head_q", "type": "int"},
            {"name": "head_kv", "type": "int"},
            {"name": "topk", "type": "int"},
            {"name": "max_seqlen_q", "type": "int"},
            {"name": "max_seqlen_k", "type": "int"},
            {"name": "causal", "type": "bool"},
        ],
        "rtol": 0.02,
        "atol": 0.02,
    }

    scalars = {
        "batch": batch,
        "total_q": total_q,
        "total_k": total_k,
        "head_q": head_q,
        "head_kv": head_kv,
        "topk": topk,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "causal": causal,
    }

    instructions = (
        f"Optimize SM100 sparse attention forward (prefill) for {dtype}. "
        f"Shapes: q=[{total_q},{head_q},{HEAD_DIM}], k=v=[{total_k},{head_kv},{HEAD_DIM}], "
        f"q2k_indices=[{head_kv},{total_q},{topk}], o=[{total_q},{head_q},{HEAD_DIM}]. "
        f"Each Q token attends to up to {topk} KV blocks of size {BLK_KV}. "
        f"GQA ratio: {qhead_per_kv}:1. {'Causal masking enabled.' if causal else 'Non-causal.'} "
        "Use CSR sparse metadata for efficient kernel scheduling."
    )

    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={
            "batch": batch,
            "total_q": total_q,
            "total_k": total_k,
            "head_q": head_q,
            "head_kv": head_kv,
            "topk": topk,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "causal": causal,
            "dtype": dtype,
        },
        task_slug=task_slug,
        reference_plugin=PLUGIN_NAME,
        spec=spec,
        scalars=scalars,
        dtype=dtype,
        instructions=instructions,
    )


PLUGIN = KernelEvalPlugin(
    name=PLUGIN_NAME,
    reference_factory=make_reference_plugin,
    contract_factory=make_operation_contract,
)
