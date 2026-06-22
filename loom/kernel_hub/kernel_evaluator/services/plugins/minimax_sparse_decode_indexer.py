"""Minimax sparse decode indexer plugin.

Wraps `flash_decode_with_topk_idx(..., disable_index_value=True)` from
sglang's minimax_sparse_ops — the score-only "indexer" path used in
production by Minimax-M3-FP4 to pick top-k KV blocks for the main attention.

The kernel returns only `topk_idx` ([num_q_heads, batch, topk] int32);
`o`/`v_cache`/`sink` are unused in this mode. Correctness for `topk_idx`
is checked as per-(head, batch) set-equality on the valid (non-sentinel)
entries, since ties in block scores (forced by init_blocks/local_blocks)
make raw index equality the wrong semantic.

The plugin uses `flash_decode_with_topk_idx` as both the correctness oracle
and the timed baseline. A candidate that simply re-calls it should benchmark
at ~1.0x speedup, correct=True — that's the smoke test for harness wiring.

The kernel comes from sglang at
sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx,
but the eval container mounts only that subtree as a top-level package at
/opt/sglang_kernel_root/minimax_sparse_ops to avoid sglang's package __init__
(which pulls orjson + transformers). So we import it as
`minimax_sparse_ops.decode.flash_with_topk_idx`. See Dockerfile + docker-compose.
"""

from collections.abc import Callable

import torch

from kernel_evaluator.services.evaluation.types import ExecutionInputs
from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import TORCH_DTYPES

PLUGIN_NAME = "minimax_sparse.decode_indexer"


def _topk_idx_set_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape:
        return False
    a = actual.detach().to("cpu")
    e = expected.detach().to("cpu")
    nqh, bs, _ = a.shape
    for h in range(nqh):
        for b in range(bs):
            a_set = {x for x in a[h, b].tolist() if x != -1}
            e_set = {x for x in e[h, b].tolist() if x != -1}
            if a_set != e_set:
                return False
    return True


def _call_indexer(inputs: ExecutionInputs) -> torch.Tensor:
    from minimax_sparse_ops.decode.flash_with_topk_idx import (
        flash_decode_with_topk_idx,
    )
    _, topk_idx = flash_decode_with_topk_idx(
        q=inputs.tensors["q"],
        sink=None,
        k_cache=inputs.tensors["k_cache"],
        v_cache=None,
        req_to_token=inputs.tensors["req_to_token"],
        seq_lens=inputs.tensors["seq_lens"],
        max_seqlen=int(inputs.scalars["seq_len"]),
        slot_ids=inputs.tensors["slot_ids"],
        block_size=int(inputs.scalars["block_size"]),
        topk=int(inputs.scalars["topk"]),
        init_blocks=int(inputs.scalars["init_blocks"]),
        local_blocks=int(inputs.scalars["local_blocks"]),
        score_type="max",
        disable_index_value=True,
    )
    return topk_idx


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    from kernel_evaluator.services.plugins.spec_helpers import scalar_values
    scalars = scalar_values(spec)
    batch = int(scalars["batch"])
    num_q_heads = int(scalars["num_q_heads"])
    num_kv_heads = int(scalars["num_kv_heads"])
    head_dim = int(scalars["head_dim"])
    block_size = int(scalars["block_size"])
    seq_len = int(scalars["seq_len"])
    topk = int(scalars["topk"])
    init_blocks = int(scalars["init_blocks"])
    local_blocks = int(scalars["local_blocks"])
    tensor_dtype = TORCH_DTYPES[dtype]
    # topk_idx is checked via custom comparator; the tolerance is only a
    # placeholder for any future numeric outputs.
    tolerance = (1e-2, 1e-2)
    max_kv_len = seq_len
    # Headroom around batch and seq_len so slot_ids and req_to_token contents are
    # *nontrivial random indices*, not the identity permutation. Production has
    # max_reqs~41 and max_slots~2M independent of batch/seq_len; we approximate
    # that here with generous slack. This is what stops candidates from stripping
    # the req_to_token / slot_ids indirection (which would be silently wrong in
    # production where the KV cache is paged).
    max_reqs = max(batch * 4, 64)
    max_slots = max(batch * max_kv_len * 4, 4096)

    def make_inputs(seed: int) -> ExecutionInputs:
        g = torch.Generator(device="cuda").manual_seed(seed)
        cpu_g = torch.Generator().manual_seed(seed ^ 0xDEADBEEF)
        q = torch.randn(batch, num_q_heads, head_dim, dtype=tensor_dtype, device="cuda", generator=g)
        k_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=tensor_dtype, device="cuda", generator=g)
        # Non-trivial slot_ids: random unique request indices in [0, max_reqs).
        slot_ids = torch.randperm(max_reqs, generator=cpu_g)[:batch].to(torch.int64).cuda()
        # Non-trivial req_to_token: each used slot maps to a random permutation
        # of distinct k_cache slots in [0, max_slots). Unused rows stay 0 (never read).
        req_to_token = torch.zeros(max_reqs, max_kv_len, dtype=torch.int32, device="cuda")
        for sid in slot_ids.cpu().tolist():
            perm = torch.randperm(max_slots, generator=cpu_g)[:max_kv_len].to(torch.int32)
            req_to_token[sid] = perm.cuda()
        seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
        topk_idx = torch.full((num_q_heads, batch, topk), -1, dtype=torch.int32, device="cuda")
        return ExecutionInputs(
            tensors={
                "q": q,
                "k_cache": k_cache,
                "req_to_token": req_to_token,
                "seq_lens": seq_lens,
                "slot_ids": slot_ids,
                "topk_idx": topk_idx,
            },
            scalars={
                "batch": batch,
                "num_q_heads": num_q_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "seq_len": seq_len,
                "topk": topk,
                "init_blocks": init_blocks,
                "local_blocks": local_blocks,
            },
            output_names=("topk_idx",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        return {"topk_idx": _call_indexer(inputs).clone()}

    def benchmark_reference(inputs: ExecutionInputs) -> Callable[[], None]:
        topk_idx_buf = inputs.tensors["topk_idx"]
        def call() -> None:
            out = _call_indexer(inputs)
            topk_idx_buf.copy_(out)
        # Warm up Triton autotune & JIT outside the timed window.
        call()
        return call

    return ReferencePlugin(
        make_inputs=make_inputs,
        reference=reference,
        tolerances=tolerance,
        output_names=("topk_idx",),
        benchmark_reference=benchmark_reference,
        output_comparators={"topk_idx": _topk_idx_set_equal},
    )


REQUIRED_SHAPE_KEYS = (
    "batch", "num_q_heads", "num_kv_heads", "head_dim", "block_size",
    "seq_len", "topk", "init_blocks", "local_blocks", "dtype",
)


def make_operation_contract(shape: dict) -> OperationContract:
    missing = [k for k in REQUIRED_SHAPE_KEYS if k not in shape]
    if missing:
        raise ValueError(
            f"minimax_sparse.decode_indexer shape missing required keys: {missing}. "
            f"Required keys: {list(REQUIRED_SHAPE_KEYS)}."
        )
    batch = int(shape["batch"])
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    block_size = int(shape["block_size"])
    seq_len = int(shape["seq_len"])
    topk = int(shape["topk"])
    init_blocks = int(shape["init_blocks"])
    local_blocks = int(shape["local_blocks"])
    dtype = shape["dtype"]
    if num_q_heads != num_kv_heads:
        raise ValueError(
            f"indexer path expects num_q_heads == num_kv_heads, got "
            f"num_q_heads={num_q_heads} num_kv_heads={num_kv_heads}"
        )
    max_kv_len = seq_len
    # Keep these in sync with make_reference_plugin.make_inputs above.
    max_reqs = max(batch * 4, 64)
    max_slots = max(batch * max_kv_len * 4, 4096)

    task_slug = (
        f"minimax_sparse_decode_indexer_{dtype}"
        f"_b{batch}_hq{num_q_heads}_hkv{num_kv_heads}_d{head_dim}"
        f"_blk{block_size}_s{seq_len}_tk{topk}_ib{init_blocks}_lb{local_blocks}"
    )

    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "q", "dtype": dtype, "shape": [batch, num_q_heads, head_dim], "role": "read"},
            {"name": "k_cache", "dtype": dtype, "shape": [max_slots, num_kv_heads, head_dim], "role": "read"},
            {"name": "req_to_token", "dtype": "int32", "shape": [max_reqs, max_kv_len], "role": "read"},
            {"name": "seq_lens", "dtype": "int32", "shape": [batch], "role": "read"},
            {"name": "slot_ids", "dtype": "int64", "shape": [batch], "role": "read"},
            {"name": "topk_idx", "dtype": "int32", "shape": [num_q_heads, batch, topk], "role": "write"},
        ],
        "scalar_args": [
            {"name": "batch", "type": "int"},
            {"name": "num_q_heads", "type": "int"},
            {"name": "num_kv_heads", "type": "int"},
            {"name": "head_dim", "type": "int"},
            {"name": "block_size", "type": "int"},
            {"name": "seq_len", "type": "int"},
            {"name": "topk", "type": "int"},
            {"name": "init_blocks", "type": "int"},
            {"name": "local_blocks", "type": "int"},
        ],
        "rtol": 1e-2,
        "atol": 1e-2,
    }

    scalars = {
        "batch": batch,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "seq_len": seq_len,
        "topk": topk,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
    }

    instructions = (
        f"SCOPE: optimize ONLY `_decode_score_kernel` from "
        f"minimax_sparse_ops.decode.flash_with_topk_idx for {dtype}. "
        f"Do NOT modify `_topk_index_partial_kernel` or `_topk_index_merge_kernel` "
        f"-- import them unchanged from minimax_sparse_ops.decode.flash_with_topk_idx "
        f"and wire them into your prepare(). "
        f"The full operation mirrors flash_decode_with_topk_idx(disable_index_value=True). "
        f"Inputs: q=[{batch},{num_q_heads},{head_dim}], "
        f"k_cache=[{max_slots},{num_kv_heads},{head_dim}], "
        f"req_to_token=[{max_reqs},{max_kv_len}] int32 (paged-slot lookup table), "
        f"seq_lens=[{batch}] int32, slot_ids=[{batch}] int64 (logical request indices into req_to_token; "
        f"NOT trivially [0..batch) -- DO NOT assume identity). "
        f"For each valid position p in [0, seq_lens[b]), the physical K-cache slot is "
        f"`req_to_token[slot_ids[b], p]`; req_to_token is a random permutation, NOT identity, so "
        f"you MUST perform the indirection. "
        f"Output: topk_idx=[{num_q_heads},{batch},{topk}] int32 (write into the "
        f"pre-allocated tensor; sentinel -1 for fewer-than-topk valid blocks). "
        f"Scalars: block_size={block_size}, topk={topk}, init_blocks={init_blocks}, "
        f"local_blocks={local_blocks}, score_type=max. "
        f"Init blocks (first init_blocks) and local blocks (last local_blocks) are "
        f"forced into the top-k regardless of score. "
        f"Your `_decode_score_kernel` must produce a `score` tensor of shape "
        f"[num_q_heads, batch, ceil(max_kv_len/block_size)] dtype float32, filled with "
        f"-inf for padding, with init/local positions force-marked (1e30 / 1e29) so the "
        f"downstream topk kernels can consume it unchanged. "
        f"Correctness is set-equality of valid topk_idx entries per (head, batch); "
        f"order within the top-k may differ."
    )

    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={
            "batch": batch,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "seq_len": seq_len,
            "topk": topk,
            "init_blocks": init_blocks,
            "local_blocks": local_blocks,
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
