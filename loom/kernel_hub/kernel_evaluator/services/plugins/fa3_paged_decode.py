import torch

from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import scalar_values
from kernel_evaluator.services.evaluation.types import ExecutionInputs

PLUGIN_NAME = "fa3.paged_decode"



def _build_block_table(seq_lens: torch.Tensor, page_size: int, npps: int, total_pages: int, generator: torch.Generator) -> torch.Tensor:
    free_list = torch.randperm(total_pages, generator=generator).tolist()
    cursor = 0
    block_tables = torch.zeros(seq_lens.shape[0], npps, dtype=torch.int32)
    for batch_idx in range(seq_lens.shape[0]):
        length = int(seq_lens[batch_idx].item())
        real_pages = (length + page_size - 1) // page_size
        block_tables[batch_idx, :real_pages] = torch.tensor(free_list[cursor:cursor + real_pages], dtype=torch.int32)
        cursor += real_pages
        if real_pages < npps:
            block_tables[batch_idx, real_pages:] = block_tables[batch_idx, 0]
    return block_tables.to("cuda")


def _reference_sdpa(q, k_cache, v_cache, block_tables, seq_lens, page_size, scale):
    batch_size, num_heads_qo, head_dim = q.shape
    num_heads_kv = k_cache.shape[1]
    group_size = num_heads_qo // num_heads_kv
    out = torch.empty_like(q, dtype=torch.float32)
    qf = q.float()
    for batch_idx in range(batch_size):
        length = int(seq_lens[batch_idx].item())
        real_pages = (length + page_size - 1) // page_size
        page_ids = block_tables[batch_idx, :real_pages].long()
        kp = k_cache.index_select(0, page_ids)
        vp = v_cache.index_select(0, page_ids)
        kb = kp.permute(1, 0, 2, 3).reshape(num_heads_kv, real_pages * page_size, head_dim)[:, :length, :]
        vb = vp.permute(1, 0, 2, 3).reshape(num_heads_kv, real_pages * page_size, head_dim)[:, :length, :]
        kb = kb.repeat_interleave(group_size, dim=0).float()
        vb = vb.repeat_interleave(group_size, dim=0).float()
        scores = (qf[batch_idx].unsqueeze(1) @ kb.transpose(-2, -1)).squeeze(1) * scale
        probs = torch.softmax(scores, dim=-1)
        out[batch_idx] = (probs.unsqueeze(1) @ vb).squeeze(1)
    return out


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    batch_size = scalars["batch_size"]
    num_heads_qo = scalars["num_heads_qo"]
    num_heads_kv = scalars["num_heads_kv"]
    head_dim = scalars["head_dim"]
    page_size = scalars["page_size"]
    max_sequence_kv = scalars["max_sequence_kv"]
    num_pages_per_seq = scalars["num_pages_per_seq"]
    total_num_pages = scalars["total_num_pages"]
    workspace_bytes = scalars["workspace_bytes"]
    scale = scalars["scale"]

    def make_inputs(seed: int) -> ExecutionInputs:
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        cuda_generator = torch.Generator(device="cuda").manual_seed(seed)
        seq_lens_cpu = torch.full((batch_size,), max_sequence_kv, dtype=torch.int32)
        actual_seq_lens_kv = seq_lens_cpu.view(batch_size, 1, 1, 1).to("cuda")
        k_cache = torch.randn(
            total_num_pages, num_heads_kv, page_size, head_dim,
            dtype=torch.bfloat16, device="cuda", generator=cuda_generator,
        )
        v_cache = torch.randn(
            total_num_pages, num_heads_kv, page_size, head_dim,
            dtype=torch.bfloat16, device="cuda", generator=cuda_generator,
        )
        k_cache_nhd = k_cache.permute(0, 2, 1, 3).contiguous()
        v_cache_nhd = v_cache.permute(0, 2, 1, 3).contiguous()
        block_tables = _build_block_table(seq_lens_cpu, page_size, num_pages_per_seq, total_num_pages, cpu_generator)
        q = torch.randn(
            batch_size, num_heads_qo, head_dim,
            dtype=torch.bfloat16, device="cuda", generator=cuda_generator,
        )
        workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        out = torch.empty(batch_size, num_heads_qo, head_dim, dtype=torch.bfloat16, device="cuda")
        return ExecutionInputs(
            tensors={
                "q": q,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "k_cache_nhd": k_cache_nhd,
                "v_cache_nhd": v_cache_nhd,
                "workspace_buffer": workspace_buffer,
                "actual_seq_lens_kv": actual_seq_lens_kv,
                "block_tables": block_tables,
                "out": out,
            },
            scalars={
                "batch_size": batch_size,
                "num_heads_qo": num_heads_qo,
                "num_heads_kv": num_heads_kv,
                "head_dim": head_dim,
                "page_size": page_size,
                "max_sequence_kv": max_sequence_kv,
                "num_pages_per_seq": num_pages_per_seq,
                "total_num_pages": total_num_pages,
                "workspace_bytes": workspace_bytes,
                "scale": scale,
            },
            output_names=("out",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        ref = _reference_sdpa(
            inputs.tensors["q"],
            inputs.tensors["k_cache"],
            inputs.tensors["v_cache"],
            inputs.tensors["block_tables"],
            inputs.tensors["actual_seq_lens_kv"].flatten().cpu(),
            page_size,
            scale,
        )
        return {"out": ref.to(torch.bfloat16)}

    def benchmark_reference(inputs: ExecutionInputs):
        from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

        seqused_k = inputs.tensors["actual_seq_lens_kv"].flatten().to(torch.int32)
        cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        scheduler_metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_sequence_kv,
            num_heads_q=num_heads_qo,
            num_heads_kv=num_heads_kv,
            headdim=head_dim,
            cache_seqlens=seqused_k,
            qkv_dtype=torch.bfloat16,
            cu_seqlens_q=cu_seqlens_q,
            page_size=page_size,
            causal=True,
            num_splits=64,
        )
        out = torch.zeros_like(inputs.tensors["q"])

        def call():
            flash_attn_varlen_func(
                q=inputs.tensors["q"],
                k=inputs.tensors["k_cache_nhd"],
                v=inputs.tensors["v_cache_nhd"],
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=seqused_k,
                max_seqlen_k=max_sequence_kv,
                softmax_scale=scale,
                causal=True,
                block_table=inputs.tensors["block_tables"],
                scheduler_metadata=scheduler_metadata,
                num_splits=64,
                fa_version=3,
            )

        return call

    return ReferencePlugin(make_inputs, reference, (2e-2, 2e-2), ("out",), benchmark_reference)


def make_operation_contract(shape: dict) -> OperationContract:
    batch_size = int(shape["batch_size"])
    num_heads_qo = int(shape["num_heads_qo"])
    num_heads_kv = int(shape["num_heads_kv"])
    head_dim = int(shape["head_dim"])
    page_size = int(shape["page_size"])
    max_sequence_kv = int(shape["max_sequence_kv"])
    num_pages_per_seq = int(shape["num_pages_per_seq"] if "num_pages_per_seq" in shape else (max_sequence_kv + page_size - 1) // page_size)
    total_num_pages = int(shape["total_num_pages"] if "total_num_pages" in shape else batch_size * num_pages_per_seq)
    workspace_bytes = int(shape["workspace_bytes"] if "workspace_bytes" in shape else 8 * 1024 * 1024)
    scale = float(shape["scale"] if "scale" in shape else head_dim ** -0.5)
    dtype = shape["dtype"] if "dtype" in shape else "bf16"
    task_slug = (
        f"fa3_paged_decode_{dtype}_b{batch_size}_hq{num_heads_qo}_hkv{num_heads_kv}"
        f"_d{head_dim}_s{max_sequence_kv}"
    )
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "q", "dtype": dtype, "shape": [batch_size, num_heads_qo, head_dim], "role": "read"},
            {"name": "k_cache", "dtype": dtype, "shape": [total_num_pages, num_heads_kv, page_size, head_dim], "role": "read"},
            {"name": "v_cache", "dtype": dtype, "shape": [total_num_pages, num_heads_kv, page_size, head_dim], "role": "read"},
            {"name": "workspace_buffer", "dtype": "uint8", "shape": [workspace_bytes], "role": "read"},
            {"name": "actual_seq_lens_kv", "dtype": "int32", "shape": [batch_size, 1, 1, 1], "role": "read"},
            {"name": "block_tables", "dtype": "int32", "shape": [batch_size, num_pages_per_seq], "role": "read"},
            {"name": "out", "dtype": dtype, "shape": [batch_size, num_heads_qo, head_dim], "role": "write"},
        ],
        "scalar_args": [
            {"name": "batch_size", "type": "int"},
            {"name": "num_heads_qo", "type": "int"},
            {"name": "num_heads_kv", "type": "int"},
            {"name": "head_dim", "type": "int"},
            {"name": "page_size", "type": "int"},
            {"name": "max_sequence_kv", "type": "int"},
            {"name": "num_pages_per_seq", "type": "int"},
            {"name": "total_num_pages", "type": "int"},
            {"name": "workspace_bytes", "type": "int"},
            {"name": "scale", "type": "float"},
        ],
        "rtol": 0.02,
        "atol": 0.02,
    }
    scalars = {
        "batch_size": batch_size,
        "num_heads_qo": num_heads_qo,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "page_size": page_size,
        "max_sequence_kv": max_sequence_kv,
        "num_pages_per_seq": num_pages_per_seq,
        "total_num_pages": total_num_pages,
        "workspace_bytes": workspace_bytes,
        "scale": scale,
    }
    instructions = (
        f"Optimize paged decode attention for {dtype}. "
        f"Shapes: q=[{batch_size},{num_heads_qo},{head_dim}], "
        f"k_cache=v_cache=[{total_num_pages},{num_heads_kv},{page_size},{head_dim}], "
        f"block_tables=[{batch_size},{num_pages_per_seq}], out=[{batch_size},{num_heads_qo},{head_dim}]. "
        "Compute grouped-query single-token decode attention over max_sequence_kv tokens using block_tables."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={
            "batch_size": batch_size,
            "num_heads_qo": num_heads_qo,
            "num_heads_kv": num_heads_kv,
            "head_dim": head_dim,
            "page_size": page_size,
            "max_sequence_kv": max_sequence_kv,
            "num_pages_per_seq": num_pages_per_seq,
            "total_num_pages": total_num_pages,
            "workspace_bytes": workspace_bytes,
            "scale": scale,
            "dtype": dtype,
        },
        task_slug=task_slug,
        reference_plugin=PLUGIN_NAME,
        spec=spec,
        scalars=scalars,
        dtype=dtype,
        instructions=instructions,
    )


PLUGIN = KernelEvalPlugin(name=PLUGIN_NAME, reference_factory=make_reference_plugin, contract_factory=make_operation_contract)
