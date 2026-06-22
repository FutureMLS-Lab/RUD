# Multi-Head Latent Attention (MLA) decode with fp8 (e4m3) inputs/outputs and a
# paged latent KV cache. The source kernel targets NVIDIA Blackwell SM100 via
# CuTe DSL (target: cutedsl, precision: fp8e4m3 in/out, fp32 accumulate/LSE);
# benchmarking candidates requires SM100-class fp8 hardware. The reference below
# is plain hardware-agnostic PyTorch.
import torch

from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import TORCH_DTYPES, scalar_values
from kernel_evaluator.services.evaluation.types import ExecutionInputs

PLUGIN_NAME = "mla.decode_fp8"

LOG2_E = 1.4426950408889634


def _build_page_table(batch_size: int, num_pages_per_seq: int, total_num_pages: int, generator: torch.Generator) -> torch.Tensor:
    free_list = torch.randperm(total_num_pages, generator=generator)
    return free_list[: batch_size * num_pages_per_seq].reshape(batch_size, num_pages_per_seq).to(torch.int32).to("cuda")


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    batch_size = scalars["batch_size"]
    seq_len_q = scalars["seq_len_q"]
    num_heads = scalars["num_heads"]
    latent_dim = scalars["latent_dim"]
    rope_dim = scalars["rope_dim"]
    page_size = scalars["page_size"]
    max_sequence_kv = scalars["max_sequence_kv"]
    num_pages_per_seq = scalars["num_pages_per_seq"]
    total_num_pages = scalars["total_num_pages"]
    softmax_scale = scalars["softmax_scale"]
    output_scale = scalars["output_scale"]
    fp8_dtype = TORCH_DTYPES[dtype]

    def make_inputs(seed: int) -> ExecutionInputs:
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        cuda_generator = torch.Generator(device="cuda").manual_seed(seed)
        q_latent = (torch.randn(batch_size, seq_len_q, num_heads, latent_dim, dtype=torch.float32, device="cuda", generator=cuda_generator) * 0.5).to(fp8_dtype)
        q_rope = (torch.randn(batch_size, seq_len_q, num_heads, rope_dim, dtype=torch.float32, device="cuda", generator=cuda_generator) * 0.5).to(fp8_dtype)
        c_latent = (torch.randn(total_num_pages, page_size, latent_dim, dtype=torch.float32, device="cuda", generator=cuda_generator) * 0.5).to(fp8_dtype)
        c_rope = (torch.randn(total_num_pages, page_size, rope_dim, dtype=torch.float32, device="cuda", generator=cuda_generator) * 0.5).to(fp8_dtype)
        page_table = _build_page_table(batch_size, num_pages_per_seq, total_num_pages, cpu_generator)
        cache_seqs = torch.full((batch_size,), max_sequence_kv, dtype=torch.int32, device="cuda")
        o = torch.empty(batch_size, seq_len_q, num_heads, latent_dim, dtype=fp8_dtype, device="cuda")
        lse = torch.empty(batch_size, seq_len_q, num_heads, dtype=torch.float32, device="cuda")
        return ExecutionInputs(
            tensors={
                "q_latent": q_latent,
                "q_rope": q_rope,
                "c_latent": c_latent,
                "c_rope": c_rope,
                "page_table": page_table,
                "cache_seqs": cache_seqs,
                "o": o,
                "lse": lse,
            },
            scalars={
                "batch_size": batch_size,
                "seq_len_q": seq_len_q,
                "num_heads": num_heads,
                "latent_dim": latent_dim,
                "rope_dim": rope_dim,
                "page_size": page_size,
                "max_sequence_kv": max_sequence_kv,
                "num_pages_per_seq": num_pages_per_seq,
                "total_num_pages": total_num_pages,
                "softmax_scale": softmax_scale,
                "output_scale": output_scale,
            },
            output_names=("o", "lse"),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        # TODO(review): verify this reference is mathematically correct before trusting any benchmark results.
        # Mirrors torch_reference_mla in the source kernel (non-causal decode):
        # q = cat(q_latent, q_rope); K = cat(c_latent, c_rope) gathered via page
        # table; V = c_latent gathered via page table; all heads share one KV.
        # LSE is in log2 domain (lse = log2(sum_k exp2(s_k * scale * log2(e)))),
        # matching the kernel's lse_ref; flag if the harness expects natural log.
        q = torch.cat([inputs.tensors["q_latent"].float(), inputs.tensors["q_rope"].float()], dim=-1)
        kv_paged = torch.cat([inputs.tensors["c_latent"].float(), inputs.tensors["c_rope"].float()], dim=-1)
        v_paged = inputs.tensors["c_latent"].float()
        page_table = inputs.tensors["page_table"].long()
        cache_seqs = inputs.tensors["cache_seqs"].cpu()
        o = torch.empty(batch_size, seq_len_q, num_heads, latent_dim, dtype=torch.float32, device=q.device)
        lse = torch.empty(batch_size, seq_len_q, num_heads, dtype=torch.float32, device=q.device)
        for batch_idx in range(batch_size):
            length = int(cache_seqs[batch_idx].item())
            pages = page_table[batch_idx]
            k_b = kv_paged.index_select(0, pages).reshape(-1, latent_dim + rope_dim)[:length]
            v_b = v_paged.index_select(0, pages).reshape(-1, latent_dim)[:length]
            scores = torch.einsum("qhd,kd->qhk", q[batch_idx], k_b) * softmax_scale
            score_max = scores.amax(dim=-1, keepdim=True)
            exp2_scores = torch.exp2((scores - score_max) * LOG2_E)
            lse[batch_idx] = (score_max * LOG2_E + torch.log2(exp2_scores.sum(dim=-1, keepdim=True))).squeeze(-1)
            probs = torch.softmax(scores, dim=-1)
            o[batch_idx] = torch.einsum("qhk,kd->qhd", probs, v_b) * output_scale
        return {"o": o.to(fp8_dtype), "lse": lse}

    # 0.13 absolute tolerance matches the source kernel's own fp8 ref check.
    return ReferencePlugin(make_inputs, reference, (1e-5, 0.13), ("o", "lse"))


def make_operation_contract(shape: dict) -> OperationContract:
    batch_size = int(shape["batch_size"])
    seq_len_q = int(shape["seq_len_q"] if "seq_len_q" in shape else 1)
    num_heads = int(shape["num_heads"])
    latent_dim = int(shape["latent_dim"] if "latent_dim" in shape else 512)
    rope_dim = int(shape["rope_dim"] if "rope_dim" in shape else 64)
    page_size = int(shape["page_size"])
    max_sequence_kv = int(shape["max_sequence_kv"])
    num_pages_per_seq = int(shape["num_pages_per_seq"] if "num_pages_per_seq" in shape else (max_sequence_kv + page_size - 1) // page_size)
    total_num_pages = int(shape["total_num_pages"] if "total_num_pages" in shape else batch_size * num_pages_per_seq)
    softmax_scale = float(shape["softmax_scale"] if "softmax_scale" in shape else (latent_dim + rope_dim) ** -0.5)
    output_scale = float(shape["output_scale"] if "output_scale" in shape else 1.0)
    dtype = shape["dtype"] if "dtype" in shape else "fp8"
    task_slug = (
        f"mla_decode_{dtype}_b{batch_size}_sq{seq_len_q}_h{num_heads}"
        f"_dl{latent_dim}_dr{rope_dim}_s{max_sequence_kv}"
    )
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "q_latent", "dtype": dtype, "shape": [batch_size, seq_len_q, num_heads, latent_dim], "role": "read"},
            {"name": "q_rope", "dtype": dtype, "shape": [batch_size, seq_len_q, num_heads, rope_dim], "role": "read"},
            {"name": "c_latent", "dtype": dtype, "shape": [total_num_pages, page_size, latent_dim], "role": "read"},
            {"name": "c_rope", "dtype": dtype, "shape": [total_num_pages, page_size, rope_dim], "role": "read"},
            {"name": "page_table", "dtype": "int32", "shape": [batch_size, num_pages_per_seq], "role": "read"},
            {"name": "cache_seqs", "dtype": "int32", "shape": [batch_size], "role": "read"},
            {"name": "o", "dtype": dtype, "shape": [batch_size, seq_len_q, num_heads, latent_dim], "role": "write"},
            {"name": "lse", "dtype": "fp32", "shape": [batch_size, seq_len_q, num_heads], "role": "write"},
        ],
        "scalar_args": [
            {"name": "batch_size", "type": "int"},
            {"name": "seq_len_q", "type": "int"},
            {"name": "num_heads", "type": "int"},
            {"name": "latent_dim", "type": "int"},
            {"name": "rope_dim", "type": "int"},
            {"name": "page_size", "type": "int"},
            {"name": "max_sequence_kv", "type": "int"},
            {"name": "num_pages_per_seq", "type": "int"},
            {"name": "total_num_pages", "type": "int"},
            {"name": "softmax_scale", "type": "float"},
            {"name": "output_scale", "type": "float"},
        ],
        "rtol": 1e-5,
        "atol": 0.13,
    }
    scalars = {
        "batch_size": batch_size,
        "seq_len_q": seq_len_q,
        "num_heads": num_heads,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "page_size": page_size,
        "max_sequence_kv": max_sequence_kv,
        "num_pages_per_seq": num_pages_per_seq,
        "total_num_pages": total_num_pages,
        "softmax_scale": softmax_scale,
        "output_scale": output_scale,
    }
    instructions = (
        f"Optimize Multi-Head Latent Attention (MLA) decode for {dtype}. "
        f"Shapes: q_latent=[{batch_size},{seq_len_q},{num_heads},{latent_dim}], "
        f"q_rope=[{batch_size},{seq_len_q},{num_heads},{rope_dim}], "
        f"c_latent=[{total_num_pages},{page_size},{latent_dim}], c_rope=[{total_num_pages},{page_size},{rope_dim}], "
        f"page_table=[{batch_size},{num_pages_per_seq}], o=[{batch_size},{seq_len_q},{num_heads},{latent_dim}], "
        f"lse=[{batch_size},{seq_len_q},{num_heads}]. "
        "Compute o = softmax(cat(q_latent,q_rope) @ cat(c_latent,c_rope)^T * softmax_scale) @ c_latent * output_scale "
        "with the KV cache gathered through page_table up to cache_seqs tokens per batch; all heads share one latent KV. "
        "Also write the log2-domain LSE per (batch, q_token, head)."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={
            "batch_size": batch_size,
            "seq_len_q": seq_len_q,
            "num_heads": num_heads,
            "latent_dim": latent_dim,
            "rope_dim": rope_dim,
            "page_size": page_size,
            "max_sequence_kv": max_sequence_kv,
            "num_pages_per_seq": num_pages_per_seq,
            "total_num_pages": total_num_pages,
            "softmax_scale": softmax_scale,
            "output_scale": output_scale,
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
