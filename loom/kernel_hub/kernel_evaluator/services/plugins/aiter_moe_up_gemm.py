import random

import torch

from kernel_evaluator.services.evaluation.types import ExecutionInputs
from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import scalar_values

PLUGIN_NAME = "aiter.moe_up_gemm"


def _make_inputs(seed: int, scalars: dict) -> ExecutionInputs:
    import aiter
    from aiter import QuantType, dtypes
    from aiter import fused_dynamic_mxfp4_quant_moe_sort
    from aiter.fused_moe import fused_topk, moe_sorting
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    torch.manual_seed(seed)
    random.seed(seed)
    model_dim = scalars["model_dim"]
    inter_dim = scalars["inter_dim"]
    num_experts = scalars["num_experts"]
    top_k = scalars["top_k"]
    block_m = scalars["block_m"]
    num_tokens = scalars["num_tokens"]
    w1_bf16 = torch.randn(num_experts, inter_dim * 2, model_dim, dtype=torch.bfloat16, device="cuda")
    w2_bf16 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device="cuda")
    hip_quant = aiter.get_hip_quant(QuantType.per_1x32)
    e1, g1, d1 = w1_bf16.shape
    e2, d2, g2 = w2_bf16.shape
    w1_qt, w1_scale = hip_quant(w1_bf16.view(e1 * g1, d1), quant_dtype=dtypes.fp4x2)
    w2_qt, _w2_scale = hip_quant(w2_bf16.view(e2 * d2, g2), quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(e1, g1, d1 // 2)
    w2_qt = w2_qt.view(e2, d2, g2 // 2)
    w1_qt = shuffle_weight(w1_qt.contiguous(), (16, 16))
    w2_qt = shuffle_weight(w2_qt.contiguous(), (16, 16))
    w1_scale = e8m0_shuffle(w1_scale).view(e1, g1, -1)
    hidden_bf16 = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = fused_topk(hidden_bf16, score, top_k, renormalize=True)
    sorted_ids, _sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights.float(),
        num_experts,
        model_dim,
        torch.bfloat16,
        block_m,
    )
    hidden_states, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_bf16,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=num_tokens,
        topk=top_k,
        block_size=block_m,
    )
    out = torch.empty(num_tokens, top_k, inter_dim, dtype=torch.bfloat16, device="cuda")
    baseline_out = torch.empty_like(out)
    return ExecutionInputs(
        tensors={
            "hidden_states": hidden_states,
            "w1": w1_qt,
            "sorted_ids": sorted_ids,
            "sorted_expert_ids": sorted_expert_ids,
            "num_valid_ids": num_valid_ids,
            "out": out,
            "a1_scale": a1_scale,
            "w1_scale": w1_scale,
            "w2_qt": w2_qt,
            "baseline_out": baseline_out,
        },
        scalars=scalars,
        output_names=("out",),
    )


def _run_aiter_baseline(inputs: ExecutionInputs, output_name: str) -> None:
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import ck_moe_stage1

    ck_moe_stage1(
        inputs.tensors["hidden_states"],
        inputs.tensors["w1"],
        inputs.tensors["w2_qt"],
        inputs.tensors["sorted_ids"],
        inputs.tensors["sorted_expert_ids"],
        inputs.tensors["num_valid_ids"],
        inputs.tensors[output_name],
        inputs.scalars["top_k"],
        inputs.scalars["block_m"],
        a1_scale=inputs.tensors["a1_scale"],
        w1_scale=inputs.tensors["w1_scale"].view(dtypes.fp8_e8m0),
        kernelName="",
        sorted_weights=None,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
    )


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    del dtype
    abi_scalars = scalar_values(spec)
    tensor_specs = {tensor["name"]: tensor for tensor in spec["tensor_args"]}
    hidden_shape = tensor_specs["hidden_states"]["shape"]
    w1_shape = tensor_specs["w1"]["shape"]
    out_shape = tensor_specs["out"]["shape"]
    scalars = {
        "model_dim": int(hidden_shape[1]) * 2,
        "inter_dim": int(out_shape[2]),
        "num_experts": int(w1_shape[0]),
        "top_k": abi_scalars["top_k"],
        "block_m": abi_scalars["block_m"],
        "num_tokens": int(hidden_shape[0]),
    }

    def make_inputs(seed: int) -> ExecutionInputs:
        return _make_inputs(seed, scalars)

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        _run_aiter_baseline(inputs, "baseline_out")
        return {"out": inputs.tensors["baseline_out"].clone()}

    def benchmark_reference(inputs: ExecutionInputs):
        def call():
            _run_aiter_baseline(inputs, "out")

        return call

    return ReferencePlugin(make_inputs, reference, (1e-2, 1e-2), ("out",), benchmark_reference)


def make_operation_contract(shape: dict) -> OperationContract:
    model_dim = int(shape["model_dim"])
    inter_dim = int(shape["inter_dim"])
    num_experts = int(shape["num_experts"])
    top_k = int(shape["top_k"])
    block_m = int(shape["block_m"])
    num_tokens = int(shape["num_tokens"])
    dtype = shape["dtype"]
    task_slug = f"aiter_moe_up_gemm_{dtype}_m{num_tokens}_n{inter_dim * 2}_k{model_dim}"
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "hidden_states", "dtype": "uint8", "shape": [num_tokens, model_dim // 2], "role": "read"},
            {"name": "w1", "dtype": "uint8", "shape": [num_experts, inter_dim * 2, model_dim // 2], "role": "read"},
            {"name": "sorted_ids", "dtype": "int32", "shape": ["padded_slots"], "role": "read"},
            {"name": "sorted_expert_ids", "dtype": "int32", "shape": ["expert_blocks"], "role": "read"},
            {"name": "num_valid_ids", "dtype": "int32", "shape": [2], "role": "read"},
            {"name": "out", "dtype": "bf16", "shape": [num_tokens, top_k, inter_dim], "role": "write"},
            {"name": "a1_scale", "dtype": "uint8", "shape": ["scale_slots", model_dim // 32], "role": "read"},
            {"name": "w1_scale", "dtype": "uint8", "shape": [num_experts, inter_dim * 2, model_dim // 32], "role": "read"},
        ],
        "scalar_args": [
            {"name": "top_k", "type": "int"},
            {"name": "block_m", "type": "int"},
        ],
        "rtol": 1e-2,
        "atol": 1e-2,
    }
    scalars = {
        "top_k": top_k,
        "block_m": block_m,
    }
    instructions = (
        f"Optimize MoE decode stage-1 MXFP4 up GEMM for M={num_tokens}, N={inter_dim * 2}, K={model_dim}. "
        "Export dispatch(hidden_states, w1, sorted_ids, sorted_expert_ids, num_valid_ids, out, a1_scale, w1_scale, top_k, block_m). "
        "hidden_states is unsorted fp4x2 with shape [num_tokens, model_dim/2]. "
        "w1 is preshuffled fp4x2 gate+up weights with shape [num_experts, 2*inter_dim, model_dim/2]. "
        "Write bf16 silu(gate) * up into out with shape [num_tokens, top_k, inter_dim]. Shape specialization is allowed."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "block_m": block_m,
            "num_tokens": num_tokens,
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
