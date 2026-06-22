import torch
import torch.nn.functional as F

from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import DEFAULT_TOLERANCES, TORCH_DTYPES, scalar_values
from kernel_evaluator.services.evaluation.types import ExecutionInputs

PLUGIN_NAME = "torch.sdpa"



def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    batch = scalars["batch"]
    qo_heads = scalars["qo_heads"]
    kv_heads = scalars["kv_heads"]
    seq_len = scalars["seq_len"]
    head_dim = scalars["head_dim"] if "head_dim" in scalars else 128
    tensor_dtype = TORCH_DTYPES[dtype]
    tolerance = DEFAULT_TOLERANCES[dtype]

    def make_inputs(seed: int) -> ExecutionInputs:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        q = torch.randn(batch, qo_heads, seq_len, head_dim, dtype=tensor_dtype, device="cuda", generator=generator)
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=tensor_dtype, device="cuda", generator=generator)
        v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=tensor_dtype, device="cuda", generator=generator)
        l_vec = torch.empty(batch, qo_heads, seq_len, 1, dtype=torch.float32, device="cuda")
        o = torch.empty(batch, qo_heads, seq_len, head_dim, dtype=tensor_dtype, device="cuda")
        return ExecutionInputs(
            tensors={"q": q, "k": k, "v": v, "l_vec": l_vec, "o": o},
            scalars={"batch": batch, "qo_heads": qo_heads, "kv_heads": kv_heads, "seq_len": seq_len, "head_dim": head_dim},
            output_names=("l_vec", "o"),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        return {
            "o": F.scaled_dot_product_attention(
                inputs.tensors["q"],
                inputs.tensors["k"],
                inputs.tensors["v"],
            )
        }

    return ReferencePlugin(make_inputs, reference, tolerance, ("o",))


def make_operation_contract(shape: dict) -> OperationContract:
    batch = int(shape["batch"])
    qo_heads = int(shape["qo_heads"])
    kv_heads = int(shape["kv_heads"])
    seq_len = int(shape["seq_len"])
    head_dim = int(shape["head_dim"] if "head_dim" in shape else 128)
    dtype = shape["dtype"]
    task_slug = f"torch_sdpa_{dtype}_b{batch}_hq{qo_heads}_hkv{kv_heads}_s{seq_len}_d{head_dim}"
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "q", "dtype": dtype, "shape": [batch, qo_heads, seq_len, head_dim], "role": "read"},
            {"name": "k", "dtype": dtype, "shape": [batch, kv_heads, seq_len, head_dim], "role": "read"},
            {"name": "v", "dtype": dtype, "shape": [batch, kv_heads, seq_len, head_dim], "role": "read"},
            {"name": "l_vec", "dtype": "fp32", "shape": [batch, qo_heads, seq_len, 1], "role": "write"},
            {"name": "o", "dtype": dtype, "shape": [batch, qo_heads, seq_len, head_dim], "role": "write"},
        ],
        "scalar_args": [
            {"name": "batch", "type": "int"},
            {"name": "qo_heads", "type": "int"},
            {"name": "kv_heads", "type": "int"},
            {"name": "seq_len", "type": "int"},
            {"name": "head_dim", "type": "int"},
        ],
        "rtol": DEFAULT_TOLERANCES[dtype][0],
        "atol": DEFAULT_TOLERANCES[dtype][1],
    }
    instructions = (
        f"Optimize scaled dot product attention for {dtype}. "
        f"Shapes: q=[{batch},{qo_heads},{seq_len},{head_dim}], "
        f"k=v=[{batch},{kv_heads},{seq_len},{head_dim}], o=[{batch},{qo_heads},{seq_len},{head_dim}]. "
        "Compute the same output as torch.nn.functional.scaled_dot_product_attention(q, k, v)."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={"batch": batch, "qo_heads": qo_heads, "kv_heads": kv_heads, "seq_len": seq_len, "head_dim": head_dim, "dtype": dtype},
        task_slug=task_slug,
        reference_plugin=PLUGIN_NAME,
        spec=spec,
        scalars={"batch": batch, "qo_heads": qo_heads, "kv_heads": kv_heads, "seq_len": seq_len, "head_dim": head_dim},
        dtype=dtype,
        instructions=instructions,
    )


PLUGIN = KernelEvalPlugin(name=PLUGIN_NAME, reference_factory=make_reference_plugin, contract_factory=make_operation_contract)
