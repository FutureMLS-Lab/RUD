import torch

from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import TORCH_DTYPES, scalar_values
from kernel_evaluator.services.evaluation.types import ExecutionInputs

PLUGIN_NAME = "torch.fp8_gemm"



def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    dim_m = scalars["dim_m"]
    dim_n = scalars["dim_n"]
    dim_k = scalars["dim_k"]
    fp8_dtype = TORCH_DTYPES[dtype]

    def make_inputs(seed: int) -> ExecutionInputs:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        a = (torch.randn(dim_m, dim_k, dtype=torch.float32, device="cuda", generator=generator) * 0.1).to(fp8_dtype)
        bt = (torch.randn(dim_n, dim_k, dtype=torch.float32, device="cuda", generator=generator) * 0.1).to(fp8_dtype)
        c = torch.empty(dim_m, dim_n, dtype=fp8_dtype, device="cuda")
        return ExecutionInputs(
            tensors={"a": a, "bt": bt, "c": c},
            scalars={"dim_m": dim_m, "dim_n": dim_n, "dim_k": dim_k},
            output_names=("c",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        return {"c": (inputs.tensors["a"].float() @ inputs.tensors["bt"].float().T).to(fp8_dtype)}

    return ReferencePlugin(make_inputs, reference, (0.25, 0.25), ("c",))


def make_operation_contract(shape: dict) -> OperationContract:
    dim_m = int(shape["m"])
    dim_n = int(shape["n"])
    dim_k = int(shape["k"])
    dtype = shape["dtype"] if "dtype" in shape else "fp8"
    task_slug = f"torch_fp8_gemm_{dtype}_m{dim_m}_n{dim_n}_k{dim_k}"
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "a", "dtype": dtype, "shape": [dim_m, dim_k], "role": "read"},
            {"name": "bt", "dtype": dtype, "shape": [dim_n, dim_k], "role": "read"},
            {"name": "c", "dtype": dtype, "shape": [dim_m, dim_n], "role": "write"},
        ],
        "scalar_args": [
            {"name": "dim_m", "type": "int"},
            {"name": "dim_n", "type": "int"},
            {"name": "dim_k", "type": "int"},
        ],
        "rtol": 0.25,
        "atol": 0.25,
    }
    instructions = (
        f"Optimize fp8 GEMM for {dtype}. "
        f"Shapes: a=[{dim_m},{dim_k}], bt=[{dim_n},{dim_k}], c=[{dim_m},{dim_n}]. "
        "Compute c = cast_fp8(float(a) @ float(bt).T)."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={"m": dim_m, "n": dim_n, "k": dim_k, "dtype": dtype},
        task_slug=task_slug,
        reference_plugin=PLUGIN_NAME,
        spec=spec,
        scalars={"dim_m": dim_m, "dim_n": dim_n, "dim_k": dim_k},
        dtype=dtype,
        instructions=instructions,
    )


PLUGIN = KernelEvalPlugin(name=PLUGIN_NAME, reference_factory=make_reference_plugin, contract_factory=make_operation_contract)
