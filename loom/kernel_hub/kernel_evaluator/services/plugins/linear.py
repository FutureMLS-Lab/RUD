import torch
import torch.nn.functional as F

from kernel_evaluator.services.evaluation.types import ExecutionInputs
from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import DEFAULT_TOLERANCES, TORCH_DTYPES, scalar_values

PLUGIN_NAME = "torch.linear"


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    dim_m = scalars["dim_m"]
    dim_n = scalars["dim_n"]
    dim_k = scalars["dim_k"]
    tensor_dtype = TORCH_DTYPES[dtype]
    tolerance = DEFAULT_TOLERANCES[dtype]

    def make_inputs(seed: int) -> ExecutionInputs:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.randn(dim_m, dim_k, dtype=tensor_dtype, device="cuda", generator=generator)
        bt = torch.randn(dim_n, dim_k, dtype=tensor_dtype, device="cuda", generator=generator)
        c = torch.empty(dim_m, dim_n, dtype=tensor_dtype, device="cuda")
        return ExecutionInputs(
            tensors={"a": a, "bt": bt, "c": c},
            scalars={"dim_m": dim_m, "dim_n": dim_n, "dim_k": dim_k},
            output_names=("c",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        return {"c": F.linear(inputs.tensors["a"], inputs.tensors["bt"])}

    return ReferencePlugin(make_inputs, reference, tolerance, ("c",))


def make_operation_contract(shape: dict) -> OperationContract:
    dim_m = int(shape["m"])
    dim_n = int(shape["n"])
    dim_k = int(shape["k"])
    dtype = shape["dtype"]
    task_slug = f"torch_linear_{dtype}_m{dim_m}_n{dim_n}_k{dim_k}"
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
        "rtol": DEFAULT_TOLERANCES[dtype][0],
        "atol": DEFAULT_TOLERANCES[dtype][1],
    }
    instructions = (
        f"Optimize torch.linear for {dtype}. "
        f"Shapes: a=[{dim_m},{dim_k}], bt=[{dim_n},{dim_k}], c=[{dim_m},{dim_n}]. "
        "Compute c = a @ bt.T."
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


PLUGIN = KernelEvalPlugin(
    name=PLUGIN_NAME,
    reference_factory=make_reference_plugin,
    contract_factory=make_operation_contract,
)
