import torch

from kernel_evaluator.services.evaluation.types import ExecutionInputs
from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import DEFAULT_TOLERANCES, TORCH_DTYPES, scalar_values

PLUGIN_NAME = "aiter.rms_norm"


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    from aiter.ops.rmsnorm import rmsnorm as aiter_rmsnorm

    scalars = scalar_values(spec)
    dim_m = scalars["dim_m"]
    dim_n = scalars["dim_n"]
    epsilon = float(scalars.get("epsilon", 1e-6))
    tensor_dtype = TORCH_DTYPES[dtype]
    tolerance = DEFAULT_TOLERANCES[dtype]

    def make_inputs(seed: int) -> ExecutionInputs:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(dim_m, dim_n, dtype=tensor_dtype, device="cuda", generator=generator)
        weight = torch.randn(dim_n, dtype=tensor_dtype, device="cuda", generator=generator)
        out = torch.empty(dim_m, dim_n, dtype=tensor_dtype, device="cuda")
        return ExecutionInputs(
            tensors={"x": x, "weight": weight, "out": out},
            scalars={"dim_m": dim_m, "dim_n": dim_n, "epsilon": epsilon},
            output_names=("out",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        out = torch.empty_like(inputs.tensors["x"])
        aiter_rmsnorm(out, inputs.tensors["x"], inputs.tensors["weight"], epsilon)
        return {"out": out}

    def benchmark_reference(inputs: ExecutionInputs):
        def call():
            aiter_rmsnorm(inputs.tensors["out"], inputs.tensors["x"], inputs.tensors["weight"], epsilon)

        return call

    return ReferencePlugin(make_inputs, reference, tolerance, ("out",), benchmark_reference)


def make_operation_contract(shape: dict) -> OperationContract:
    dim_m = int(shape["m"])
    dim_n = int(shape["n"])
    dtype = shape["dtype"]
    epsilon = float(shape.get("epsilon", 1e-6))
    task_slug = f"aiter_rms_norm_{dtype}_m{dim_m}_n{dim_n}"
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "x", "dtype": dtype, "shape": [dim_m, dim_n], "role": "read"},
            {"name": "weight", "dtype": dtype, "shape": [dim_n], "role": "read"},
            {"name": "out", "dtype": dtype, "shape": [dim_m, dim_n], "role": "write"},
        ],
        "scalar_args": [
            {"name": "dim_m", "type": "int"},
            {"name": "dim_n", "type": "int"},
            {"name": "epsilon", "type": "float"},
        ],
        "rtol": DEFAULT_TOLERANCES[dtype][0],
        "atol": DEFAULT_TOLERANCES[dtype][1],
    }
    instructions = (
        f"Optimize RMSNorm for {dtype}. "
        f"Shapes: x=[{dim_m},{dim_n}], weight=[{dim_n}], out=[{dim_m},{dim_n}]. "
        f"Compute out[i, :] = x[i, :] / sqrt(mean(x[i, :]^2) + {epsilon}) * weight. "
        f"Each row is normalized independently; the reduction is along the "
        f"last dimension. This is a memory-bound op (2 reads + 1 write per "
        f"element); the headroom is in dispatch overhead at small M and in "
        f"vectorized loads at large M."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={"m": dim_m, "n": dim_n, "dtype": dtype, "epsilon": epsilon},
        task_slug=task_slug,
        reference_plugin=PLUGIN_NAME,
        spec=spec,
        scalars={"dim_m": dim_m, "dim_n": dim_n, "epsilon": epsilon},
        dtype=dtype,
        instructions=instructions,
    )


PLUGIN = KernelEvalPlugin(
    name=PLUGIN_NAME,
    reference_factory=make_reference_plugin,
    contract_factory=make_operation_contract,
)