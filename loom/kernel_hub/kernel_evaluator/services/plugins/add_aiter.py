import torch

from kernel_evaluator.services.evaluation.types import ExecutionInputs
from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import DEFAULT_TOLERANCES, TORCH_DTYPES, scalar_values

PLUGIN_NAME = "aiter.add_rms_norm"


def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    from aiter.ops.rmsnorm import add_rmsnorm as aiter_add_rmsnorm

    scalars = scalar_values(spec)
    dim_m = scalars["dim_m"]
    dim_n = scalars["dim_n"]
    epsilon = float(scalars.get("epsilon", 1e-6))
    tensor_dtype = TORCH_DTYPES[dtype]
    tolerance = DEFAULT_TOLERANCES[dtype]

    def make_inputs(seed: int) -> ExecutionInputs:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(dim_m, dim_n, dtype=tensor_dtype, device="cuda", generator=generator)
        residual_in = torch.randn(dim_m, dim_n, dtype=tensor_dtype, device="cuda", generator=generator)
        weight = torch.randn(dim_n, dtype=tensor_dtype, device="cuda", generator=generator)
        out = torch.empty(dim_m, dim_n, dtype=tensor_dtype, device="cuda")
        residual_out = torch.empty(dim_m, dim_n, dtype=tensor_dtype, device="cuda")
        return ExecutionInputs(
            tensors={
                "x": x,
                "residual_in": residual_in,
                "weight": weight,
                "out": out,
                "residual_out": residual_out,
            },
            scalars={"dim_m": dim_m, "dim_n": dim_n, "epsilon": epsilon},
            output_names=("out", "residual_out"),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        out = torch.empty_like(inputs.tensors["x"])
        residual_out = torch.empty_like(inputs.tensors["x"])
        aiter_add_rmsnorm(
            out,
            inputs.tensors["x"],
            inputs.tensors["residual_in"],
            residual_out,
            inputs.tensors["weight"],
            epsilon,
        )
        return {"out": out, "residual_out": residual_out}

    def benchmark_reference(inputs: ExecutionInputs):
        def call():
            aiter_add_rmsnorm(
                inputs.tensors["out"],
                inputs.tensors["x"],
                inputs.tensors["residual_in"],
                inputs.tensors["residual_out"],
                inputs.tensors["weight"],
                epsilon,
            )

        return call

    return ReferencePlugin(
        make_inputs, reference, tolerance, ("out", "residual_out"), benchmark_reference
    )


def make_operation_contract(shape: dict) -> OperationContract:
    dim_m = int(shape["m"])
    dim_n = int(shape["n"])
    dtype = shape["dtype"]
    epsilon = float(shape.get("epsilon", 1e-6))
    task_slug = f"aiter_add_rms_norm_{dtype}_m{dim_m}_n{dim_n}"
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "x", "dtype": dtype, "shape": [dim_m, dim_n], "role": "read"},
            {"name": "residual_in", "dtype": dtype, "shape": [dim_m, dim_n], "role": "read"},
            {"name": "weight", "dtype": dtype, "shape": [dim_n], "role": "read"},
            {"name": "out", "dtype": dtype, "shape": [dim_m, dim_n], "role": "write"},
            {"name": "residual_out", "dtype": dtype, "shape": [dim_m, dim_n], "role": "write"},
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
        f"Optimize fused residual-add + RMSNorm for {dtype}. "
        f"Shapes: x=[{dim_m},{dim_n}], residual_in=[{dim_m},{dim_n}], "
        f"weight=[{dim_n}], out=[{dim_m},{dim_n}], residual_out=[{dim_m},{dim_n}]. "
        f"Compute residual_out[i, :] = x[i, :] + residual_in[i, :], then "
        f"out[i, :] = residual_out[i, :] / sqrt(mean(residual_out[i, :]^2) + {epsilon}) * weight. "
        f"Each row is normalized independently along the last dimension. "
        f"Memory-bound (3 reads + 2 writes per element); the fusion saves "
        f"an HBM round-trip vs separate add + rmsnorm. Headroom is in "
        f"dispatch overhead at small M and vectorized loads at large M."
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
        use_custom_reference_timing=True,
    )


PLUGIN = KernelEvalPlugin(
    name=PLUGIN_NAME,
    reference_factory=make_reference_plugin,
    contract_factory=make_operation_contract,
)