from collections.abc import Callable
import importlib.util
import os
from pathlib import Path

import torch

from kernel_evaluator.services.plugins import KernelEvalPlugin, OperationContract, ReferencePlugin
from kernel_evaluator.services.plugins.spec_helpers import scalar_values
from kernel_evaluator.services.evaluation.types import ExecutionInputs

DIRECT_MAX_M = 17
PLUGIN_NAME = "cuda.int4_matmul"



def _load_tllm_harness():
    candidates = [
        "/opt/tllm_harness/tllm_load.py",
        *[
            str(parent / "int4-matmul-bench" / "tllm_load.py")
            for parent in Path(__file__).resolve().parents
        ],
    ]
    selected = next(path for path in candidates if os.path.exists(path))
    spec = importlib.util.spec_from_file_location("tllm_load", selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tllm_harness()


def _dispatch(x, qweight, scales, zeros, bits):
    import quant_matmul_cuda

    return quant_matmul_cuda.quant_matmul(x, qweight, scales, zeros, 1.0, 0.0, None, bits)



def make_reference_plugin(dtype: str, spec: dict) -> ReferencePlugin:
    scalars = scalar_values(spec)
    dim_m = scalars["dim_m"]
    dim_n = scalars["dim_n"]
    dim_k = scalars["dim_k"]
    dim_k_half = scalars["dim_k_half"]
    num_groups = scalars["num_groups"]
    group_size = scalars["group_size"]
    bits = scalars["bits"]
    gemv_direct = None

    def make_inputs(seed: int) -> ExecutionInputs:
        import quant_matmul_cuda

        generator = torch.Generator(device="cpu").manual_seed(seed)
        pack = 8 // bits
        x = torch.randn(dim_m, dim_k, dtype=torch.float16, generator=generator).to("cuda")
        qweight_raw = torch.randint(
            low=-128,
            high=127,
            size=(dim_k, dim_n // pack),
            dtype=torch.int8,
            generator=generator,
        )
        qweight = quant_matmul_cuda.preprocess_weight(qweight_raw, bits).to("cuda")
        scales = (torch.randn(num_groups, dim_n, dtype=torch.float16, generator=generator) * 0.01).to("cuda")
        zeros = (torch.randn(num_groups, dim_n, dtype=torch.float16, generator=generator) * 0.01).to("cuda")
        out = torch.empty(dim_m, dim_n, dtype=torch.float16, device="cuda")
        return ExecutionInputs(
            tensors={"x": x, "qweight": qweight, "scales": scales, "zeros": zeros, "out": out},
            scalars={
                "dim_m": dim_m,
                "dim_n": dim_n,
                "dim_k": dim_k,
                "dim_k_half": dim_k_half,
                "num_groups": num_groups,
                "group_size": group_size,
                "bits": bits,
            },
            output_names=("out",),
        )

    def reference(inputs: ExecutionInputs) -> dict[str, torch.Tensor]:
        out = _dispatch(
            inputs.tensors["x"],
            inputs.tensors["qweight"],
            inputs.tensors["scales"],
            inputs.tensors["zeros"],
            bits,
        )
        return {"out": out}

    def benchmark_reference(inputs: ExecutionInputs) -> Callable[[], None]:
        nonlocal gemv_direct
        if dim_m <= DIRECT_MAX_M:
            if gemv_direct is None:
                gemv_direct = _load_tllm_harness().gemv_direct

            def call():
                gemv_direct(
                    inputs.tensors["x"],
                    inputs.tensors["qweight"],
                    inputs.tensors["scales"],
                    inputs.tensors["zeros"],
                    group_size,
                    bits,
                )

            return call

        def call():
            _dispatch(
                inputs.tensors["x"],
                inputs.tensors["qweight"],
                inputs.tensors["scales"],
                inputs.tensors["zeros"],
                bits,
            )

        return call

    return ReferencePlugin(make_inputs, reference, (5e-2, 5e-2), ("out",), benchmark_reference)


def make_operation_contract(shape: dict) -> OperationContract:
    dim_m = int(shape["m"])
    dim_n = int(shape["n"])
    dim_k = int(shape["k"])
    group_size = int(shape["group_size"])
    bits = int(shape["bits"])
    if dim_k % group_size != 0:
        raise ValueError("k must be divisible by group_size")
    if bits != 4:
        raise ValueError("cuda.int4_matmul currently supports bits=4")
    dtype = "fp16"
    dim_k_half = dim_k // 2
    num_groups = dim_k // group_size
    task_slug = f"cuda_int4_matmul_{dtype}_m{dim_m}_n{dim_n}_k{dim_k}"
    spec = {
        "function_name": task_slug,
        "reference_plugin": PLUGIN_NAME,
        "tensor_args": [
            {"name": "x", "dtype": dtype, "role": "read"},
            {"name": "qweight", "dtype": dtype, "role": "read"},
            {"name": "scales", "dtype": dtype, "role": "read"},
            {"name": "zeros", "dtype": dtype, "role": "read"},
            {"name": "out", "dtype": dtype, "role": "write"},
        ],
        "scalar_args": [
            {"name": "dim_m", "type": "int"},
            {"name": "dim_n", "type": "int"},
            {"name": "dim_k", "type": "int"},
            {"name": "dim_k_half", "type": "int"},
            {"name": "num_groups", "type": "int"},
            {"name": "group_size", "type": "int"},
            {"name": "bits", "type": "int"},
        ],
        "rtol": 0.1,
        "atol": 0.1,
    }
    scalars = {
        "dim_m": dim_m,
        "dim_n": dim_n,
        "dim_k": dim_k,
        "dim_k_half": dim_k_half,
        "num_groups": num_groups,
        "group_size": group_size,
        "bits": bits,
    }
    instructions = (
        "Optimize cuda.int4_matmul for fp16 activations and packed int4 weights. "
        f"Shapes: x=[{dim_m},{dim_k}], qweight is evaluator-packed, "
        f"scales=zeros=[{num_groups},{dim_n}], out=[{dim_m},{dim_n}]. "
        "Compute the same output as quant_matmul_cuda.quant_matmul(x, qweight, scales, zeros, 1.0, 0.0, None, bits)."
    )
    return OperationContract(
        plugin=PLUGIN_NAME,
        shape={"m": dim_m, "n": dim_n, "k": dim_k, "group_size": group_size, "bits": bits, "dtype": dtype},
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
