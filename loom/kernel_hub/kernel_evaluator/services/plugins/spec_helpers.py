from collections.abc import Mapping
from typing import Any

import torch

TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8": torch.float8_e4m3fn,
}

DEFAULT_TOLERANCES = {
    "bf16": (1e-2, 1e-2),
    "fp16": (1e-3, 1e-3),
    "fp32": (1e-5, 1e-5),
}


def scalar_values(spec: Mapping[str, Any]) -> dict[str, Any]:
    scalar_args = spec["scalar_args"]
    if isinstance(scalar_args, list):
        return {value["name"]: value["value"] for value in scalar_args}
    return {name: value["value"] for name, value in scalar_args.items()}
