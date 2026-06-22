import argparse
import json
import os
from pathlib import Path

import torch

from kernel_evaluator.services.compile_subprocess import PACKAGE_MANIFEST
from kernel_evaluator.services.evaluation import BuildContext, default_registry
from kernel_evaluator.services.evaluation.packages import parse_candidate_package, submission_from_package
from kernel_evaluator.services.evaluation.task_factory import reference_task_from_spec
from kernel_evaluator.services.evaluation.types import CandidateKind, ExecutionInputs


def _clear_outputs(inputs: ExecutionInputs) -> None:
    for name in inputs.output_names:
        inputs.tensors[name].zero_()


def _reference_task_from_shape(shape_contract: dict):
    return reference_task_from_spec(
        shape_contract["spec"],
        shape_contract["scalars"],
        shape_contract["dtype"],
        shape_contract["task_slug"],
    )


def _submission_from_package(payload: dict, task, shape_contract: dict):
    artifact_dir = Path(payload["artifact_dir"])
    manifest = json.loads((artifact_dir / PACKAGE_MANIFEST).read_text(encoding="utf-8"))
    package = parse_candidate_package(artifact_dir, manifest)
    return submission_from_package(package, task.abi, shape_contract["scalars"])


def run(payload: dict, shape_index: int, gpu_id: int, launch_count: int, warmup_launches: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    kind = CandidateKind(payload["candidate_kind"])
    if kind == CandidateKind.CUTEDSL_AOT:
        os.environ["CUTE_DSL_LINEINFO"] = "1"
    shape_contract = payload["benchmark_shapes"][shape_index]
    task = _reference_task_from_shape(shape_contract)
    submission = _submission_from_package(payload, task, shape_contract)
    context = BuildContext(work_dir=Path(payload["artifact_dir"]), cuda_arch=payload["cuda_arch"])
    candidate = default_registry().require(submission.kind).build(submission, context)
    executor = candidate.executor_factory()
    try:
        inputs = task.make_inputs(0)
        executor.prepare(inputs)
        _clear_outputs(inputs)
        for _ in range(warmup_launches):
            executor.launch()
        torch.cuda.synchronize()
        gated = kind != CandidateKind.HIP_SOURCE
        if gated:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            for _ in range(launch_count):
                executor.launch()
            torch.cuda.synchronize()
        finally:
            if gated:
                torch.cuda.cudart().cudaProfilerStop()
    finally:
        executor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--shape-index", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--launch-count", type=int, required=True)
    parser.add_argument("--warmup-launches", type=int, required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    run(payload, args.shape_index, args.gpu_id, args.launch_count, args.warmup_launches)


if __name__ == "__main__":
    main()
