import argparse
import json
from pathlib import Path

import torch

from kernel_evaluator.services.evaluation import (
    BenchmarkController,
    BenchmarkPolicy,
    BuildContext,
    TimingMode,
    default_registry,
)
from kernel_evaluator.services.evaluation.types import BenchmarkResult, ShapeBenchmarkResult
from kernel_evaluator.services.evaluation.packages import parse_candidate_package, submission_from_package
from kernel_evaluator.services.evaluation.task_factory import reference_task_from_spec

PACKAGE_MANIFEST = "candidate_package.json"


def _benchmark_policy(payload: dict) -> BenchmarkPolicy:
    policy = payload["benchmark_policy"]
    return BenchmarkPolicy(
        timing_mode=TimingMode(policy["timing_mode"]),
        warmup=int(policy["warmup"]),
        iterations=int(policy["iterations"]),
        graph_calls=int(policy["graph_calls"]),
        repeats=int(policy["repeats"]),
        sleep_s=float(policy.get("sleep_s", 10.0)),
        clear_outputs_after_prepare=bool(policy["clear_outputs_after_prepare"]),
    )


def _result_payload(result) -> dict:
    return {
        "correct": result.correct,
        "baseline_us": result.baseline_us,
        "candidate_us": result.candidate_us,
        "speedup": result.speedup,
        "repeats": [
            {
                "seed": repeat.seed,
                "baseline_us": repeat.baseline_us,
                "candidate_us": repeat.candidate_us,
                "correct": repeat.correct,
            }
            for repeat in result.repeats
        ],
        "shape_results": [
            {
                "shape": dict(shape_result.shape),
                "task_slug": shape_result.task_slug,
                "correct": shape_result.correct,
                "baseline_us": shape_result.baseline_us,
                "candidate_us": shape_result.candidate_us,
                "speedup": shape_result.speedup,
                "repeats": [
                    {
                        "seed": repeat.seed,
                        "baseline_us": repeat.baseline_us,
                        "candidate_us": repeat.candidate_us,
                        "correct": repeat.correct,
                    }
                    for repeat in shape_result.repeats
                ],
            }
            for shape_result in result.shape_results
        ],
    }


def _reference_task_from_shape(shape_contract: dict):
    task = reference_task_from_spec(
        shape_contract["spec"],
        shape_contract["scalars"],
        shape_contract["dtype"],
        shape_contract["task_slug"],
    )
    return task


def _submission_from_compiled_package(payload: dict, task, shape_contract: dict):
    artifact_dir = Path(payload["artifact_dir"])
    manifest_path = artifact_dir / PACKAGE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = parse_candidate_package(artifact_dir, manifest)
    return submission_from_package(package, task.abi, shape_contract["scalars"])


def run(payload: dict, gpu_id: int) -> dict:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    first_shape = payload["benchmark_shapes"][0]
    first_task = _reference_task_from_shape(first_shape)
    submission = _submission_from_compiled_package(payload, first_task, first_shape)
    context = BuildContext(work_dir=Path(payload["artifact_dir"]), cuda_arch=payload["cuda_arch"])
    controller = BenchmarkController(registry=default_registry(), policy=_benchmark_policy(payload))
    candidate = controller.build(submission, context)
    shape_results = []
    for shape_contract in payload["benchmark_shapes"]:
        task = _reference_task_from_shape(shape_contract)
        result = controller.benchmark(task, candidate)
        shape_results.append(
            ShapeBenchmarkResult(
                shape=shape_contract["shape"],
                task_slug=shape_contract["task_slug"],
                repeats=result.repeats,
            )
        )
    return _result_payload(BenchmarkResult(shape_results=tuple(shape_results)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        result = {"ok": True, "result": run(payload, args.gpu_id)}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
