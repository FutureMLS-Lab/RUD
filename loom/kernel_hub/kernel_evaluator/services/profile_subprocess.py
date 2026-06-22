import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from kernel_evaluator.services.artifact_requests import RequestedArtifactKind
from kernel_evaluator.services.evaluation.profiling import (
    NcuCli,
    ProfileController,
    ProfilePolicy,
    ProfilerCli,
    RocprofCli,
)
from kernel_evaluator.services.evaluation.profiling.strategies import default_profile_registry
from kernel_evaluator.services.evaluation.types import CandidateKind


@dataclass(frozen=True)
class _Backend:
    name: str
    profiler: ProfilerCli
    summary_kind: RequestedArtifactKind
    report_kind: RequestedArtifactKind


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _render_summary(shape_results) -> str:
    parts = []
    for shape_result in shape_results:
        parts.append(f"shape_index: {shape_result.shape_index}\n")
        parts.append(f"task_slug: {shape_result.task_slug}\n")
        parts.append(shape_result.summary_text)
        parts.append("\n")
    return "".join(parts)


def _write_archive(path: Path, shape_results) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for shape_result in shape_results:
            archive.add(shape_result.report_path, arcname=shape_result.report_path.name)


def _backend(kind: CandidateKind, timeout_s: float) -> _Backend:
    if kind == CandidateKind.HIP_SOURCE:
        return _Backend(
            "rocprof",
            RocprofCli(timeout_s),
            RequestedArtifactKind.ROCPROF_SUMMARY,
            RequestedArtifactKind.ROCPROF_REPORT,
        )
    return _Backend(
        "ncu",
        NcuCli(timeout_s),
        RequestedArtifactKind.NCU_SUMMARY,
        RequestedArtifactKind.NCU_REPORT,
    )


def run(payload: dict, gpu_id: int, timeout_s: float) -> dict:
    artifact_dir = Path(payload["artifact_dir"])
    backend = _backend(CandidateKind(payload["candidate_kind"]), timeout_s)
    output_dir = artifact_dir / backend.name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "profile_job.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    controller = ProfileController(
        registry=default_profile_registry(),
        profiler=backend.profiler,
        policy=ProfilePolicy(),
    )
    shape_results = tuple(
        controller.profile_shape(payload_path, payload, gpu_id, shape_index, output_dir)
        for shape_index in range(len(payload["benchmark_shapes"]))
    )
    summary_path = output_dir / f"{backend.name}_summary.txt"
    summary_path.write_text(_render_summary(shape_results), encoding="utf-8")
    report_path = output_dir / f"{backend.name}_report.tar.gz"
    _write_archive(report_path, shape_results)
    return {
        "artifacts": [
            {"kind": str(backend.summary_kind), "path": _relative(artifact_dir, summary_path)},
            {"kind": str(backend.report_kind), "path": _relative(artifact_dir, report_path)},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        result = {"ok": True, "result": run(payload, args.gpu_id, args.timeout_s)}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
