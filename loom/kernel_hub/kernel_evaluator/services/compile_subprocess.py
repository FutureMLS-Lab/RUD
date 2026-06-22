import argparse
import json
from pathlib import Path

from kernel_evaluator.services.evaluation import (
    Artifact,
    ArtifactKind,
    BenchmarkController,
    BenchmarkPolicy,
    BuildContext,
    CandidateKind,
    CandidateSubmission,
    default_registry,
)
from kernel_evaluator.services.evaluation.runtime import DEFAULT_RUNTIME_ENV
from kernel_evaluator.services.evaluation.task_factory import reference_task_from_spec

PACKAGE_MANIFEST = "candidate_package.json"


def _source(payload: dict) -> Path:
    artifact_dir = Path(payload["artifact_dir"])
    kind = CandidateKind(payload["candidate_kind"])
    if kind == CandidateKind.CUDA_SOURCE:
        suffix = ".cu"
    elif kind == CandidateKind.HIP_SOURCE:
        suffix = ".cpp"
    elif kind in (CandidateKind.CUTEDSL_AOT, CandidateKind.TRITON_CALLABLE, CandidateKind.PYTHON_CALLABLE):
        suffix = ".py"
    else:
        raise ValueError(f"unsupported candidate kind: {kind}")
    return artifact_dir / f"source{suffix}"


def _relative_artifact(root: Path, artifact: Artifact) -> dict:
    return {
        "kind": str(artifact.kind),
        "path": str(artifact.path.resolve().relative_to(root.resolve())),
    }


def _runtime_for_kind(kind: CandidateKind) -> dict:
    if kind == CandidateKind.HIP_SOURCE:
        return {"env": DEFAULT_RUNTIME_ENV, "build_profile": "hip_pybind", "import_roots": []}
    if kind == CandidateKind.CUTEDSL_AOT:
        return {"env": DEFAULT_RUNTIME_ENV, "build_profile": "cutedsl_entrypoint", "import_roots": []}
    if kind == CandidateKind.TRITON_CALLABLE:
        return {"env": DEFAULT_RUNTIME_ENV, "build_profile": "triton_entrypoint", "import_roots": []}
    if kind == CandidateKind.PYTHON_CALLABLE:
        return {"env": DEFAULT_RUNTIME_ENV, "build_profile": "python_entrypoint", "import_roots": []}
    if kind == CandidateKind.EXTERNAL_PTX_SO:
        return {"env": DEFAULT_RUNTIME_ENV, "build_profile": "prebuilt", "import_roots": []}
    raise ValueError(f"unsupported candidate kind: {kind}")


def _write_package(root: Path, manifest: dict) -> None:
    (root / PACKAGE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run(payload: dict) -> dict:
    artifact_dir = Path(payload["artifact_dir"])
    first_shape = payload["benchmark_shapes"][0]
    task = reference_task_from_spec(
        first_shape["spec"],
        first_shape["scalars"],
        first_shape["dtype"],
        first_shape["task_slug"],
    )
    source = _source(payload)
    kind = CandidateKind(payload["candidate_kind"])
    if kind not in (CandidateKind.CUDA_SOURCE, CandidateKind.HIP_SOURCE):
        _write_package(
            artifact_dir,
            {
                "kind": str(kind),
                "entrypoint": payload["entrypoint"],
                "runtime": _runtime_for_kind(kind),
                "artifacts": [
                    {
                        "kind": str(ArtifactKind.SOURCE),
                        "path": source.name,
                    }
                ],
            },
        )
        return {"package_manifest": str(artifact_dir / PACKAGE_MANIFEST)}
    if kind == CandidateKind.HIP_SOURCE:
        submission = CandidateSubmission(
            kind=kind,
            abi=task.abi,
            artifacts=(Artifact(ArtifactKind.SOURCE, source),),
            source=str(source),
            entrypoint=payload["entrypoint"],
            metadata={"scalars": first_shape["scalars"]},
        )
        context = BuildContext(work_dir=artifact_dir, cuda_arch=payload["cuda_arch"])
        candidate = BenchmarkController(registry=default_registry(), policy=BenchmarkPolicy()).build(submission, context)
        package_artifacts = tuple(
            artifact
            for artifact in candidate.artifacts
            if artifact.kind in (
                ArtifactKind.SOURCE,
                ArtifactKind.SHARED_OBJECT,
                ArtifactKind.LOG,
                ArtifactKind.MANIFEST,
            )
        )
        _write_package(
            artifact_dir,
            {
                "kind": str(kind),
                "entrypoint": payload["entrypoint"],
                "runtime": _runtime_for_kind(kind),
                "artifacts": [_relative_artifact(artifact_dir, artifact) for artifact in package_artifacts],
            },
        )
        return {"package_manifest": str(artifact_dir / PACKAGE_MANIFEST)}
    submission = CandidateSubmission(
        kind=kind,
        abi=task.abi,
        artifacts=(Artifact(ArtifactKind.SOURCE, source),),
        source=str(source),
        entrypoint=payload["entrypoint"],
        metadata={"scalars": first_shape["scalars"]},
    )
    context = BuildContext(work_dir=artifact_dir, cuda_arch=payload["cuda_arch"])
    candidate = BenchmarkController(registry=default_registry(), policy=BenchmarkPolicy()).build(submission, context)
    package_artifacts = tuple(
        artifact
        for artifact in candidate.artifacts
        if artifact.kind in (
            ArtifactKind.SOURCE,
            ArtifactKind.SHARED_OBJECT,
            ArtifactKind.CUBIN,
            ArtifactKind.PTX,
            ArtifactKind.LOG,
            ArtifactKind.RESOURCE_USAGE,
        )
    )
    _write_package(
        artifact_dir,
        {
            "kind": str(CandidateKind.EXTERNAL_PTX_SO),
            "entrypoint": None,
            "runtime": _runtime_for_kind(CandidateKind.EXTERNAL_PTX_SO),
            "artifacts": [_relative_artifact(artifact_dir, artifact) for artifact in package_artifacts],
        },
    )
    return {"package_manifest": str(artifact_dir / PACKAGE_MANIFEST)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        result = {"ok": True, "result": run(payload)}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
