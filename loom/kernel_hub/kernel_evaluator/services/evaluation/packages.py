from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_evaluator.services.evaluation.artifacts import materialize_artifact, require_inside
from kernel_evaluator.services.evaluation.runtime import RuntimePolicy, validate_runtime_policy
from kernel_evaluator.services.evaluation.types import Artifact, ArtifactKind, CandidateKind, CandidateSubmission, KernelABI


@dataclass(frozen=True)
class CandidatePackage:
    kind: CandidateKind
    artifacts: tuple[Artifact, ...]
    entrypoint: str | None
    runtime: RuntimePolicy


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, owner: str) -> str:
    if key not in mapping:
        raise ValueError(f"{owner} missing required field '{key}'")
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{owner}.{key} must be a non-empty string")
    return value


def _optional_str(mapping: Mapping[str, Any], key: str, owner: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{owner}.{key} must be a non-empty string")
    return value


def _candidate_kind(value: str) -> CandidateKind:
    try:
        return CandidateKind(value)
    except ValueError as exc:
        raise ValueError(f"unknown candidate kind: {value}") from exc


def _artifact_kind(value: str) -> ArtifactKind:
    try:
        return ArtifactKind(value)
    except ValueError as exc:
        raise ValueError(f"unknown artifact kind: {value}") from exc


def _runtime_policy(manifest: Mapping[str, Any]) -> RuntimePolicy:
    if "runtime" not in manifest:
        raise ValueError("candidate package missing runtime")
    runtime = _require_mapping(manifest["runtime"], "runtime")
    env = _require_str(runtime, "env", "runtime")
    if "build_profile" in runtime:
        build_profile = _require_str(runtime, "build_profile", "runtime")
    elif "build_profile" in manifest:
        build_profile = _require_str(manifest, "build_profile", "candidate package")
    else:
        raise ValueError("candidate package missing build_profile")
    if "import_roots" in runtime:
        raw_roots = runtime["import_roots"]
        if not isinstance(raw_roots, list):
            raise ValueError("runtime.import_roots must be a list")
        import_roots = tuple(str(root) for root in raw_roots)
    else:
        import_roots = ()
    return validate_runtime_policy(RuntimePolicy(env, build_profile, import_roots))


def _artifacts(root: Path, manifest: Mapping[str, Any]) -> tuple[Artifact, ...]:
    if "artifacts" not in manifest:
        raise ValueError("candidate package missing artifacts")
    raw_artifacts = manifest["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ValueError("artifacts must be a list")
    artifacts = []
    for index, item in enumerate(raw_artifacts):
        owner = f"artifacts[{index}]"
        artifact = _require_mapping(item, owner)
        kind = _artifact_kind(_require_str(artifact, "kind", owner))
        rel_path = Path(_require_str(artifact, "path", owner))
        path = require_inside(root, root / rel_path)
        if not path.exists():
            raise ValueError(f"artifact path does not exist: {rel_path}")
        artifacts.append(materialize_artifact(kind, path))
    return tuple(artifacts)


def parse_candidate_package(root: Path, manifest: Mapping[str, Any]) -> CandidatePackage:
    kind = _candidate_kind(_require_str(manifest, "kind", "candidate package"))
    runtime = _runtime_policy(manifest)
    return CandidatePackage(
        kind=kind,
        artifacts=_artifacts(root, manifest),
        entrypoint=_optional_str(manifest, "entrypoint", "candidate package"),
        runtime=runtime,
    )


def submission_from_package(
    package: CandidatePackage,
    abi: KernelABI,
    scalars: dict,
) -> CandidateSubmission:
    metadata = {
        "scalars": scalars,
        "runtime_env": package.runtime.env,
        "build_profile": package.runtime.build_profile,
        "import_roots": package.runtime.import_roots,
    }
    source_artifacts = tuple(artifact for artifact in package.artifacts if artifact.kind == ArtifactKind.SOURCE)
    source = str(source_artifacts[0].path) if len(source_artifacts) == 1 and package.kind in (CandidateKind.CUDA_SOURCE, CandidateKind.HIP_SOURCE) else None
    return CandidateSubmission(
        kind=package.kind,
        abi=abi,
        artifacts=package.artifacts,
        source=source,
        entrypoint=package.entrypoint,
        metadata=metadata,
    )
