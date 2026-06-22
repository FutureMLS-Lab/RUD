import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kernel_evaluator.services.evaluation.types import Artifact, ArtifactKind


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes package root: {path}") from exc
    return resolved_path


def artifact_by_kind(artifacts: tuple[Artifact, ...], kind: ArtifactKind) -> Artifact:
    matches = [artifact for artifact in artifacts if artifact.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} artifact, found {len(matches)}")
    return matches[0]


def artifacts_by_kind(artifacts: tuple[Artifact, ...], kind: ArtifactKind) -> tuple[Artifact, ...]:
    return tuple(artifact for artifact in artifacts if artifact.kind == kind)


def materialize_artifact(kind: ArtifactKind, path: Path) -> Artifact:
    return Artifact(kind, path, sha256_file(path))


def normalize_artifacts(artifacts: tuple[Artifact, ...]) -> tuple[Artifact, ...]:
    normalized = []
    for artifact in artifacts:
        checksum = artifact.checksum
        if checksum is None and artifact.path.exists():
            checksum = sha256_file(artifact.path)
        normalized.append(Artifact(artifact.kind, artifact.path, checksum))
    return tuple(normalized)


def write_json(path: Path, data: Mapping[str, Any]) -> Artifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return materialize_artifact(ArtifactKind.MANIFEST, path)


def write_log(path: Path, text: str) -> Artifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return materialize_artifact(ArtifactKind.LOG, path)
