from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class RequestedArtifactKind(StrEnum):
    CUBIN = "cubin"
    PTX = "ptx"
    RESOURCE_USAGE = "resource_usage"
    NCU_REPORT = "ncu_report"
    NCU_SUMMARY = "ncu_summary"
    ROCPROF_REPORT = "rocprof_report"
    ROCPROF_SUMMARY = "rocprof_summary"


AVAILABLE_ARTIFACTS = {
    RequestedArtifactKind.CUBIN,
    RequestedArtifactKind.PTX,
    RequestedArtifactKind.RESOURCE_USAGE,
    RequestedArtifactKind.NCU_REPORT,
    RequestedArtifactKind.NCU_SUMMARY,
    RequestedArtifactKind.ROCPROF_REPORT,
    RequestedArtifactKind.ROCPROF_SUMMARY,
}


@dataclass(frozen=True)
class ProducedArtifact:
    kind: RequestedArtifactKind
    path: Path
    content_type: str


def parse_requested_artifacts(raw: list[str]) -> tuple[RequestedArtifactKind, ...]:
    requested = []
    for item in raw:
        kind = RequestedArtifactKind(item)
        if kind not in AVAILABLE_ARTIFACTS:
            raise ValueError(f"artifact is not supported yet: {kind}")
        if kind not in requested:
            requested.append(kind)
    return tuple(requested)


def artifact_content_type(kind: RequestedArtifactKind) -> str:
    if kind in (RequestedArtifactKind.CUBIN, RequestedArtifactKind.NCU_REPORT, RequestedArtifactKind.ROCPROF_REPORT):
        return "application/octet-stream"
    return "text/plain"
