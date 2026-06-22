from pathlib import Path

from kernel_evaluator.services.evaluation.specs import inject_scalar_values, parse_kernel_abi, parse_spec
from kernel_evaluator.services.evaluation.types import (
    Artifact,
    ArtifactKind,
    CandidateKind,
    CandidateSubmission,
    KernelABI,
    ReferenceTask,
)
from kernel_evaluator.services.plugins import make_reference_plugin


def spec_to_abi(spec: dict) -> KernelABI:
    return parse_kernel_abi(spec)


def _inject_values(spec: dict, scalars: dict) -> dict:
    return inject_scalar_values(spec, scalars)


def reference_task_from_spec(spec: dict, scalars: dict, dtype: str, slug: str) -> ReferenceTask:
    normalized = parse_spec(spec)
    spec_with_values = _inject_values(spec, scalars)
    plugin = make_reference_plugin(normalized.reference_plugin, dtype, spec_with_values)
    tolerances = normalized.tolerances if normalized.tolerances is not None else plugin.tolerances
    return ReferenceTask(
        slug=slug,
        abi=normalized.abi,
        make_inputs=plugin.make_inputs,
        reference=plugin.reference,
        tolerances=tolerances,
        benchmark_reference=plugin.benchmark_reference,
    )


def cubin_so_submission(so_path: Path, cubin_path: Path, spec: dict, scalars: dict) -> CandidateSubmission:
    abi = spec_to_abi(spec)
    return CandidateSubmission(
        kind=CandidateKind.EXTERNAL_PTX_SO,
        abi=abi,
        artifacts=(
            Artifact(ArtifactKind.SHARED_OBJECT, so_path),
            Artifact(ArtifactKind.CUBIN, cubin_path),
        ),
        metadata={"scalars": scalars},
    )


def so_submission(so_path: Path, spec: dict, scalars: dict) -> CandidateSubmission:
    abi = spec_to_abi(spec)
    return CandidateSubmission(
        kind=CandidateKind.EXTERNAL_PTX_SO,
        abi=abi,
        artifacts=(Artifact(ArtifactKind.SHARED_OBJECT, so_path),),
        metadata={"scalars": scalars},
    )
