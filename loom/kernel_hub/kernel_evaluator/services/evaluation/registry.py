from dataclasses import dataclass, field

from kernel_evaluator.services.evaluation.types import CandidateBuilder, CandidateKind


@dataclass
class BuilderRegistry:
    builders: dict[CandidateKind, CandidateBuilder] = field(default_factory=dict)

    def register(self, builder: CandidateBuilder) -> None:
        self.builders[builder.kind] = builder

    def require(self, kind: CandidateKind) -> CandidateBuilder:
        try:
            return self.builders[kind]
        except KeyError as exc:
            raise ValueError(f"no builder registered for candidate kind {kind}") from exc


def default_registry() -> BuilderRegistry:
    from kernel_evaluator.services.evaluation.builders import CudaSourceBuilder, CuTeDSLBuilder, ExternalPtxSoBuilder, HipSourceBuilder, PythonCallableBuilder, TritonCallableBuilder

    registry = BuilderRegistry()
    registry.register(CudaSourceBuilder())
    registry.register(HipSourceBuilder())
    registry.register(ExternalPtxSoBuilder())
    registry.register(CuTeDSLBuilder())
    registry.register(PythonCallableBuilder())
    registry.register(TritonCallableBuilder())
    return registry
