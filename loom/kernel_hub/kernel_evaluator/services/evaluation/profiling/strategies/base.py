from typing import Protocol

from kernel_evaluator.services.evaluation.types import CandidateKind


class ProfileStrategy(Protocol):
    def supports(self, kind: CandidateKind) -> bool:
        ...

    def update_env(self, kind: CandidateKind, env: dict[str, str]) -> dict[str, str]:
        ...

    def profiler_args(self) -> tuple[str, ...]:
        ...
