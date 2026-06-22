from kernel_evaluator.services.evaluation.types import CandidateKind


class HipKernelProfileStrategy:
    def supports(self, kind: CandidateKind) -> bool:
        return kind == CandidateKind.HIP_SOURCE

    def update_env(self, kind: CandidateKind, env: dict[str, str]) -> dict[str, str]:
        del kind
        return env

    def profiler_args(self) -> tuple[str, ...]:
        return ()
