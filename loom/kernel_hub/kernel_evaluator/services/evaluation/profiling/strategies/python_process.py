from kernel_evaluator.services.evaluation.types import CandidateKind


class PythonProcessProfileStrategy:
    def supports(self, kind: CandidateKind) -> bool:
        return kind in (
            CandidateKind.CUTEDSL_AOT,
            CandidateKind.TRITON_CALLABLE,
            CandidateKind.PYTHON_CALLABLE,
        )

    def update_env(self, kind: CandidateKind, env: dict[str, str]) -> dict[str, str]:
        if kind == CandidateKind.CUTEDSL_AOT:
            env["CUTE_DSL_LINEINFO"] = "1"
        return env

    def profiler_args(self) -> tuple[str, ...]:
        return ()
