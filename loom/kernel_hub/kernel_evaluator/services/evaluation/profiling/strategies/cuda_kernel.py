from kernel_evaluator.services.evaluation.types import CandidateKind


class CudaKernelProfileStrategy:
    def supports(self, kind: CandidateKind) -> bool:
        return kind in (CandidateKind.CUDA_SOURCE, CandidateKind.EXTERNAL_PTX_SO)

    def update_env(self, kind: CandidateKind, env: dict[str, str]) -> dict[str, str]:
        del kind
        return env

    def profiler_args(self) -> tuple[str, ...]:
        return ()
