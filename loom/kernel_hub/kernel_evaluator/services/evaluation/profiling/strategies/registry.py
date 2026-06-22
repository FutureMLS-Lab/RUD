from dataclasses import dataclass, field

from kernel_evaluator.services.evaluation.profiling.strategies.base import ProfileStrategy
from kernel_evaluator.services.evaluation.profiling.strategies.cuda_kernel import CudaKernelProfileStrategy
from kernel_evaluator.services.evaluation.profiling.strategies.hip_kernel import HipKernelProfileStrategy
from kernel_evaluator.services.evaluation.profiling.strategies.python_process import PythonProcessProfileStrategy
from kernel_evaluator.services.evaluation.types import CandidateKind


@dataclass
class ProfileStrategyRegistry:
    strategies: list[ProfileStrategy] = field(default_factory=list)

    def register(self, strategy: ProfileStrategy) -> None:
        self.strategies.append(strategy)

    def require(self, kind: CandidateKind) -> ProfileStrategy:
        for strategy in self.strategies:
            if strategy.supports(kind):
                return strategy
        raise ValueError(f"no profile strategy registered for candidate kind {kind}")


def default_profile_registry() -> ProfileStrategyRegistry:
    registry = ProfileStrategyRegistry()
    registry.register(CudaKernelProfileStrategy())
    registry.register(HipKernelProfileStrategy())
    registry.register(PythonProcessProfileStrategy())
    return registry
