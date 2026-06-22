from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kernel_evaluator.services.evaluation.types import CandidateExecutor, ExecutionInputs


@dataclass
class CallableExecutor:
    fn: Callable[[ExecutionInputs], Any]
    inputs: ExecutionInputs | None = None

    def prepare(self, inputs: ExecutionInputs) -> None:
        self.inputs = inputs

    def launch(self) -> None:
        if self.inputs is None:
            raise RuntimeError("executor launched before prepare")
        self.fn(self.inputs)

    def close(self) -> None:
        self.inputs = None


@dataclass
class PreparedFunctionExecutor:
    prepare_fn: Callable[[ExecutionInputs], Callable[[], Any]]
    launch_fn: Callable[[], Any] | None = None

    def prepare(self, inputs: ExecutionInputs) -> None:
        self.launch_fn = self.prepare_fn(inputs)

    def launch(self) -> None:
        if self.launch_fn is None:
            raise RuntimeError("executor launched before prepare")
        self.launch_fn()

    def close(self) -> None:
        self.launch_fn = None


def close_executor(executor: CandidateExecutor) -> None:
    executor.close()
