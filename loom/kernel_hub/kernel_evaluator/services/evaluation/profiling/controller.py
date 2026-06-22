import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from kernel_evaluator.services.evaluation.profiling.strategies.registry import ProfileStrategyRegistry
from kernel_evaluator.services.evaluation.profiling.types import ProfilePolicy, ProfilerCli, ProfileShapeResult
from kernel_evaluator.services.evaluation.types import CandidateKind


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _with_pythonpath(env: dict[str, str], *paths: Path) -> dict[str, str]:
    existing = env["PYTHONPATH"] if "PYTHONPATH" in env else ""
    entries = [str(path) for path in paths]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


@dataclass
class ProfileController:
    registry: ProfileStrategyRegistry
    profiler: ProfilerCli
    policy: ProfilePolicy

    def profile_shape(
        self,
        payload_path: Path,
        payload: dict,
        gpu_id: int,
        shape_index: int,
        output_dir: Path,
    ) -> ProfileShapeResult:
        kind = CandidateKind(payload["candidate_kind"])
        strategy = self.registry.require(kind)
        shape_contract = payload["benchmark_shapes"][shape_index]
        task_slug = shape_contract["task_slug"]
        env = strategy.update_env(kind, os.environ.copy())
        env = _with_pythonpath(env, _repo_root(), Path(payload["artifact_dir"]))
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        report_base = output_dir / f"shape_{shape_index:03d}_{_safe_slug(task_slug)}"
        runner_cmd = [
            sys.executable,
            "-m",
            "kernel_evaluator.services.evaluation.profiling.runner",
            "--job-json",
            str(payload_path),
            "--shape-index",
            str(shape_index),
            "--gpu-id",
            "0",
            "--launch-count",
            str(self.policy.launch_count),
            "--warmup-launches",
            str(self.policy.warmup_launches),
        ]
        report_path, summary_text = self.profiler.profile(
            runner_cmd,
            report_base,
            env,
            output_dir,
            self.policy,
            strategy.profiler_args(),
        )
        return ProfileShapeResult(
            shape_index=shape_index,
            task_slug=task_slug,
            report_path=report_path,
            summary_text=summary_text,
        )
