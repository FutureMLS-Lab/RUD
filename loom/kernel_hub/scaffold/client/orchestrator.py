import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from scaffold.client.api import EvaluatorClient
from scaffold.client.config import KB_DOCS_SUBDIR, ScaffoldConfig
from scaffold.client.containers import ContainerRuntime, Mount
from scaffold.client.prompts import PromptContext, Prompts

SKILL_BACKEND_DIRS = set(KB_DOCS_SUBDIR)


def cli_for_model(model: str) -> str:
    if model.startswith("claude-"):
        return "claude"
    if model.startswith("gpt-") or re.match(r"^o[0-9]", model):
        return "codex"
    raise ValueError(f"could not determine CLI for model {model!r}")


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    task_slug: str
    plugin: str
    target: str
    task_slugs: tuple[str, ...] = field(default_factory=tuple)


def _run_slug(task_slugs: list[str]) -> str:
    if not task_slugs:
        return ""
    if len(task_slugs) == 1:
        return task_slugs[0]
    digest = hashlib.sha1("\n".join(task_slugs).encode()).hexdigest()[:12]
    return f"multishape-{len(task_slugs)}-{digest}"


@dataclass(frozen=True)
class AgentHandle:
    container_name: str
    agent_index: int
    model: str
    cli: str
    workdir: Path
    api_key: str
    container_id: str


class Orchestrator:
    def __init__(self, config: ScaffoldConfig, api: EvaluatorClient,
                 runtime: ContainerRuntime | None = None, image: str | None = None):
        self.config = config
        self.api = api
        preferred_runtime = os.environ.get("CONTAINER_RUNTIME") or config.run.container_runtime
        self.runtime = runtime or ContainerRuntime.detect(preferred_runtime)
        self.image = image or config.run.image
        self.prompts = Prompts(config)

    @property
    def script_dir(self) -> Path:
        return self.config.scaffold_dir / "agent_runner"

    @property
    def repo_dir(self) -> Path:
        return self.config.scaffold_dir.parent

    @property
    def runs_dir(self) -> Path:
        return self.script_dir / "runs"

    def create_run(self, plugin: str, target: str, shapes: list[dict], target_speedup: float | None = None,
                   benchmark_policy: dict | None = None) -> RunInfo:
        if benchmark_policy is None and self.config.backend_for(target) == "amd":
            benchmark_policy = {"timing_mode": "standard"}
        result = self.api.create_run(plugin, target, shapes, target_speedup, benchmark_policy)
        run_id = result["run_id"]
        task_slugs = [s["task_slug"] for s in result.get("benchmark_shapes", [])]
        return RunInfo(run_id=run_id, task_slug=_run_slug(task_slugs), plugin=plugin,
                       target=target, task_slugs=tuple(task_slugs))

    def build_image(self) -> None:
        self.runtime.build(self.image, self.script_dir / "Dockerfile", self.repo_dir)

    def _knowledge_base_mount(self, target: str) -> Mount | None:
        backend = self.config.backend_for(target)
        rel = self.config.knowledge_base.path_for(backend)
        if rel is None:
            return None
        kb_dir = self.config.scaffold_dir / rel
        if not kb_dir.is_dir():
            submodule = self.config.knowledge_base.submodule
            if submodule:
                subprocess.run(
                    ["git", "-C", str(self.config.scaffold_dir), "submodule", "update", "--init", submodule],
                    check=True,
                )
        if not kb_dir.is_dir():
            raise FileNotFoundError(f"knowledge base path {kb_dir} not found (submodule checked out?)")
        return Mount(source=str(kb_dir), target=self.config.knowledge_base.mount_point, readonly=True)

    def _instructions_filename(self, cli: str) -> str:
        return "CLAUDE.md" if cli == "claude" else "AGENTS.md"

    def _prepare_workdir(self, run: RunInfo, agent_index: int, cli: str, model: str) -> tuple[Path, str]:
        workdir = self.runs_dir / run.run_id / f"agent-{agent_index}"
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)

        instructions_file = self._instructions_filename(cli)
        ctx = PromptContext(
            task_slug=run.task_slug,
            kernel_file=self.config.kernel_file_for(run.target),
            instructions_file=instructions_file,
        )
        (workdir / instructions_file).write_text(self.prompts.build_instructions(run.target, ctx))

        self._copy_agent_skills(workdir, self.config.backend_for(run.target))

        if cli == "codex":
            (workdir / ".codex").mkdir()
            config_text = (self.script_dir / "config.toml").read_text()
            config_text = re.sub(r"(?m)^model = .*$", f'model = "{model}"', config_text)
            (workdir / ".codex" / "config.toml").write_text(config_text)
        else:
            (workdir / ".claude").mkdir()
            shutil.copy(self.script_dir / "claude_settings.json", workdir / ".claude" / "settings.json")

        return workdir, instructions_file

    def _copy_agent_skills(self, workdir: Path, backend: str) -> None:
        src_skills = self.script_dir / ".agents" / "skills"
        if not src_skills.is_dir():
            return
        dst_skills = workdir / ".agents" / "skills"
        dst_skills.mkdir(parents=True)
        for entry in sorted(src_skills.iterdir()):
            if not entry.is_dir() or entry.name in SKILL_BACKEND_DIRS:
                continue
            shutil.copytree(entry, dst_skills / entry.name)
        backend_dir = src_skills / backend
        if backend_dir.is_dir():
            for entry in sorted(backend_dir.iterdir()):
                if entry.is_dir():
                    shutil.copytree(entry, dst_skills / entry.name)

    def _agent_command(self, cli: str, model: str, kernel_file: str,
                       max_iterations: int) -> list[str]:
        if cli == "codex":
            exec_cmd = 'codex exec --sandbox danger-full-access --ephemeral --skip-git-repo-check "$CURRENT_PROMPT"'
        else:
            exec_cmd = f'claude -p "$CURRENT_PROMPT" --dangerously-skip-permissions --model {model}'
        loop = (
            f"for i in $(seq 1 {max_iterations}); do "
            f'if [ -f /workspace/{kernel_file} ]; then CURRENT_PROMPT="$AGENT_CONTINUE_PROMPT"; '
            'else CURRENT_PROMPT="$AGENT_PROMPT"; fi; '
            f'echo "[agent] starting iteration $i/{max_iterations}"; '
            f"{exec_cmd} || true; "
            f'echo "[agent] iteration $i/{max_iterations} ended"; '
            "sleep 5; "
            "done; "
            f'echo "[agent] reached max iterations ({max_iterations}), exiting"'
        )
        return ["bash", "-c", loop]

    def start_agent(self, run: RunInfo, agent_index: int, model: str,
                    starter_mode: str = "none", preset_path: str | None = None,
                    replace: bool = True, max_iterations: int | None = None) -> AgentHandle:
        cli = cli_for_model(model)
        if max_iterations is None:
            max_iterations = self.config.run.max_iterations
        if starter_mode == "preset" and preset_path is None:
            raise ValueError("starter_mode 'preset' requires preset_path")
        if preset_path is not None and not Path(preset_path).is_file():
            raise ValueError(f"preset path {preset_path!r} does not exist or is not a file")
        container_name = f"kernel-agent-{run.run_id}-{agent_index}"
        if self.runtime.exists(container_name):
            if not replace:
                raise RuntimeError(f"container {container_name} already exists")
            self.runtime.remove(container_name)

        workdir, instructions_file = self._prepare_workdir(run, agent_index, cli, model)

        ctx = PromptContext(
            task_slug=run.task_slug,
            kernel_file=self.config.kernel_file_for(run.target),
            instructions_file=instructions_file,
        )
        agent_prompt = self.prompts.prompt_for_agent(agent_index, ctx)
        continue_prompt = self.prompts.continue_prompt(ctx)

        api_key = self.api.mint_user_key()
        port = os.environ.get("KERNEL_EVALUATOR_PORT", "8000")

        env: dict[str, str] = {
            "HOME": "/workspace",
            "KERNEL_EVALUATOR_PORT": port,
            "BENCH_RUN_ID": run.run_id,
            "BENCH_AGENT_INDEX": str(agent_index),
            "KERNEL_EVALUATOR_API_KEY": api_key,
            "BENCH_STARTER_MODE": starter_mode,
            "AGENT_PROMPT": agent_prompt,
            "AGENT_CONTINUE_PROMPT": continue_prompt,
        }
        if cli == "codex":
            env["CODEX_API_KEY"] = os.environ["OPENAI_API_KEY"]
        else:
            env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
            env["IS_SANDBOX"] = "1"

        mounts = [Mount(source=str(workdir), target="/workspace")]
        kb_mount = self._knowledge_base_mount(run.target)
        if kb_mount is not None:
            mounts.append(kb_mount)
        if preset_path is not None:
            preset_name = Path(preset_path).name
            env["BENCH_PRESET_PATH"] = f"/preset/{preset_name}"
            mounts.append(Mount(source=str(Path(preset_path).resolve()),
                                target=f"/preset/{preset_name}", readonly=True))

        user = None if self.runtime.is_podman else f"{os.getuid()}:{os.getgid()}"
        command = self._agent_command(cli, model, self.config.kernel_file_for(run.target),
                                       max_iterations)
        container_id = self.runtime.run_detached(
            name=container_name, image=self.image, command=command,
            env=env, mounts=mounts, user=user,
        )
        return AgentHandle(
            container_name=container_name, agent_index=agent_index, model=model, cli=cli,
            workdir=workdir, api_key=api_key, container_id=container_id,
        )

    def launch(self, run: RunInfo, model: str, n_agents: int = 1,
               starter_mode: str = "none", preset_path: str | None = None,
               start_index: int = 1, max_iterations: int | None = None) -> list[AgentHandle]:
        return [
            self.start_agent(run, i, model, starter_mode=starter_mode, preset_path=preset_path,
                             max_iterations=max_iterations)
            for i in range(start_index, start_index + n_agents)
        ]

    def stop_agent(self, handle: AgentHandle | str) -> None:
        name = handle if isinstance(handle, str) else handle.container_name
        self.runtime.remove(name)

    def stop_run(self, run: RunInfo | str) -> list[str]:
        run_id = run if isinstance(run, str) else run.run_id
        names = self.runtime.list_by_prefix(f"kernel-agent-{run_id}-")
        for name in names:
            self.runtime.remove(name)
        return names

    def _slug_for_kernel(self, best: dict) -> str:
        scalar_args = best.get("scalar_args") or {}
        m = scalar_args.get("dim_m")
        n = scalar_args.get("dim_n")
        k = scalar_args.get("dim_k")
        if m and n and k:
            return f"f-linear-{m}x{n}x{k}"
        return (best.get("function_name") or "").replace(".", "-")

    def _postprocess_best_kernel(self, best: dict) -> None:
        kernel_id = best.get("id")
        kernel_source = best.get("kernel_source")
        if not kernel_id or not kernel_source:
            return
        slug = self._slug_for_kernel(best)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            kernel_file = tmp_dir / "kernel.cu"
            kernel_file.write_text(kernel_source)
            postprocessed = tmp_dir / "kernel_postprocessed.cu"
            tma = subprocess.run(
                ["bash", str(self.script_dir / "postprocess_tma.sh"), slug,
                 "--kernel-file", str(kernel_file), "--output-file", str(postprocessed)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if tma.returncode != 0 or not postprocessed.exists():
                shutil.copy(kernel_file, postprocessed)
            pyreg = tmp_dir / "kernel.py"
            reg = subprocess.run(
                ["bash", str(self.script_dir / "generate_python_registration.sh"), slug,
                 "--kernel-file", str(postprocessed), "--output-file", str(pyreg)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if reg.returncode != 0 or not pyreg.exists():
                pyreg.write_text("")
            self.api.patch_kernel(kernel_id, {
                "postprocessed_source": postprocessed.read_text(),
                "python_registration": pyreg.read_text(),
            })

    def finish_run(self, run: RunInfo | str, postprocess: bool = True) -> dict:
        run_id = run if isinstance(run, str) else run.run_id
        stopped = self.stop_run(run_id)

        best = self.api.best_kernel(run_id)
        postprocessed = False
        if postprocess and best and best.get("id"):
            if isinstance(run, RunInfo):
                target = run.target
            else:
                target = self.api.get_run(run_id).get("target", "cuda")
            if self.config.backend_for(target) == "cuda":
                self._postprocess_best_kernel(best)
                postprocessed = True

        run_dir = self.runs_dir / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir)

        return {"run_id": run_id, "stopped": stopped, "postprocessed": postprocessed,
                "best_kernel_id": best.get("id") if best else None}

    def watch_speedup(self, run: RunInfo | str, poll_interval: float = 60.0,
                      postprocess: bool = True) -> dict:
        run_id = run if isinstance(run, str) else run.run_id
        run_info = self.api.get_run(run_id)
        target_speedup = run_info.get("target_speedup")
        if not target_speedup:
            raise ValueError(f"run {run_id} has no target_speedup set")

        while True:
            best = self.api.best_kernel(run_id)
            if best:
                kernel_us = best.get("kernel_us")
                baseline_us = best.get("baseline_us")
                if kernel_us and baseline_us:
                    current = baseline_us / kernel_us
                    stamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{stamp}] best speedup: {current:.4f}x (target: {target_speedup}x)", flush=True)
                    if current >= target_speedup:
                        print("target speedup achieved; finishing run", flush=True)
                        return self.finish_run(run, postprocess=postprocess)
            time.sleep(poll_interval)
