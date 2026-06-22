import textwrap
from pathlib import Path

import pytest

from scaffold.client import admin_cli
from scaffold.client.config import ScaffoldConfig
from scaffold.client.containers import Mount
from scaffold.client.orchestrator import AgentHandle, Orchestrator, RunInfo, cli_for_model
from scaffold.client.prompts import PromptContext, Prompts

SCAFFOLD_YAML = """
knowledge_base:
  enabled: true
  mount_point: /kb
  amd:
    path: knowledge_base/kb_amd
  cuda:
    path: knowledge_base/kb_cuda
multiagent:
  roles_file: multiagent/roles.yaml
  role_set: gemm_specialists
run:
  plugin: aiter.rms_norm
  target: hip
  shapes:
    - {"m": 16, "n": 32, "dtype": "bf16"}
  model: gpt-5.5
  n_agents: 3
  container_runtime: podman
"""

ROLES_YAML = """
default_role: "Default $TASK_SLUG $KERNEL_FILE $INSTRUCTIONS_FILE"
continue_role: "Continue $TASK_SLUG"
gemm_specialists:
  - prompt: "Spec1 $TASK_SLUG $KERNEL_FILE"
  - prompt: "Spec2 $TASK_SLUG"
"""


REAL_SCAFFOLD = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def scaffold_dir(tmp_path: Path) -> Path:
    _write(tmp_path / "scaffold.yaml", SCAFFOLD_YAML)
    _write(tmp_path / "instructions" / "common" / "10-task.md", "INSTRUCTIONS_BODY\n")
    _write(tmp_path / "instructions" / "common" / "20-eval.md", "WORKFLOW_BODY\n")
    _write(tmp_path / "instructions" / "amd" / "40-amd.md", "AMD_FRAGMENT_BODY\n")
    _write(tmp_path / "instructions" / "cuda" / "40-cuda.md", "CUDA_FRAGMENT_BODY\n")
    _write(tmp_path / "multiagent" / "roles.yaml", ROLES_YAML)
    _write(tmp_path / "knowledge_base" / "amd.md", "REFERENCE_BODY\n")
    # backend docs marker so mount logic does not trigger a git submodule init
    (tmp_path / "knowledge_base" / "kb_amd" / "amd-docs").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def config(scaffold_dir: Path) -> ScaffoldConfig:
    return ScaffoldConfig.load(scaffold_dir)


class FakeApi:
    def __init__(self):
        self.create_calls = []

    def create_run(self, plugin, target, shapes, target_speedup=None, benchmark_policy=None):
        self.create_calls.append(
            {"plugin": plugin, "target": target, "shapes": shapes,
             "target_speedup": target_speedup, "benchmark_policy": benchmark_policy}
        )
        return {"run_id": "run-123",
                "benchmark_shapes": [{"task_slug": f"slug-{i}"} for i in range(len(shapes))]}


class FakeRuntime:
    is_podman = True

    def __init__(self):
        self.removed = []

    def exists(self, name):
        return False

    def remove(self, name, force=True):
        self.removed.append(name)


def _orch(config) -> Orchestrator:
    return Orchestrator(config=config, api=FakeApi(), runtime=FakeRuntime())


def test_load_run_config(config):
    rc = config.run
    assert rc.plugin == "aiter.rms_norm"
    assert rc.target == "hip"
    assert rc.shapes == [{"m": 16, "n": 32, "dtype": "bf16"}]
    assert rc.model == "gpt-5.5"
    assert rc.n_agents == 3
    assert rc.container_runtime == "podman"
    assert rc.starter_mode == "none"
    assert rc.poll_interval == 60.0
    # not set in the fixture yaml -> defaults to 1
    assert rc.max_iterations == 1


def test_max_iterations_loaded_from_yaml(scaffold_dir):
    text = (scaffold_dir / "scaffold.yaml").read_text().replace(
        "n_agents: 3", "n_agents: 3\n  max_iterations: 5", 1)
    (scaffold_dir / "scaffold.yaml").write_text(text)
    cfg = ScaffoldConfig.load(scaffold_dir)
    assert cfg.run.max_iterations == 5


def test_run_config_defaults_when_absent(tmp_path):
    (tmp_path / "scaffold.yaml").write_text("multiagent:\n  role_set: x\n")
    cfg = ScaffoldConfig.load(tmp_path)
    assert cfg.run.n_agents == 1
    assert cfg.run.image == "turbo-kernel-agent"
    assert cfg.run.plugin is None
    assert cfg.run.shapes is None
    assert cfg.run.auto_terminate is False
    assert cfg.run.max_iterations == 1


def test_missing_scaffold_yaml_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="agents require a scaffold"):
        ScaffoldConfig.load(tmp_path)


def test_backend_and_kernel_file(config):
    assert config.backend_for("hip") == "amd"
    assert config.backend_for("cuda") == "cuda"
    assert config.kernel_file_for("hip") == "kernel.cu"
    assert config.kernel_file_for("cutedsl") == "kernel.py"
    with pytest.raises(KeyError):
        config.backend_for("sycl")


def test_knowledge_base_path_for(config):
    assert config.knowledge_base.path_for("amd") == "knowledge_base/kb_amd"
    assert config.knowledge_base.path_for("cuda") == "knowledge_base/kb_cuda"


def test_kb_disabled_returns_none(scaffold_dir):
    text = (scaffold_dir / "scaffold.yaml").read_text().replace("enabled: true", "enabled: false", 1)
    (scaffold_dir / "scaffold.yaml").write_text(text)
    cfg = ScaffoldConfig.load(scaffold_dir)
    assert cfg.knowledge_base.path_for("amd") is None


def test_cli_for_model():
    assert cli_for_model("claude-sonnet-4") == "claude"
    assert cli_for_model("gpt-5.5") == "codex"
    assert cli_for_model("o3-mini") == "codex"
    with pytest.raises(ValueError):
        cli_for_model("gemini-pro")


def test_prompt_rendering(config):
    prompts = Prompts(config)
    ctx = PromptContext(task_slug="slugX", kernel_file="kernel.cu", instructions_file="AGENTS.md")
    assert prompts.default_prompt(ctx) == "Default slugX kernel.cu AGENTS.md"
    assert prompts.continue_prompt(ctx) == "Continue slugX"
    specs = prompts.specialist_prompts(ctx)
    assert specs == ["Spec1 slugX kernel.cu", "Spec2 slugX"]
    # agent 1/2 get specialists, agent 3 falls back to default (and warns loudly)
    assert prompts.prompt_for_agent(1, ctx) == "Spec1 slugX kernel.cu"
    with pytest.warns(UserWarning, match="exceeds the 2 specialist"):
        assert prompts.prompt_for_agent(3, ctx) == "Default slugX kernel.cu AGENTS.md"


CTX = PromptContext(task_slug="slugX", kernel_file="kernel.cu", instructions_file="AGENTS.md")


def test_unknown_role_set_fails_fast(scaffold_dir):
    text = (scaffold_dir / "scaffold.yaml").read_text().replace(
        "role_set: gemm_specialists", "role_set: gemm_speciallists", 1)
    (scaffold_dir / "scaffold.yaml").write_text(text)
    prompts = Prompts(ScaffoldConfig.load(scaffold_dir))
    with pytest.raises(ValueError, match="role_set 'gemm_speciallists' not found"):
        prompts.specialist_prompts(CTX)


def test_empty_role_set_yields_no_specialists(scaffold_dir):
    text = (scaffold_dir / "scaffold.yaml").read_text().replace(
        "role_set: gemm_specialists", 'role_set: ""', 1)
    (scaffold_dir / "scaffold.yaml").write_text(text)
    prompts = Prompts(ScaffoldConfig.load(scaffold_dir))
    assert prompts.specialist_prompts(CTX) == []
    # with no specialists everyone gets default, no warning
    assert prompts.prompt_for_agent(5, CTX) == "Default slugX kernel.cu AGENTS.md"


def test_role_set_must_be_nonempty_list(scaffold_dir):
    (scaffold_dir / "multiagent" / "roles.yaml").write_text(
        'default_role: "Default $TASK_SLUG $KERNEL_FILE $INSTRUCTIONS_FILE"\n'
        'continue_role: "Continue $TASK_SLUG"\n'
        "gemm_specialists: {}\n")
    prompts = Prompts(ScaffoldConfig.load(scaffold_dir))
    with pytest.raises(ValueError, match="must be a non-empty list"):
        prompts.specialist_prompts(CTX)


def test_specialist_missing_prompt_field_fails_fast(scaffold_dir):
    (scaffold_dir / "multiagent" / "roles.yaml").write_text(
        'default_role: "Default $TASK_SLUG"\n'
        'continue_role: "Continue $TASK_SLUG"\n'
        "gemm_specialists:\n"
        "  - notprompt: foo\n")
    prompts = Prompts(ScaffoldConfig.load(scaffold_dir))
    with pytest.raises(ValueError, match="missing a 'prompt' field"):
        prompts.specialist_prompts(CTX)


def test_unknown_placeholder_in_role_fails_fast(scaffold_dir):
    (scaffold_dir / "multiagent" / "roles.yaml").write_text(
        'default_role: "Default $TASK_SLG"\n'
        'continue_role: "Continue $TASK_SLUG"\n'
        "gemm_specialists:\n"
        '  - prompt: "Spec $TASK_SLUG"\n')
    prompts = Prompts(ScaffoldConfig.load(scaffold_dir))
    with pytest.raises(ValueError, match=r"unknown placeholder\(s\) \$TASK_SLG"):
        prompts.default_prompt(CTX)


def test_missing_default_role_fails_fast(scaffold_dir):
    (scaffold_dir / "multiagent" / "roles.yaml").write_text(
        'continue_role: "Continue $TASK_SLUG"\n'
        "gemm_specialists:\n"
        '  - prompt: "Spec $TASK_SLUG"\n')
    prompts = Prompts(ScaffoldConfig.load(scaffold_dir))
    with pytest.raises(ValueError, match="default_role missing or empty"):
        prompts.default_prompt(CTX)


def test_build_instructions_composes_common_backend_and_kb(config):
    prompts = Prompts(config)
    ctx = PromptContext(task_slug="s", kernel_file="kernel.cu", instructions_file="AGENTS.md")
    body = prompts.build_instructions("hip", ctx)
    assert "INSTRUCTIONS_BODY" in body
    assert "WORKFLOW_BODY" in body
    assert "REFERENCE_BODY" in body
    # hip -> amd backend bucket, never the cuda one
    assert "AMD_FRAGMENT_BODY" in body
    assert "CUDA_FRAGMENT_BODY" not in body


def test_build_instructions_renders_kernel_file(config):
    prompts = Prompts(config)
    _write(config.scaffold_dir / "instructions" / "common" / "15-render.md", "use $KERNEL_FILE here\n")
    ctx = PromptContext(task_slug="s", kernel_file="kernel.py", instructions_file="AGENTS.md")
    body = prompts.build_instructions("cutedsl", ctx)
    assert "use kernel.py here" in body
    assert "$KERNEL_FILE" not in body


def test_mount_as_arg():
    assert Mount("/a", "/b", readonly=True).as_arg() == "/a:/b:ro"
    assert Mount("/a", "/b").as_arg() == "/a:/b"


def test_knowledge_base_mount_uses_backend_subdir(config):
    orch = _orch(config)
    mount = orch._knowledge_base_mount("hip")
    assert mount is not None
    assert mount.readonly is True
    assert mount.target == "/kb"
    assert mount.source.endswith("knowledge_base/kb_amd")


def test_create_run_injects_hip_timing_policy(config):
    orch = _orch(config)
    run = orch.create_run("aiter.rms_norm", "hip", [{"m": 16}])
    assert isinstance(run, RunInfo)
    assert run.run_id == "run-123"
    assert run.task_slug == "slug-0"
    assert run.task_slugs == ("slug-0",)
    assert orch.api.create_calls[-1]["benchmark_policy"] == {"timing_mode": "standard"}


def test_create_run_no_policy_for_cuda(config):
    orch = _orch(config)
    orch.create_run("torch.linear", "cuda", [{"m": 16}])
    assert orch.api.create_calls[-1]["benchmark_policy"] is None


def test_create_run_multishape_slug_is_hash(config):
    orch = _orch(config)
    run = orch.create_run("aiter.rms_norm", "hip", [{"m": 16}, {"m": 32}, {"m": 64}])
    assert run.task_slugs == ("slug-0", "slug-1", "slug-2")
    assert run.task_slug.startswith("multishape-3-")
    assert orch.api.create_calls[-1]["shapes"] == [{"m": 16}, {"m": 32}, {"m": 64}]


def test_start_agent_preset_requires_path(config):
    orch = _orch(config)
    run = RunInfo(run_id="r", task_slug="s", plugin="p", target="hip")
    with pytest.raises(ValueError, match="preset"):
        orch.start_agent(run, agent_index=1, model="gpt-5.5", starter_mode="preset")


def test_start_agent_preset_rejects_directory(config, tmp_path):
    orch = _orch(config)
    run = RunInfo(run_id="r", task_slug="s", plugin="p", target="cuda")
    with pytest.raises(ValueError, match="is not a file"):
        orch.start_agent(run, agent_index=1, model="gpt-5.5",
                         starter_mode="preset", preset_path=str(tmp_path))


def test_start_agent_preset_mounts_file_readonly(config, monkeypatch, tmp_path):
    orch = _orch(config)
    captured = {}
    _stub_start_agent(orch, monkeypatch, captured, tmp_path)
    monkeypatch.setattr(orch.runtime, "run_detached",
                        lambda **kw: captured.update(kw) or "cid-1", raising=False)

    preset_file = tmp_path / "my_starter.cu"
    preset_file.write_text("// starter\n")
    run = RunInfo(run_id="r", task_slug="s", plugin="p", target="cuda")
    orch.start_agent(run, agent_index=1, model="gpt-5.5",
                     starter_mode="preset", preset_path=str(preset_file))

    assert captured["env"]["BENCH_STARTER_MODE"] == "preset"
    assert captured["env"]["BENCH_PRESET_PATH"] == "/preset/my_starter.cu"
    preset_mounts = [m for m in captured["mounts"] if m.target == "/preset/my_starter.cu"]
    assert len(preset_mounts) == 1
    assert preset_mounts[0].readonly is True
    assert preset_mounts[0].source == str(preset_file.resolve())


def test_bench_cli_preset_copies_file_into_kernel(monkeypatch, tmp_path):
    import scaffold.client.bench_cli as bc

    preset = tmp_path / "starter.cu"
    preset.write_text("// my preset kernel\n")
    out = tmp_path / "kernel.cu"
    monkeypatch.setattr(bc, "_STARTER_MODE", "preset")
    monkeypatch.setattr(bc, "_PRESET_PATH", str(preset))

    bc._fetch_starter("run-1", "torch.linear", str(out))
    assert out.read_text() == "// my preset kernel\n"


class _WatchApi:
    def __init__(self, run_info, best_sequence):
        self._run_info = run_info
        self._best = list(best_sequence)
        self.best_calls = 0

    def get_run(self, run_id):
        return self._run_info

    def best_kernel(self, run_id):
        i = min(self.best_calls, len(self._best) - 1)
        self.best_calls += 1
        return self._best[i]


def _no_hang_sleep(monkeypatch, slept, limit=10):
    def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) > limit:
            raise RuntimeError("watch_speedup did not terminate")
    monkeypatch.setattr("scaffold.client.orchestrator.time.sleep", fake_sleep)


def test_watch_speedup_requires_target(config):
    orch = _orch(config)
    orch.api = _WatchApi({"target_speedup": None}, [])
    with pytest.raises(ValueError, match="no target_speedup"):
        orch.watch_speedup("run-1")


def test_watch_speedup_finishes_immediately_when_target_met(config, monkeypatch):
    orch = _orch(config)
    orch.api = _WatchApi({"target_speedup": 1.2},
                         [{"kernel_us": 40.0, "baseline_us": 50.0}])  # 1.25x >= 1.2
    finished = {}
    monkeypatch.setattr(orch, "finish_run",
                        lambda run, postprocess=True: finished.update(run=run, pp=postprocess) or {"done": True})
    slept = []
    _no_hang_sleep(monkeypatch, slept)

    result = orch.watch_speedup("run-1", poll_interval=5, postprocess=False)
    assert result == {"done": True}
    assert finished == {"run": "run-1", "pp": False}
    assert slept == []  # target hit on first poll; never slept


def test_watch_speedup_polls_until_target_then_finishes(config, monkeypatch):
    orch = _orch(config)
    seq = [
        None,                                       # no kernel on leaderboard yet
        {"kernel_us": 50.0, "baseline_us": 50.0},   # 1.00x < 1.2
        {"kernel_us": 40.0, "baseline_us": 50.0},   # 1.25x >= 1.2
    ]
    orch.api = _WatchApi({"target_speedup": 1.2}, seq)
    monkeypatch.setattr(orch, "finish_run", lambda run, postprocess=True: {"done": True})
    slept = []
    _no_hang_sleep(monkeypatch, slept)

    result = orch.watch_speedup("run-1", poll_interval=7)
    assert result == {"done": True}
    assert slept == [7, 7]  # slept after the None poll and the 1.0x poll, finished on the 3rd


def test_copy_agent_skills_selects_backend(config, scaffold_dir, tmp_path):
    skills_root = scaffold_dir / "agent_runner" / ".agents" / "skills"
    _write(skills_root / "eval-service" / "SKILL.md", "common\n")
    _write(skills_root / "amd" / "profiling" / "SKILL.md", "rocprof\n")
    _write(skills_root / "cuda" / "profiling" / "SKILL.md", "ncu\n")

    orch = _orch(config)
    workdir = tmp_path / "wd"
    workdir.mkdir()
    orch._copy_agent_skills(workdir, "amd")

    dst = workdir / ".agents" / "skills"
    assert (dst / "eval-service" / "SKILL.md").read_text() == "common\n"
    assert (dst / "profiling" / "SKILL.md").read_text() == "rocprof\n"
    assert not (dst / "amd").exists()
    assert not (dst / "cuda").exists()


def test_agent_command_is_pure_string(config):
    orch = _orch(config)
    cmd = orch._agent_command("codex", "gpt-5.5", "kernel.cu", max_iterations=3)
    assert cmd[0] == "bash" and cmd[1] == "-c"
    assert "/workspace/kernel.cu" in cmd[2]
    assert "$AGENT_PROMPT" in cmd[2] and "$AGENT_CONTINUE_PROMPT" in cmd[2]


def test_agent_command_is_bounded_not_infinite(config):
    orch = _orch(config)
    cmd = orch._agent_command("codex", "gpt-5.5", "kernel.cu", max_iterations=3)
    loop = cmd[2]
    # never an unbounded loop again
    assert "while true" not in loop
    # bounded by a counted for-loop over the requested iteration count
    assert "seq 1 3" in loop
    assert "reached max iterations (3)" in loop


def test_agent_command_respects_max_iterations_value(config):
    orch = _orch(config)
    cmd = orch._agent_command("claude", "claude-sonnet-4", "kernel.py", max_iterations=7)
    assert "seq 1 7" in cmd[2]
    assert "reached max iterations (7)" in cmd[2]
    assert "while true" not in cmd[2]


def _stub_start_agent(orch, monkeypatch, captured, tmp_path):
    def fake_cmd(cli, model, kernel_file, max_iterations):
        captured["max_iterations"] = max_iterations
        return ["bash", "-c", "echo ok"]

    monkeypatch.setattr(orch, "_agent_command", fake_cmd)
    monkeypatch.setattr(orch, "_prepare_workdir", lambda *a, **k: (tmp_path, "AGENTS.md"))
    monkeypatch.setattr(orch, "_knowledge_base_mount", lambda target: None)
    monkeypatch.setattr(orch.runtime, "run_detached", lambda **kw: "cid-1", raising=False)
    monkeypatch.setattr(orch.api, "mint_user_key", lambda: "key", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_start_agent_uses_config_max_iterations_by_default(config, monkeypatch, tmp_path):
    orch = _orch(config)
    captured = {}
    _stub_start_agent(orch, monkeypatch, captured, tmp_path)

    run = RunInfo(run_id="r", task_slug="s", plugin="p", target="hip")
    orch.start_agent(run, agent_index=1, model="gpt-5.5")
    # scaffold.yaml in the fixture sets no max_iterations -> default 1
    assert captured["max_iterations"] == 1


def test_start_agent_explicit_max_iterations_overrides_config(config, monkeypatch, tmp_path):
    orch = _orch(config)
    captured = {}
    _stub_start_agent(orch, monkeypatch, captured, tmp_path)

    run = RunInfo(run_id="r", task_slug="s", plugin="p", target="hip")
    orch.start_agent(run, agent_index=1, model="gpt-5.5", max_iterations=10)
    assert captured["max_iterations"] == 10


def test_copy_agent_skills_selects_cuda_backend(config, scaffold_dir, tmp_path):
    skills_root = scaffold_dir / "agent_runner" / ".agents" / "skills"
    _write(skills_root / "eval-service" / "SKILL.md", "common\n")
    _write(skills_root / "amd" / "profiling" / "SKILL.md", "rocprof\n")
    _write(skills_root / "cuda" / "profiling" / "SKILL.md", "ncu\n")

    orch = _orch(config)
    workdir = tmp_path / "wd"
    workdir.mkdir()
    orch._copy_agent_skills(workdir, "cuda")

    dst = workdir / ".agents" / "skills"
    assert (dst / "eval-service" / "SKILL.md").read_text() == "common\n"
    # cuda backend gets the NCU profiling skill, never the rocprof one
    assert (dst / "profiling" / "SKILL.md").read_text() == "ncu\n"
    assert not (dst / "amd").exists()
    assert not (dst / "cuda").exists()


# --- guards against deleting/renaming the real shipped files ---

def test_real_scaffold_ships_both_backend_profiling_skills():
    skills = REAL_SCAFFOLD / "agent_runner" / ".agents" / "skills"
    assert (skills / "eval-service" / "SKILL.md").is_file()
    amd = (skills / "amd" / "profiling" / "SKILL.md").read_text()
    cuda = (skills / "cuda" / "profiling" / "SKILL.md").read_text()
    assert "rocprof" in amd.lower()
    assert "ncu" in cuda.lower() or "nsight" in cuda.lower()


def test_run_agents_shell_launcher_is_bounded():
    # the shell launch path must not regress to an infinite respawn loop either
    script = (REAL_SCAFFOLD / "agent_runner" / "run_agents.sh").read_text()
    assert "while true" not in script
    assert "MAX_ITERATIONS=1" in script
    assert "--max-iterations" in script
    assert "seq 1 $MAX_ITERATIONS" in script


@pytest.mark.parametrize("target,backend_marker", [("hip", "rocprof"), ("cuda", "nsight")])
def test_real_instructions_nudge_profiling(target, backend_marker):
    cfg = ScaffoldConfig.load(REAL_SCAFFOLD)
    prompts = Prompts(cfg)
    ctx = PromptContext(task_slug="s", kernel_file="kernel.cu", instructions_file="AGENTS.md")
    body = prompts.build_instructions(target, ctx).lower()
    # every agent is told skills exist and to profile before sweeping
    assert "profiling" in body
    assert "skill" in body
    assert backend_marker in body


# --- admin_cli (kernel-orchestrator) CLI surface ---

def test_resolve_keeps_explicit_including_zero():
    assert admin_cli._resolve(None, 7) == 7
    assert admin_cli._resolve(3, 7) == 3
    # 0 is a real value, not "unset" -> must be kept, never replaced by fallback
    assert admin_cli._resolve(0, 7) == 0
    assert admin_cli._resolve(False, True) is False


def test_parse_shapes_single_object_becomes_list():
    assert admin_cli._parse_shapes('{"m": 16, "n": 32}') == [{"m": 16, "n": 32}]


def test_parse_shapes_passes_list_through():
    assert admin_cli._parse_shapes('[{"m": 16}, {"m": 32}]') == [{"m": 16}, {"m": 32}]


class FakeOrchCLI:
    """Records orchestrator calls so we can assert what the CLI resolved/forwarded."""

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.runs_dir = Path("/tmp/_unused_runs")
        import types
        self.runtime = types.SimpleNamespace(logs_command=lambda name: f"logs {name}")

    def _handle(self, idx=1):
        return AgentHandle(container_name=f"kernel-agent-run-xyz-{idx}", agent_index=idx,
                           model="gpt-5.5", cli="codex", workdir=Path("/tmp/wd"),
                           api_key="k", container_id="cid")

    def build_image(self):
        self.calls.append(("build_image",))

    def create_run(self, plugin, target, shapes, target_speedup=None):
        self.calls.append(("create_run", plugin, target, shapes, target_speedup))
        return RunInfo(run_id="run-xyz", task_slug="slug", plugin=plugin, target=target,
                       task_slugs=("slug",))

    def launch(self, run, model, n_agents, starter_mode, preset_path, start_index, max_iterations):
        self.calls.append(("launch", {"model": model, "n_agents": n_agents,
                                       "starter_mode": starter_mode, "start_index": start_index,
                                       "max_iterations": max_iterations}))
        return [self._handle(i) for i in range(start_index, start_index + n_agents)]

    def start_agent(self, run, agent_index, model, starter_mode, preset_path, max_iterations):
        self.calls.append(("start_agent", {"agent_index": agent_index, "model": model,
                                            "starter_mode": starter_mode,
                                            "max_iterations": max_iterations}))
        return self._handle(agent_index)

    def stop_agent(self, name):
        self.calls.append(("stop_agent", name))

    def stop_run(self, run_id):
        self.calls.append(("stop_run", run_id))
        return ["c1", "c2"]

    def finish_run(self, run_id, postprocess=True):
        self.calls.append(("finish_run", run_id, postprocess))
        return {"run_id": run_id}


@pytest.fixture
def cli(config, monkeypatch):
    fake = FakeOrchCLI(config)
    monkeypatch.setattr(admin_cli, "_orchestrator", lambda args: fake)

    def _run(argv):
        # default --scaffold-dir is harmless since _orchestrator is stubbed
        return admin_cli.main(argv), fake

    return _run


def _only(fake, name):
    return [c for c in fake.calls if c[0] == name]


def test_cli_launch_uses_config_default_max_iterations(cli):
    rc, fake = cli(["launch", "--no-build"])
    assert rc == 0
    launch = _only(fake, "launch")[0][1]
    # fixture yaml sets no max_iterations -> default 1
    assert launch["max_iterations"] == 1
    # config-driven values flow through
    assert launch["model"] == "gpt-5.5"
    assert launch["n_agents"] == 3
    # no rebuild requested
    assert _only(fake, "build_image") == []


def test_cli_launch_flag_overrides_max_iterations(cli):
    rc, fake = cli(["launch", "--no-build", "--max-iterations", "5", "--n-agents", "2"])
    assert rc == 0
    launch = _only(fake, "launch")[0][1]
    assert launch["max_iterations"] == 5
    assert launch["n_agents"] == 2


def test_cli_launch_builds_image_without_no_build(cli):
    rc, fake = cli(["launch"])
    assert rc == 0
    assert _only(fake, "build_image")


def test_cli_launch_missing_required_config_fails(config, monkeypatch, scaffold_dir):
    # strip required fields from the yaml so nothing resolves them
    (scaffold_dir / "scaffold.yaml").write_text("multiagent:\n  role_set: x\n")
    fake = FakeOrchCLI(ScaffoldConfig.load(scaffold_dir))
    monkeypatch.setattr(admin_cli, "_orchestrator", lambda args: fake)
    rc = admin_cli.main(["launch", "--no-build"])
    assert rc == 1
    assert _only(fake, "create_run") == []


def test_cli_launch_auto_terminate_requires_target_speedup(cli):
    rc, fake = cli(["launch", "--no-build", "--auto-terminate"])
    assert rc == 1
    assert _only(fake, "launch") == []


def test_cli_start_agent_forwards_max_iterations(cli):
    rc, fake = cli(["start-agent", "--no-build", "--run-id", "run-xyz",
                    "--task-slug", "slug", "--agent-index", "4", "--max-iterations", "9"])
    assert rc == 0
    sa = _only(fake, "start_agent")[0][1]
    assert sa["agent_index"] == 4
    assert sa["max_iterations"] == 9


def test_cli_start_agent_defaults_max_iterations_to_config(cli):
    rc, fake = cli(["start-agent", "--no-build", "--run-id", "run-xyz",
                    "--task-slug", "slug", "--agent-index", "2"])
    assert rc == 0
    sa = _only(fake, "start_agent")[0][1]
    assert sa["max_iterations"] == 1


def test_cli_stop_dispatches(cli):
    rc, fake = cli(["stop", "kernel-agent-run-xyz-1"])
    assert rc == 0
    assert _only(fake, "stop_agent") == [("stop_agent", "kernel-agent-run-xyz-1")]


def test_cli_stop_run_dispatches(cli):
    rc, fake = cli(["stop-run", "run-xyz"])
    assert rc == 0
    assert _only(fake, "stop_run") == [("stop_run", "run-xyz")]


def test_cli_finish_dispatches(cli):
    rc, fake = cli(["finish", "run-xyz"])
    assert rc == 0
    assert _only(fake, "finish_run") == [("finish_run", "run-xyz", True)]


def test_cli_finish_no_postprocess(cli):
    rc, fake = cli(["finish", "run-xyz", "--no-postprocess"])
    assert rc == 0
    assert _only(fake, "finish_run") == [("finish_run", "run-xyz", False)]


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        admin_cli.main([])


def test_cli_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        admin_cli.main(["frobnicate"])
