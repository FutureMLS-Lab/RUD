from dataclasses import dataclass
from pathlib import Path

import yaml

BACKEND_BY_TARGET = {
    "cuda": "cuda",
    "cutedsl": "cuda",
    "triton": "cuda",
    "hip": "amd",
}

KERNEL_FILE_BY_TARGET = {
    "cuda": "kernel.cu",
    "hip": "kernel.cu",
    "cutedsl": "kernel.py",
    "triton": "kernel.py",
}

KB_DOCS_SUBDIR = {
    "cuda": "nvidia-docs",
    "amd": "amd-docs",
}


def _normalize_shapes(raw) -> list[dict] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


@dataclass(frozen=True)
class KnowledgeBase:
    enabled: bool
    mount_point: str
    submodule: str | None
    paths: dict[str, str]

    def path_for(self, backend: str) -> str | None:
        if not self.enabled:
            return None
        if backend not in self.paths:
            raise ValueError(
                f"knowledge_base is enabled but has no '{backend}' path configured; "
                f"add a '{backend}:' block with a 'path' to scaffold.yaml "
                f"(configured backends: {sorted(self.paths) or 'none'})"
            )
        return self.paths[backend]


@dataclass(frozen=True)
class Multiagent:
    roles_file: str
    role_set: str


@dataclass(frozen=True)
class RunConfig:
    plugin: str | None
    target: str | None
    shapes: list[dict] | None
    model: str | None
    n_agents: int
    start_index: int
    target_speedup: float | None
    starter_mode: str
    preset_path: str | None
    auto_terminate: bool
    poll_interval: float
    image: str
    container_runtime: str | None
    max_iterations: int


@dataclass(frozen=True)
class ScaffoldConfig:
    scaffold_dir: Path
    knowledge_base: KnowledgeBase
    multiagent: Multiagent
    run: RunConfig

    @classmethod
    def load(cls, scaffold_dir: Path | str) -> "ScaffoldConfig":
        scaffold_dir = Path(scaffold_dir).resolve()
        scaffold_yaml = scaffold_dir / "scaffold.yaml"
        if not scaffold_yaml.is_file():
            raise FileNotFoundError(
                f"scaffold config not found at {scaffold_yaml}; agents require a scaffold to run"
            )
        raw = yaml.safe_load(scaffold_yaml.read_text()) or {}

        kb_raw = raw.get("knowledge_base", {})
        kb = KnowledgeBase(
            enabled=bool(kb_raw.get("enabled")),
            mount_point=kb_raw.get("mount_point", "/kb"),
            submodule=kb_raw.get("submodule"),
            paths={b: kb_raw[b]["path"] for b in ("cuda", "amd") if b in kb_raw},
        )

        ma_raw = raw.get("multiagent", {})
        multiagent = Multiagent(
            roles_file=ma_raw.get("roles_file", "multiagent/roles.yaml"),
            role_set=ma_raw.get("role_set", ""),
        )

        run_raw = raw.get("run", {})
        run = RunConfig(
            plugin=run_raw.get("plugin"),
            target=run_raw.get("target"),
            shapes=_normalize_shapes(run_raw.get("shapes")),
            model=run_raw.get("model"),
            n_agents=run_raw.get("n_agents", 1),
            start_index=run_raw.get("start_index", 1),
            target_speedup=run_raw.get("target_speedup"),
            starter_mode=run_raw.get("starter_mode", "none"),
            preset_path=run_raw.get("preset_path"),
            auto_terminate=bool(run_raw.get("auto_terminate", False)),
            poll_interval=run_raw.get("poll_interval", 60.0),
            image=run_raw.get("image", "turbo-kernel-agent"),
            container_runtime=run_raw.get("container_runtime"),
            max_iterations=int(run_raw.get("max_iterations", 1)),
        )

        return cls(scaffold_dir=scaffold_dir, knowledge_base=kb,
                   multiagent=multiagent, run=run)

    def backend_for(self, target: str) -> str:
        return BACKEND_BY_TARGET[target]

    def kernel_file_for(self, target: str) -> str:
        return KERNEL_FILE_BY_TARGET[target]
