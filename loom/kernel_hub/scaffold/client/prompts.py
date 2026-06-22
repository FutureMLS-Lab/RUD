import re
import warnings
from dataclasses import dataclass

import yaml

from scaffold.client.config import ScaffoldConfig

# Role templates may reference only these placeholders. Any other `$TOKEN` in a
# role is treated as a typo and fails loudly (instructions markdown is exempt —
# it legitimately contains shell `$VARS`).
_KNOWN_PLACEHOLDERS = {"$TASK_SLUG", "$KERNEL_FILE", "$INSTRUCTIONS_FILE"}
_PLACEHOLDER_RE = re.compile(r"\$[A-Z][A-Z0-9_]*")


@dataclass(frozen=True)
class PromptContext:
    task_slug: str
    kernel_file: str
    instructions_file: str


class Prompts:
    def __init__(self, config: ScaffoldConfig):
        self.config = config
        self._roles_path = config.scaffold_dir / config.multiagent.roles_file
        self._roles = yaml.safe_load(self._roles_path.read_text()) or {}

    def _render(self, text: str, ctx: PromptContext) -> str:
        return (
            text.strip()
            .replace("$TASK_SLUG", ctx.task_slug)
            .replace("$KERNEL_FILE", ctx.kernel_file)
            .replace("$INSTRUCTIONS_FILE", ctx.instructions_file)
        )

    def _render_role(self, text: str, ctx: PromptContext, what: str) -> str:
        unknown = sorted(set(_PLACEHOLDER_RE.findall(text)) - _KNOWN_PLACEHOLDERS)
        if unknown:
            raise ValueError(
                f"{what} in {self._roles_path.name} uses unknown placeholder(s) "
                f"{', '.join(unknown)}; supported: {', '.join(sorted(_KNOWN_PLACEHOLDERS))}"
            )
        return self._render(text, ctx)

    def _role_set_names(self) -> list[str]:
        # top-level list-valued keys are role sets (default_role/continue_role are strings)
        return [k for k, v in self._roles.items() if isinstance(v, list)]

    def default_prompt(self, ctx: PromptContext) -> str:
        text = self._roles.get("default_role", "")
        if not text or not text.strip():
            raise ValueError(f"default_role missing or empty in {self._roles_path.name}")
        return self._render_role(text, ctx, "default_role")

    def continue_prompt(self, ctx: PromptContext) -> str:
        text = self._roles.get("continue_role", "")
        if not text or not text.strip():
            raise ValueError(f"continue_role missing or empty in {self._roles_path.name}")
        return self._render_role(text, ctx, "continue_role")

    def specialist_prompts(self, ctx: PromptContext) -> list[str]:
        role_set = self.config.multiagent.role_set
        if not role_set:
            return []
        if role_set not in self._roles:
            available = ", ".join(self._role_set_names()) or "(none)"
            raise ValueError(
                f"role_set '{role_set}' not found in {self._roles_path.name}; "
                f"available role sets: {available}"
            )
        specs = self._roles[role_set]
        if not isinstance(specs, list) or not specs:
            raise ValueError(
                f"role_set '{role_set}' in {self._roles_path.name} must be a non-empty list"
            )
        out: list[str] = []
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict) or "prompt" not in spec:
                raise ValueError(
                    f"role_set '{role_set}' entry {i} in {self._roles_path.name} "
                    f"is missing a 'prompt' field"
                )
            out.append(self._render_role(spec["prompt"], ctx, f"role_set '{role_set}' entry {i}"))
        return out

    def prompt_for_agent(self, agent_index: int, ctx: PromptContext) -> str:
        specialists = self.specialist_prompts(ctx)
        idx = agent_index - 1
        if 0 <= idx < len(specialists):
            return specialists[idx]
        if specialists:
            warnings.warn(
                f"agent_index {agent_index} exceeds the {len(specialists)} specialist role(s) "
                f"in role_set '{self.config.multiagent.role_set}'; falling back to default_role",
                stacklevel=2,
            )
        return self.default_prompt(ctx)

    def build_instructions(self, target: str, ctx: PromptContext) -> str:
        scaffold_dir = self.config.scaffold_dir
        backend = self.config.backend_for(target)
        instructions_dir = scaffold_dir / "instructions"
        parts: list[str] = []
        for bucket in (instructions_dir / "common", instructions_dir / backend):
            if bucket.is_dir():
                parts.extend(f.read_text() for f in sorted(bucket.glob("*.md")))
        if self.config.knowledge_base.enabled:
            kb_intro = scaffold_dir / "knowledge_base" / f"{backend}.md"
            if kb_intro.is_file():
                parts.append(kb_intro.read_text())
        return "\n".join(self._render(part, ctx) for part in parts)
