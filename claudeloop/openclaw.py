"""First-class OpenClaw gateway integration for claudeloop."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpenClawConfig:
    enabled: bool = False
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    method: str = "POST"
    timeout_ms: int = 10000
    debug: bool = False
    hook: str = "wake"
    wake_mode: str = "now"
    agent_name: str = "claudeloop"
    agent_id: str = ""
    deliver: bool = False
    channel: str = ""
    to: str = ""


def _parse_header(raw: str) -> tuple[str, str]:
    if ":" in raw:
        key, value = raw.split(":", 1)
    elif "=" in raw:
        key, value = raw.split("=", 1)
    else:
        raise ValueError(f"Invalid header {raw!r}; use 'Name: value' or 'Name=value'")
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError("OpenClaw header name cannot be empty")
    return key, value


def _load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OpenClaw config must be a JSON object")
    return data


def build_openclaw_config(
    *,
    enabled: bool = False,
    url: str | None = None,
    token: str | None = None,
    headers: list[str] | None = None,
    config_path: Path | None = None,
    timeout_ms: int | None = None,
    hook: str | None = None,
    wake_mode: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    deliver: bool | None = None,
    channel: str | None = None,
    to: str | None = None,
    debug: bool = False,
) -> OpenClawConfig:
    """Build a direct claudeloop OpenClaw config from CLI args and optional JSON."""
    data = _load_config_file(config_path)
    cfg_headers: dict[str, str] = {}
    file_headers = data.get("headers", {})
    if isinstance(file_headers, dict):
        cfg_headers.update({str(k): str(v) for k, v in file_headers.items()})
    for raw in headers or []:
        key, value = _parse_header(raw)
        cfg_headers[key] = value
    cfg_token = str(token or data.get("token", "")).strip()
    if cfg_token and not any(k.lower() == "authorization" for k in cfg_headers):
        cfg_headers["Authorization"] = f"Bearer {cfg_token}"

    cfg_url = str(url or data.get("url", "")).strip()
    cfg_enabled = bool(enabled or data.get("enabled", False) or cfg_url)
    cfg_timeout = int(timeout_ms if timeout_ms is not None else data.get("timeout", data.get("timeout_ms", 10000)))
    cfg_debug = bool(debug or data.get("debug", False))
    cfg_hook = str(hook or data.get("hook", data.get("endpoint", ""))).strip().lower()
    if not cfg_hook:
        cfg_hook = "agent" if cfg_url.rstrip("/").endswith("/agent") else "wake"
    if cfg_hook not in ("wake", "agent"):
        cfg_hook = "wake"
    cfg_wake_mode = str(wake_mode or data.get("mode", data.get("wake_mode", "now"))).strip()
    if cfg_wake_mode not in ("now", "next-heartbeat"):
        cfg_wake_mode = "now"
    method = str(data.get("method", "POST")).upper()
    return OpenClawConfig(
        enabled=cfg_enabled,
        url=cfg_url,
        headers=cfg_headers,
        method=method,
        timeout_ms=cfg_timeout,
        debug=cfg_debug,
        hook=cfg_hook,
        wake_mode=cfg_wake_mode,
        agent_name=str(agent_name or data.get("agent_name", data.get("name", "claudeloop"))),
        agent_id=str(agent_id or data.get("agent_id", data.get("agentId", ""))),
        deliver=bool(deliver if deliver is not None else data.get("deliver", False)),
        channel=str(channel or data.get("channel", "")),
        to=str(to or data.get("to", "")),
    )


def config_from_environment() -> OpenClawConfig:
    """Allow non-web/manual runs to opt into claudeloop OpenClaw directly."""
    headers: dict[str, str] = {}
    raw_headers = os.environ.get("CLAUDELOOP_OPENCLAW_HEADERS", "")
    for item in [x.strip() for x in raw_headers.splitlines() if x.strip()]:
        key, value = _parse_header(item)
        headers[key] = value
    url = os.environ.get("CLAUDELOOP_OPENCLAW_URL", "")
    return OpenClawConfig(
        enabled=os.environ.get("CLAUDELOOP_OPENCLAW", "") == "1" or bool(url),
        url=url,
        headers=headers,
        timeout_ms=int(os.environ.get("CLAUDELOOP_OPENCLAW_TIMEOUT_MS", "10000")),
        debug=os.environ.get("CLAUDELOOP_OPENCLAW_DEBUG", "") == "1",
        hook=os.environ.get("CLAUDELOOP_OPENCLAW_HOOK", "wake"),
        wake_mode=os.environ.get("CLAUDELOOP_OPENCLAW_WAKE_MODE", "now"),
        agent_name=os.environ.get("CLAUDELOOP_OPENCLAW_AGENT_NAME", "claudeloop"),
        agent_id=os.environ.get("CLAUDELOOP_OPENCLAW_AGENT_ID", ""),
        deliver=os.environ.get("CLAUDELOOP_OPENCLAW_DELIVER", "") == "1",
        channel=os.environ.get("CLAUDELOOP_OPENCLAW_CHANNEL", ""),
        to=os.environ.get("CLAUDELOOP_OPENCLAW_TO", ""),
    )


class OpenClawClient:
    """Fire-and-forget event sender for an OpenClaw gateway."""

    def __init__(self, config: OpenClawConfig | None = None) -> None:
        self.config = config or OpenClawConfig()

    @property
    def enabled(self) -> bool:
        return self.config.enabled and bool(self.config.url)

    def emit(
        self,
        event: str,
        *,
        instruction: str,
        project_root: Path,
        task_slug: str | None = None,
        repo: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        details: dict[str, Any] = {
            "source": "claudeloop",
            "event": event,
            "projectRoot": str(project_root),
        }
        if task_slug:
            details["taskSlug"] = task_slug
        if repo:
            details["repo"] = repo
        if data:
            details["data"] = data
        text = f"{instruction}\n\nclaudeloop event:\n{json.dumps(details, ensure_ascii=False, indent=2)}"
        if self.config.hook == "agent":
            payload: dict[str, Any] = {
                "message": text,
                "name": self.config.agent_name,
                "wakeMode": self.config.wake_mode,
                "deliver": self.config.deliver,
            }
            if self.config.agent_id:
                payload["agentId"] = self.config.agent_id
            if self.config.channel:
                payload["channel"] = self.config.channel
            if self.config.to:
                payload["to"] = self.config.to
        else:
            payload = {"text": text, "mode": self.config.wake_mode}
        thread = threading.Thread(target=self._post, args=(payload,), daemon=True)
        thread.start()

    def _post(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.config.headers}
        req = urllib.request.Request(
            self.config.url,
            data=body,
            headers=headers,
            method=self.config.method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_ms / 1000) as resp:
                if self.config.debug:
                    print(f"[openclaw] {self.config.hook} -> {resp.status}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[openclaw] failed {self.config.hook} error={exc}", flush=True)


def openclaw_status(config: OpenClawConfig | None = None) -> str:
    """Human-readable status for startup logs."""
    cfg = config or config_from_environment()
    if not cfg.enabled:
        return "disabled"
    if not cfg.url:
        return "enabled, missing url"
    parts = [f"enabled url={cfg.url}"]
    if cfg.debug:
        parts.append("debug=on")
    parts.append(f"hook={cfg.hook}")
    parts.append(f"mode={cfg.wake_mode}")
    if cfg.hook == "agent" and cfg.deliver:
        parts.append("deliver=on")
    return ", ".join(parts)
