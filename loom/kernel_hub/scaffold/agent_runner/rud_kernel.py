#!/usr/bin/env python3
"""Loom integration helper for the kernel-optimization harness.

A thin, JSON-in/JSON-out wrapper that Loom's web backend shells out to so it can
drive kernel runs without knowing the harness internals. Every subcommand prints a
single JSON object to stdout (diagnostics go to stderr) and exits 0 on success,
1 on error. Stdlib only — no third-party deps.

Subcommands:
  plugins                       list plugins/targets/shape-templates for the form
  service-status                health-check the eval service
  up [--build]                  ensure the eval service is running (docker compose up -d)
  launch --plugin ... [opts]    ensure service up, then run_agents.sh; return run_id + containers
  status --run-id ID            docker + eval API -> agents/leaderboard/best speedup
  stop   --run-id ID            finish_run.sh (stop containers + postprocess winner)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # scaffold/agent_runner
SCAFFOLD_DIR = SCRIPT_DIR.parent                      # scaffold
REPO_ROOT = SCAFFOLD_DIR.parent                       # repo root (has docker-compose.yml, env.dev)
RUN_AGENTS = SCRIPT_DIR / "run_agents.sh"
FINISH_RUN = SCRIPT_DIR / "finish_run.sh"

# Fallback list (used only if the live registry can't be parsed). The real
# source of truth is the service's plugin registry, parsed below.
_FALLBACK_PLUGINS = [
    "torch.linear",
    "torch.sdpa",
    "torch.fp8_gemm",
    "cuda.int4_matmul",
    "fa3.paged_decode",
    "sparse_attention.fwd",
    "aiter.moe_up_gemm",
]


def _discover_plugins():
    """Plugin names the service actually registers, parsed statically from
    kernel_evaluator/services/plugins/__init__.py (the `for _module in (...)`
    block) + each module's PLUGIN_NAME. Avoids importing heavy deps. Falls back
    to _FALLBACK_PLUGINS on any error."""
    pdir = REPO_ROOT / "kernel_evaluator" / "services" / "plugins"
    try:
        init_text = (pdir / "__init__.py").read_text()
    except OSError:
        return list(_FALLBACK_PLUGINS)
    m = re.search(r"for\s+_module\s+in\s*\(([^)]*)\)", init_text)
    if not m:
        return list(_FALLBACK_PLUGINS)
    names = []
    for mod in (x.strip() for x in m.group(1).split(",")):
        if not mod:
            continue
        try:
            mod_text = (pdir / f"{mod}.py").read_text()
        except OSError:
            continue
        nm = re.search(r'^PLUGIN_NAME\s*=\s*["\']([^"\']+)["\']', mod_text, re.MULTILINE)
        if nm:
            names.append(nm.group(1))
    return names or list(_FALLBACK_PLUGINS)


# Shape templates pre-fill the form; the user can edit the JSON freely.
PLUGINS = _discover_plugins()
TARGETS = ["cuda", "cutedsl", "triton", "hip"]
STARTER_MODES = ["none", "generic", "best-similar", "preset"]
SUGGESTED_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "gpt-5.5-xtra-high",
]
SHAPE_TEMPLATES = {
    "torch.linear": {"m": 4096, "n": 4096, "k": 4096, "dtype": "bf16"},
    "torch.sdpa": {"batch": 1, "qo_heads": 32, "kv_heads": 32, "seq_len": 2048,
                   "head_dim": 128, "dtype": "bf16"},
    "torch.fp8_gemm": {"m": 4096, "n": 4096, "k": 4096, "dtype": "fp8_e4m3"},
    "cuda.int4_matmul": {"m": 4096, "n": 4096, "k": 4096},
    "fa3.paged_decode": {"batch": 1, "qo_heads": 32, "kv_heads": 8, "seq_len": 4096,
                         "head_dim": 128, "page_size": 16, "dtype": "bf16"},
    "sparse_attention.fwd": {"batch": 4, "total_q": 4096, "seq_len": 4096,
                             "num_heads": 32, "head_dim": 128, "dtype": "bf16"},
    "aiter.moe_up_gemm": {"m": 4096, "n": 4096, "k": 4096, "dtype": "bf16"},
    "minimax_sparse.decode_indexer": {"batch": 1, "seq_len": 4096, "num_heads": 32},
    # Keys match make_operation_contract() in mla_decode_fp8.py
    # (requires batch_size/num_heads/page_size/max_sequence_kv).
    "mla.decode_fp8": {"batch_size": 4, "num_heads": 128, "page_size": 64,
                       "max_sequence_kv": 1024, "seq_len_q": 1,
                       "latent_dim": 512, "rope_dim": 64, "dtype": "fp8"},
    # rms_norm plugins use {m, n, dtype} (+ optional epsilon).
    "aiter.rms_norm": {"m": 4096, "n": 4096, "dtype": "bf16"},
    "aiter.add_rms_norm": {"m": 4096, "n": 4096, "dtype": "bf16"},
}


# --------------------------------------------------------------------------- #
# Environment / service helpers
# --------------------------------------------------------------------------- #
def load_env():
    """Build the subprocess env: env.dev defaults overlaid by the real os.environ."""
    env = {}
    envfile = REPO_ROOT / "env.dev"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            env[key.strip()] = val.strip().strip('"').strip("'")
    merged = {**env, **os.environ}  # real environment wins over env.dev defaults
    return merged


def service_url(env):
    api = env.get("KERNEL_EVALUATOR_API")
    if api:
        return api.rstrip("/")
    port = env.get("KERNEL_EVALUATOR_PORT", "8000")
    return f"http://localhost:{port}"


def admin_key(env):
    return env.get("KERNEL_EVALUATOR_ADMIN_API_KEY") or env.get("KERNEL_EVALUATOR_API_KEY") or ""


def api_get(env, path, timeout=10):
    """GET {service}{path} with admin key. Returns (status, json_or_none)."""
    url = service_url(env) + path
    req = urllib.request.Request(url, method="GET")
    key = admin_key(env)
    if key:
        req.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, None


def is_up(env, timeout=4):
    status, _ = api_get(env, "/openapi.json", timeout=timeout)
    return status == 200


def ensure_up(env, build=False, wait_s=180):
    """Health-check the service; if down, `docker compose up -d` and wait until healthy."""
    if is_up(env):
        return {"up": True, "started": False, "url": service_url(env)}

    compose = ["docker", "compose", "up", "-d"]
    if build:
        compose.append("--build")
    # Stream the (potentially very long) docker build + bring-up to stderr so the
    # caller (RUD) can tee it into a per-run log and show progress live in the UI.
    # The final result JSON is printed to stdout by main(), keeping stdout clean.
    print(f"[rud_kernel] $ {' '.join(compose)}  (cwd={REPO_ROOT})", file=sys.stderr, flush=True)
    proc = subprocess.run(compose, cwd=str(REPO_ROOT), env=env,
                          stdout=sys.stderr, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return {"up": False, "started": False, "url": service_url(env),
                "error": "docker compose up failed (see build log)"}

    print(f"[rud_kernel] image/containers up; waiting for service health at {service_url(env)} ...",
          file=sys.stderr, flush=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if is_up(env):
            print("[rud_kernel] eval service healthy.", file=sys.stderr, flush=True)
            return {"up": True, "started": True, "url": service_url(env)}
        time.sleep(3)
    return {"up": False, "started": True, "url": service_url(env),
            "error": f"service did not become healthy within {wait_s}s"}


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_plugins(args, env):
    return {
        "ok": True,
        "plugins": PLUGINS,
        "targets": TARGETS,
        "starter_modes": STARTER_MODES,
        "suggested_models": SUGGESTED_MODELS,
        "shape_templates": SHAPE_TEMPLATES,
    }


def cmd_service_status(args, env):
    up = is_up(env)
    return {"ok": True, "up": up, "url": service_url(env)}


def cmd_up(args, env):
    res = ensure_up(env, build=args.build)
    res["ok"] = res.get("up", False)
    return res


_EVAL_CONTAINER = "kernel-evaluator"


def ensure_db_migrated():
    """Idempotently apply alembic migrations inside the eval container so a fresh
    Postgres has its schema (and the seeded admin key). Streams to stderr so the
    output shows up in RUD's build/run log. Best-effort: warns on failure."""
    print("[rud_kernel] ensuring DB schema (alembic upgrade head) ...", file=sys.stderr, flush=True)
    try:
        subprocess.run(
            ["docker", "exec", "-w", "/app/kernel_evaluator", _EVAL_CONTAINER,
             "/opt/venv/bin/python", "-c",
             "from kernel_evaluator.db.session import create_tables; create_tables()"],
            stdout=sys.stderr, stderr=subprocess.STDOUT, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[rud_kernel] DB migrate warning: {exc}", file=sys.stderr, flush=True)


def cmd_launch(args, env):
    # Validate the model has a usable API key in the container env.
    if args.model.startswith("claude-") and not env.get("ANTHROPIC_API_KEY"):
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set; claude agents cannot start"}
    if (args.model.startswith("gpt-") or re.match(r"^o[0-9]", args.model)) and not env.get("OPENAI_API_KEY"):
        return {"ok": False, "error": "OPENAI_API_KEY is not set; codex agents cannot start"}

    # Resolve the shape. Callers (the RUD web UI / its launch agent) normally
    # pass a chosen --shape; when none is given we fall back to the plugin's
    # default SHAPE_TEMPLATES entry so the shape is never a hard human input.
    if args.shape is None or str(args.shape).strip() == "":
        tpl = SHAPE_TEMPLATES.get(args.plugin)
        if tpl is None:
            return {
                "ok": False,
                "error": (
                    f"no --shape given and no default shape template for plugin "
                    f"'{args.plugin}'; pass --shape with a JSON shape"
                ),
            }
        shape_str = json.dumps(tpl)
    else:
        try:
            json.loads(args.shape)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"--shape must be valid JSON: {e}"}
        shape_str = args.shape

    up = ensure_up(env, build=args.build)
    if not up.get("up"):
        return {"ok": False, "error": "eval service is not available", "service": up}

    # A fresh DB has no schema; run migrations (idempotent: alembic upgrade head
    # is a no-op once at head) so run-creation doesn't 500 on a missing table.
    ensure_db_migrated()

    cmd = ["bash", str(RUN_AGENTS),
           "--plugin", args.plugin,
           "--target", args.target,
           "--shape", shape_str,
           "--model", args.model,
           "--n-agents", str(args.n_agents),
           "--starter-mode", args.starter_mode]
    if args.target_speedup is not None:
        cmd += ["--target-speedup", str(args.target_speedup)]
    if args.auto_terminate:
        cmd += ["--auto-terminate", "--poll-interval", str(args.poll_interval)]
    if args.preset_path:
        cmd += ["--preset-path", args.preset_path]
    if args.build_mode:
        cmd += ["--build-mode"]

    # Stream run_agents.sh (incl. the agent docker build + launch) to stderr live
    # so RUD's run log scrolls, while still capturing stdout to parse the run id.
    proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    if proc.stdout is not None:
        for line in proc.stdout:
            sys.stderr.write(line)
            sys.stderr.flush()
            out_lines.append(line)
    proc.wait()
    out = "".join(out_lines)
    if proc.returncode != 0:
        return {"ok": False, "error": "run_agents.sh failed",
                "stderr": out[-2000:]}

    # run_agents.sh prints:  Run: <run_id>  task: <task_slug>  agents: N  model: ...
    m = re.search(r"^Run:\s+(\S+)\s+task:\s+(\S+)\s+agents:", out, re.MULTILINE)
    if not m:
        return {"ok": False, "error": "could not parse run_id from run_agents.sh output",
                "stdout_tail": out[-2000:]}
    run_id, task_slug = m.group(1), m.group(2)
    containers = [f"kernel-agent-{run_id}-{i}" for i in range(1, args.n_agents + 1)]
    return {
        "ok": True,
        "run_id": run_id,
        "task_slug": task_slug,
        "plugin": args.plugin,
        "target": args.target,
        "shape": shape_str,
        "model": args.model,
        "n_agents": args.n_agents,
        "starter_mode": args.starter_mode,
        "target_speedup": args.target_speedup,
        "auto_terminate": args.auto_terminate,
        "build_mode": args.build_mode,
        "containers": containers,
    }


def _docker_running(name_prefix):
    """Return {container_name: running_bool} for containers matching the prefix."""
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={name_prefix}",
         "--format", "{{.Names}}\t{{.State}}"],
        capture_output=True, text=True).stdout
    result = {}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        name, state = line.split("\t", 1)
        result[name.strip()] = state.strip() == "running"
    return result


def cmd_status(args, env):
    run_id = args.run_id
    qs = "?run_id=" + urllib.request.quote(run_id)

    agents_state = _docker_running(f"kernel-agent-{run_id}-")
    agents = []
    for name, running in sorted(agents_state.items()):
        idx = name.rsplit("-", 1)[-1]
        agents.append({"name": name, "index": idx, "running": running})

    _, run_detail = api_get(env, f"/evaluation/runs/{urllib.request.quote(run_id)}")
    target_speedup = (run_detail or {}).get("target_speedup")

    best_status, best = api_get(env, "/scaffold/best" + qs)
    if best_status != 200:
        best = None

    _, agent_bests = api_get(env, "/scaffold/agent-bests" + qs)
    _, archive = api_get(env, "/scaffold/archive" + qs)

    return {
        "ok": True,
        "run_id": run_id,
        "target_speedup": target_speedup,
        "agents": agents,
        "agents_running": sum(1 for a in agents if a["running"]),
        "best": best,
        "agent_bests": (agent_bests or {}).get("agent_bests", []),
        "archive": (archive or {}).get("entries", []),
        "improvements": len((archive or {}).get("entries", [])),
    }


def api_get_text(env, path, timeout=10):
    """GET {service}{path} with admin key. Returns (status, text)."""
    url = service_url(env) + path
    req = urllib.request.Request(url, method="GET")
    key = admin_key(env)
    if key:
        req.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, ""


def cmd_kernel_source(args, env):
    status, text = api_get_text(
        env, f"/scaffold/kernel-source/{urllib.request.quote(args.job_id)}"
    )
    if status != 200:
        return {"ok": False, "error": f"kernel source not found (status {status})",
                "job_id": args.job_id}
    return {"ok": True, "job_id": args.job_id, "source": text}


def cmd_best_kernel(args, env):
    status, data = api_get(
        env, f"/evaluation/runs/{urllib.request.quote(args.run_id)}/best-kernel"
    )
    if status != 200 or not isinstance(data, dict):
        return {"ok": False, "error": f"no best kernel yet (status {status})",
                "run_id": args.run_id}
    data["ok"] = True
    return data


def cmd_stop(args, env):
    proc = subprocess.run(["bash", str(FINISH_RUN), args.run_id],
                          cwd=str(SCRIPT_DIR), env=env,
                          capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "run_id": args.run_id,
        "output": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-1000:] if proc.returncode != 0 else "",
    }


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="Loom kernel-run helper (JSON output)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("plugins")
    sub.add_parser("service-status")

    up = sub.add_parser("up")
    up.add_argument("--build", action="store_true")

    lr = sub.add_parser("launch")
    lr.add_argument("--plugin", required=True, choices=PLUGINS)
    lr.add_argument("--target", required=True, choices=TARGETS)
    lr.add_argument(
        "--shape",
        required=False,
        default=None,
        help="benchmark shape as JSON. Optional: when omitted, the plugin's "
             "default SHAPE_TEMPLATES entry is used (the caller/agent normally "
             "supplies a chosen shape instead).",
    )
    lr.add_argument("--model", required=True)
    lr.add_argument("--n-agents", type=int, default=1, dest="n_agents")
    lr.add_argument("--starter-mode", default="none", choices=STARTER_MODES, dest="starter_mode")
    lr.add_argument("--target-speedup", type=float, default=None, dest="target_speedup")
    lr.add_argument("--auto-terminate", action="store_true", dest="auto_terminate")
    lr.add_argument("--poll-interval", type=int, default=60, dest="poll_interval")
    lr.add_argument("--preset-path", default="", dest="preset_path")
    lr.add_argument("--build", action="store_true")
    lr.add_argument("--build-mode", action="store_true", dest="build_mode")

    st = sub.add_parser("status")
    st.add_argument("--run-id", required=True, dest="run_id")

    sp = sub.add_parser("stop")
    sp.add_argument("--run-id", required=True, dest="run_id")

    ks = sub.add_parser("kernel-source")
    ks.add_argument("--job-id", required=True, dest="job_id")

    bk = sub.add_parser("best-kernel")
    bk.add_argument("--run-id", required=True, dest="run_id")

    return p


HANDLERS = {
    "plugins": cmd_plugins,
    "service-status": cmd_service_status,
    "up": cmd_up,
    "launch": cmd_launch,
    "status": cmd_status,
    "stop": cmd_stop,
    "kernel-source": cmd_kernel_source,
    "best-kernel": cmd_best_kernel,
}


def main():
    args = build_parser().parse_args()
    env = load_env()
    try:
        result = HANDLERS[args.command](args, env)
    except Exception as e:  # never leak a traceback into RUD's JSON parser
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
