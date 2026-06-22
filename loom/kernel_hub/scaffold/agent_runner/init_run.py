#!/usr/bin/env python3
"""Create an evaluation run via the kernel_evaluator API."""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

_PORT = os.environ.get("KERNEL_EVALUATOR_PORT", "8000")
SERVICE = os.environ.get("KERNEL_EVALUATOR_API", f"http://localhost:{_PORT}")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Fallback only; the real source of truth is the plugin registry parsed below.
_FALLBACK_PLUGINS = [
    "aiter.moe_up_gemm",
    "torch.linear",
    "torch.sdpa",
    "torch.fp8_gemm",
    "cuda.int4_matmul",
    "fa3.paged_decode",
    "sparse_attention.fwd",
]


def _discover_plugins():
    """Plugin names the service actually registers, parsed statically from
    kernel_evaluator/services/plugins/__init__.py (the `for _module in (...)`
    block) + each module's PLUGIN_NAME. Keeps this CLI's --plugin choices in
    sync with the registry instead of a hand-maintained list that drifts."""
    pdir = _REPO_ROOT / "kernel_evaluator" / "services" / "plugins"
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


AVAILABLE_PLUGINS = _discover_plugins()

AVAILABLE_TARGETS = ["cuda", "cutedsl", "hip", "triton"]


def _default_api_key():
    return (
        os.environ.get("KERNEL_EVALUATOR_ADMIN_API_KEY")
        or os.environ.get("KERNEL_EVALUATOR_API_KEY")
        or ""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create an evaluation run for kernel optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available plugins: {', '.join(AVAILABLE_PLUGINS)}
Available targets: {', '.join(AVAILABLE_TARGETS)}

Examples:
  # GEMM kernel
  python init_run.py --plugin torch.linear --target cuda \\
    --shape '{{"m": 1, "n": 4096, "k": 4096, "dtype": "float16"}}'

  # Attention kernel
  python init_run.py --plugin torch.sdpa --target cuda \\
    --shape '{{"batch": 1, "qo_heads": 32, "kv_heads": 32, "seq_len": 2048, "head_dim": 128, "dtype": "bf16"}}'
""",
    )
    parser.add_argument(
        "--plugin",
        required=True,
        choices=AVAILABLE_PLUGINS,
        help="Plugin to use for the evaluation run",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=AVAILABLE_TARGETS,
        help="Target compilation mode",
    )
    parser.add_argument(
        "--shape",
        required=True,
        help="Shape parameters as JSON (plugin-specific)",
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        default=None,
        help="Target speedup threshold for automatic completion",
    )
    parser.add_argument(
        "--api-key",
        default=_default_api_key(),
        help="Admin API key (or set KERNEL_EVALUATOR_ADMIN_API_KEY / KERNEL_EVALUATOR_API_KEY)",
    )
    args = parser.parse_args()

    try:
        shape = json.loads(args.shape)
    except json.JSONDecodeError as e:
        print(f"Error: --shape must be valid JSON: {e}", file=sys.stderr)
        return 1

    payload = {
        "plugin": args.plugin,
        "target": args.target,
        "shapes": [shape],
    }
    if args.target_speedup is not None:
        payload["target_speedup"] = args.target_speedup

    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    req = urllib.request.Request(
        f"{SERVICE}/evaluation/runs",
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.load(resp)
            run_id = result.get("run_id", "")
            task_slug = ""
            benchmark_shapes = result.get("benchmark_shapes", [])
            if benchmark_shapes:
                task_slug = benchmark_shapes[0].get("task_slug", "")
            print(run_id)
            print(task_slug)
            return 0
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"Error creating run: {e.code} {e.reason}", file=sys.stderr)
        if error_body:
            print(error_body, file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
