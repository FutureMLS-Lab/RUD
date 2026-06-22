#!/usr/bin/env python3
"""Fetch kernels from the evaluation service and save to replacements/experimental.

Usage:
    # By spec (JSON dict or file path):
    python fetch_kernel.py --spec '{"function_name": "mm_abt", "m": 4096, ...}'
    python fetch_kernel.py --spec spec.json

    # By run_id (best kernel from that run):
    python fetch_kernel.py --run-id my-run-123

    # By kernel id (specific kernel):
    python fetch_kernel.py --kernel-id 42

Environment:
    KERNEL_EVALUATOR_PORT: API port (default: 8000)
    KERNEL_EVALUATOR_ADMIN_API_KEY: API key for authentication
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _get_api_url() -> str:
    port = os.environ.get("KERNEL_EVALUATOR_PORT", "8000")
    return os.environ.get("KERNEL_EVALUATOR_API", f"http://localhost:{port}")


def _get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("KERNEL_EVALUATOR_ADMIN_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _api_get(path: str) -> dict:
    url = f"{_get_api_url()}{path}"
    req = urllib.request.Request(url, headers=_get_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def build_filename(data: dict) -> str:
    """Build a descriptive filename from API response."""
    func = data["function_name"]
    spec = data.get("spec_json") or data.get("scalar_args") or {}

    parts = [func]
    if spec:
        for key in sorted(spec.keys()):
            val = spec[key]
            if isinstance(val, (int, float)) and key != "function_name":
                parts.append(str(int(val) if isinstance(val, float) and val.is_integer() else val))

    speedup = data["speedup"]
    parts.append(f"{speedup:.2f}x")
    parts.append(f"id{data['id']}")

    return "_".join(parts) + ".cu"


def fetch_by_spec(spec_json: dict, gpu: str) -> None:
    spec_str = urllib.parse.quote(json.dumps(spec_json))
    path = f"/kernels/best?spec_json={spec_str}&gpu={gpu}"
    data = _api_get(path)
    if not data:
        print(f"No kernel found for spec: {spec_json}")
        sys.exit(1)
    save_kernel(data)


def fetch_by_run_id(run_id: str) -> None:
    path = f"/evaluation/runs/{run_id}/best-kernel"
    data = _api_get(path)
    if not data:
        print(f"No kernel found for run_id: {run_id}")
        sys.exit(1)
    save_kernel(data)


def fetch_by_kernel_id(kernel_id: int) -> None:
    path = f"/kernels/{kernel_id}"
    data = _api_get(path)
    if not data:
        print(f"No kernel found with id: {kernel_id}")
        sys.exit(1)
    save_kernel(data)


def save_kernel(data: dict) -> None:
    filename = build_filename(data)
    output_dir = REPO_ROOT / "replacements" / "experimental"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    source = data.get("postprocessed_source") or data["kernel_source"]

    with open(output_path, "w") as f:
        f.write(source)

    print(f"Saved: {output_path}")
    print(f"  Kernel ID: {data['id']}")
    print(f"  Speedup: {data['speedup']:.3f}x ({data['baseline_us']:.2f}us -> {data['kernel_us']:.2f}us)")
    print(f"  Run ID: {data['run_id']}")
    print(f"  Function: {data['function_name']}")


def main():
    parser = argparse.ArgumentParser(description="Fetch kernels from API to experimental/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", help="Spec JSON (inline or path to .json file)")
    group.add_argument("--run-id", help="Get best kernel from this run")
    group.add_argument("--kernel-id", type=int, help="Get specific kernel by ID")
    parser.add_argument("--gpu", default="h100", help="GPU type (default: h100)")

    args = parser.parse_args()

    if args.spec:
        if os.path.isfile(args.spec):
            with open(args.spec) as f:
                spec_json = json.load(f)
        else:
            spec_json = json.loads(args.spec)
        fetch_by_spec(spec_json, args.gpu)
    elif args.run_id:
        fetch_by_run_id(args.run_id)
    elif args.kernel_id:
        fetch_by_kernel_id(args.kernel_id)


if __name__ == "__main__":
    main()
