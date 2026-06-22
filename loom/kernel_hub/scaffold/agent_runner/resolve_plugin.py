#!/usr/bin/env python3
"""Agentic "kernel -> plugin" resolver for the kernel hub.

Given a kernel (a GitHub raw URL or a name), drive the logged-in `claude` CLI on
the HOST to:
  1. read the kernel source + the local plugin registry,
  2. decide: reuse an existing plugin, or create a new one,
  3. if new, WRITE the plugin module + register it in the registry.

The generated reference() is left with a mandatory human-review TODO, and the
eval image is NOT rebuilt — review the reference first (it defines correctness
for every future optimization of that op).

Auth: uses the host claude CLI's logged-in session (~/.claude/.credentials.json)
— no ANTHROPIC_API_KEY required. Must run on the host (not inside a container).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PLUGINS_DIR = REPO_ROOT / "kernel_evaluator" / "services" / "plugins"

PROMPT_TMPL = """You are resolving which kernel-eval "plugin" a GPU kernel maps to, and creating one if needed. Work inside the repo at {repo}.

KERNEL TO CLASSIFY: {src_desc}

A "plugin" is the math/IO contract of an operation (NOT the kernel): inputs (tensors+shapes+dtypes), output, the reference math, and the dtype/quantization. Two ops are the same plugin only if math + IO layout + dtype all match.

STEPS:
1. Read the kernel {src_read}. Determine the OPERATION it computes: input tensors (names/shapes/dtypes), output, the math, and any fp8/int4 quantization. Ignore implementation details (tiling/warps/TMA/target arch).
2. Read the local plugin registry to learn the available plugins + the interface:
   - {plugins_dir}/__init__.py  (KernelEvalPlugin, ReferencePlugin, register_plugin, and the `for _module in (...)` registration block)
   - {plugins_dir}/spec_helpers.py  (scalar_values, DEFAULT_TOLERANCES, TORCH_DTYPES)
   - each {plugins_dir}/*.py defines PLUGIN_NAME + make_reference_plugin(make_inputs+reference+tolerances) + a contract factory + PLUGIN = KernelEvalPlugin(...). Use linear.py and sdpa.py as templates.
3. DECIDE — does this kernel's operation EXACTLY match an existing plugin (same math + IO + dtype)?
   - REUSE: if yes, change NO code. Print exactly: `RESULT: REUSE <plugin_name>` + one line why.
   - CREATE: if no existing plugin matches, create a new one:
{create_instructions}
       c. CRITICAL (review gate): put this EXACT comment on its own line immediately inside reference(), before any compute:
          `# TODO(review): verify this reference is mathematically correct before trusting any benchmark results.`
       d. If the kernel is hardware-specific (e.g. Blackwell SM100 / fp8), STILL write reference() in plain hardware-agnostic PyTorch, and add a top-of-file comment noting the intended target (cuda/cutedsl/triton) + precision + that benchmarking needs that hardware.
       e. Print exactly: `RESULT: CREATE <plugin_name> at <path relative to repo>` + a 2-3 line summary of the contract (inputs/dtype/math).
4. Do NOT run docker or rebuild anything. {scope} Be exact about tensor layouts and dtypes; where the math is uncertain, write your best version and make the uncertainty explicit in comments next to the TODO."""

_REF_NOTE = """          IMPORTANT — prefer the source's OWN reference: if the kernel source itself contains a PyTorch reference implementation (e.g. a function named `torch_reference_*` / `ref_*`, or an explicit math + tolerance in comments), PORT THAT VERBATIM into reference()/tolerances (adapt only the IO to the ExecutionInputs tensors/scalars). It is the authoritative ground truth — do NOT re-derive the math when the source already provides it."""

# Default: write into the kernel_evaluator package registry + edit __init__.py (needs a rebuild).
_CREATE_REGISTRY = """       a. Write {plugins_dir}/<slug>.py (slug from the op, e.g. mla_decode_fp8.py) modeled on the existing modules: PLUGIN_NAME (e.g. "mla.decode_fp8"), make_reference_plugin(dtype, spec) with make_inputs() (random inputs, correct shapes/dtypes) and reference() (the correct math in PLAIN PyTorch), tolerances, a contract factory (task_slug/spec/scalars/dtype), and PLUGIN = KernelEvalPlugin(...).
{ref_note}
       b. Register it: edit {plugins_dir}/__init__.py — add the module to BOTH the `from kernel_evaluator.services.plugins import ...` line AND the `for _module in (...)` tuple."""

# --out-dir: write a self-contained module into the work dir; the eval service
# auto-loads it from KERNEL_EVALUATOR_EXTRA_PLUGINS_PATH — do NOT edit the package.
_CREATE_OUTDIR = """       a. Write a SELF-CONTAINED plugin module {out_dir}/<slug>.py (slug from the op, e.g. mla_decode_fp8.py), modeled on the existing modules in {plugins_dir} (read linear.py / sdpa.py as templates): PLUGIN_NAME, make_reference_plugin(dtype, spec) with make_inputs() and reference() (correct math in PLAIN PyTorch), tolerances, a contract factory (task_slug/spec/scalars/dtype), and PLUGIN = KernelEvalPlugin(...). Import the framework normally from `kernel_evaluator.services.plugins` and `kernel_evaluator.services.evaluation.types` (the kernel_evaluator package is pip-installed).
       b. Do NOT edit __init__.py and do NOT touch the kernel_evaluator package — the eval service auto-loads this module from KERNEL_EVALUATOR_EXTRA_PLUGINS_PATH.
{ref_note}"""


def fetch_source(source: str):
    if source.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(source, timeout=30) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"error: could not fetch {source}: {e}", file=sys.stderr)
            sys.exit(1)
        tmp = Path(tempfile.mkdtemp()) / "kernel_source.txt"
        tmp.write_text(text)
        return f"{source} (fetched to {tmp})", f"at the local path {tmp}"
    return f'name "{source}" (find its source via WebSearch/WebFetch)', "by searching the web for its source code"


def _plugin_mtimes(directory: Path):
    state = {}
    if directory.is_dir():
        for f in directory.glob("*.py"):
            try:
                state[f.name] = f.stat().st_mtime
            except OSError:
                pass
    return state


def main():
    ap = argparse.ArgumentParser(description="Resolve a kernel to a kernel-eval plugin (reuse or create) via the logged-in claude CLI")
    ap.add_argument("--source", required=True, help="GitHub raw URL or a kernel name")
    ap.add_argument("--model", default="", help="optional claude model override")
    ap.add_argument("--out-dir", default="", dest="out_dir",
                    help="write a self-contained plugin here (e.g. the task work dir) instead of editing the "
                         "package registry; the eval service auto-loads it via KERNEL_EVALUATOR_EXTRA_PLUGINS_PATH")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true", help="fetch + print the prompt, do not invoke claude")
    args = ap.parse_args()

    if args.out_dir:
        watch_dir = Path(args.out_dir).expanduser().resolve()
        watch_dir.mkdir(parents=True, exist_ok=True)
        create_instructions = _CREATE_OUTDIR.format(out_dir=watch_dir, plugins_dir=PLUGINS_DIR, ref_note=_REF_NOTE)
        scope = f"Do NOT touch files outside {watch_dir} (do not edit the kernel_evaluator package itself)."
    else:
        watch_dir = PLUGINS_DIR
        create_instructions = _CREATE_REGISTRY.format(plugins_dir=PLUGINS_DIR, ref_note=_REF_NOTE)
        scope = f"Do NOT touch files outside {PLUGINS_DIR}."

    src_desc, src_read = fetch_source(args.source)
    prompt = PROMPT_TMPL.format(repo=REPO_ROOT, plugins_dir=PLUGINS_DIR, src_desc=src_desc, src_read=src_read,
                                create_instructions=create_instructions, scope=scope)

    if args.dry_run:
        print("=== DRY RUN: prompt that would be sent to `claude -p` ===")
        print(prompt)
        return 0

    if not (Path.home() / ".claude" / ".credentials.json").exists() and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: no host claude login (~/.claude/.credentials.json) and no ANTHROPIC_API_KEY. Run `claude login`.", file=sys.stderr)
        return 1

    cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
    if args.model:
        cmd += ["--model", args.model]

    print(f"[resolve-plugin] invoking host claude (logged-in session) in {REPO_ROOT} ...", file=sys.stderr)
    before = _plugin_mtimes(watch_dir)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=args.timeout)
    after = _plugin_mtimes(watch_dir)

    new_files = sorted(set(after) - set(before))
    changed_init = (not args.out_dir) and before.get("__init__.py") != after.get("__init__.py")
    print("\n" + "=" * 64, file=sys.stderr)
    if new_files or changed_init:
        print("⚠ REVIEW REQUIRED — a plugin was created:", file=sys.stderr)
        for f in new_files:
            print(f"   new module: {watch_dir}/{f}", file=sys.stderr)
        if changed_init:
            print("   __init__.py registration changed", file=sys.stderr)
        print("   → review the reference() math (search for TODO(review)) before trusting results.", file=sys.stderr)
        if args.out_dir:
            print(f"   → the eval service auto-loads it when KERNEL_EVALUATOR_EXTRA_PLUGINS_PATH includes {watch_dir} (no rebuild).", file=sys.stderr)
        else:
            print("   → then rebuild the eval image: `docker compose build`.", file=sys.stderr)
    else:
        print("No plugin files changed (reuse, or no change made).", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
