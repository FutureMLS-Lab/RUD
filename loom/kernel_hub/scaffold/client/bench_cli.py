import argparse
import os
import sys
import time
import urllib.parse
from pathlib import Path

from scaffold.client.api import EvaluatorClient, EvaluatorError

_RUN_ID = os.environ.get("BENCH_RUN_ID", "")
_AGENT_INDEX = os.environ.get("BENCH_AGENT_INDEX", "")
_STARTER_MODE = os.environ.get("BENCH_STARTER_MODE", "none")
_PRESET_PATH = os.environ.get("BENCH_PRESET_PATH", "")

_CLIENT: EvaluatorClient | None = None


def _client() -> EvaluatorClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = EvaluatorClient.from_env()
    return _CLIENT


def _qs(**kwargs) -> str:
    params = {k: v for k, v in kwargs.items() if v}
    return ("?" + urllib.parse.urlencode(params)) if params else ""


_GENERIC_STARTERS = {
    "torch.linear": "pkg:generic_starter_kernels/mm_tk_abt.cu",
    "torch.sdpa": "pkg:generic_starter_kernels/sdpa_tk_mha.cu",
    "sparse_attention.fwd": "pkg:generic_starter_kernels/sparse_attention_fwd.py",
    "aiter.rms_norm": "pkg:generic_starter_kernels/rms_norm_hip.cpp",
    "aiter.add_rms_norm": "pkg:generic_starter_kernels/add_rms_norm_hip.cpp",
}


def _starter_root() -> Path:
    import importlib.util
    spec = importlib.util.find_spec("kernel_evaluator")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("kernel_evaluator package not found")
    return Path(spec.submodule_search_locations[0])


def _fetch_starter(run_id: str, plugin: str, out_file: str) -> None:
    mode = _STARTER_MODE
    if not mode or mode == "none":
        return

    print()
    if mode == "best-similar":
        try:
            r = _client().get_json(f"/scaffold/starter{_qs(run_id=run_id)}")
        except EvaluatorError as e:
            if e.status == 404:
                print("Starter:     best-similar (no match found, write from scratch)")
                return
            raise
        with open(out_file, "w") as f:
            f.write(r["source"])
        match_info = f"{r['match_type']} match"
        if r["match_type"] == "exact":
            match_info += f" (function={r['matched_function']}, scalars={r['matched_scalars']})"
        else:
            match_info += f" (function={r['matched_function']} only, scalars may differ)"
        print(f"Starter:     {match_info}, {r['speedup']:.2f}x -> {out_file}")

    elif mode == "generic":
        template_path = _GENERIC_STARTERS.get(plugin)
        if not template_path:
            print(f"Starter:     generic (no template for plugin '{plugin}')")
            return
        if template_path.startswith("pkg:"):
            full_path = _starter_root() / template_path[4:]
        else:
            full_path = _starter_root().parent / template_path
        if not full_path.exists():
            print(f"Starter:     generic (template not found: {template_path})")
            return
        with open(full_path) as f:
            source = f.read()
        with open(out_file, "w") as f:
            f.write(source)
        print(f"Starter:     generic template ({template_path}) -> {out_file}")

    elif mode == "preset":
        if not _PRESET_PATH:
            print("Starter:     preset (ERROR: BENCH_PRESET_PATH not set)")
            return
        preset_file = Path(_PRESET_PATH)
        if not preset_file.is_file():
            print(f"Starter:     preset (ERROR: {_PRESET_PATH} not found)")
            return
        with open(preset_file) as f:
            source = f.read()
        with open(out_file, "w") as f:
            f.write(source)
        print(f"Starter:     preset ({_PRESET_PATH}) -> {out_file}")


def cmd_run():
    parser = argparse.ArgumentParser(prog="bench-run", description="Get info about the current evaluation run")
    parser.add_argument("--run-id", default=_RUN_ID)
    parser.add_argument("--starter-out", default=None, help="Starter code output file (default based on target)")
    args = parser.parse_args()

    if not args.run_id:
        print("error: --run-id or BENCH_RUN_ID required", file=sys.stderr)
        raise SystemExit(1)

    try:
        r = _client().get_json(f"/evaluation/runs/{args.run_id}")
    except EvaluatorError as e:
        if e.status == 404:
            print(f"run not found: {args.run_id}", file=sys.stderr)
            raise SystemExit(1)
        raise

    print(f"run_id:      {r['run_id']}")
    print(f"plugin:      {r['plugin']}")
    print(f"target:      {r['target']}")
    print()
    benchmark_shapes = r.get("benchmark_shapes", [])
    print(f"Shapes ({len(benchmark_shapes)}):")
    for bs in benchmark_shapes:
        shape = bs.get("shape", {})
        dtype = bs.get("dtype", "unknown")
        shape_str = ", ".join(f"{k}={v}" for k, v in shape.items())
        print(f"  {bs['task_slug']}: {shape_str} - dtype: {dtype}")
    print()
    print(f"Instructions:\n{r.get('instructions', 'N/A')}")

    starter_out = args.starter_out
    if not starter_out:
        target = r.get("target", "cuda")
        starter_out = "kernel.py" if target in ("cutedsl", "triton") else "kernel.cu"
    _fetch_starter(args.run_id, r.get("plugin", ""), starter_out)


def cmd_submit():
    parser = argparse.ArgumentParser(prog="bench-submit", description="Submit a kernel for benchmarking")
    parser.add_argument("file", help="Path to kernel source file")
    parser.add_argument("--run-id", default=_RUN_ID)
    parser.add_argument("--agent-index", type=int, default=int(_AGENT_INDEX) if _AGENT_INDEX else None)
    parser.add_argument("--artifacts", nargs="*", default=[], help="Artifacts to request (e.g., cuobjdump ptx)")
    args = parser.parse_args()

    if not args.run_id:
        print("error: --run-id or BENCH_RUN_ID required", file=sys.stderr)
        raise SystemExit(1)

    source = open(args.file).read()
    try:
        resp = _client().submit(
            args.run_id, source,
            artifacts=tuple(args.artifacts), agent_index=args.agent_index,
        )
    except EvaluatorError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"job_id={resp['job_id']}  state={resp['state']}")
    print(f"run: bench-poll {resp['job_id']}")


def _print_result(r: dict) -> None:
    baseline = r.get("baseline_us", 0)
    kernel = r.get("candidate_us", 0)
    correct = r.get("correct", False)
    speedup = baseline / kernel if kernel > 0 else 0
    print(f"correct={correct}  baseline={baseline:.1f}us  kernel={kernel:.1f}us  speedup={speedup:.2f}x")


def cmd_poll():
    parser = argparse.ArgumentParser(prog="bench-poll", description="Poll for benchmark result")
    parser.add_argument("job_id")
    args = parser.parse_args()

    last_state = None
    while True:
        try:
            r = _client().job(args.job_id)
        except EvaluatorError as e:
            if e.status == 404:
                print(f"job not found: {args.job_id}", file=sys.stderr)
                raise SystemExit(1)
            raise

        state = r.get("state", "unknown")
        if r.get("compile_error"):
            print(f"COMPILE ERROR:\n{r['compile_error']}")
            raise SystemExit(1)
        if r.get("benchmark_error"):
            print(f"BENCHMARK ERROR:\n{r['benchmark_error']}")
            raise SystemExit(1)
        if state == "completed":
            _print_result(r)
            if r.get("artifacts"):
                print(f"artifacts={','.join(r['artifacts'])}")
            break
        if state != last_state:
            print(f"state={state}")
            last_state = state
        time.sleep(1)


def cmd_result():
    parser = argparse.ArgumentParser(prog="bench-result", description="Get job result without polling")
    parser.add_argument("job_id")
    args = parser.parse_args()

    try:
        _print_result(_client().job_result(args.job_id))
    except EvaluatorError as e:
        if e.status == 409:
            print("job not complete")
            raise SystemExit(1)
        if e.status == 404:
            print(f"job not found: {args.job_id}", file=sys.stderr)
            raise SystemExit(1)
        raise


def cmd_best():
    parser = argparse.ArgumentParser(prog="bench-best", description="Fetch the run leaderboard best")
    parser.add_argument("--run-id", default=_RUN_ID)
    parser.add_argument("--out", metavar="FILE", default="leaderboard_best.cu", help="Save kernel source to FILE")
    args = parser.parse_args()

    if not args.run_id:
        print("error: --run-id or BENCH_RUN_ID required", file=sys.stderr)
        raise SystemExit(1)

    try:
        r = _client().get_json(f"/scaffold/best{_qs(run_id=args.run_id)}")
    except EvaluatorError as e:
        if e.status == 404:
            print("no best kernel yet — leaderboard is empty")
            return
        raise
    speedup = r["baseline_us"] / r["kernel_us"] if r["kernel_us"] > 0 else 0
    print(f"leaderboard best: {r['kernel_us']:.1f}us ({speedup:.2f}x)  job={r['job_id']}")
    with open(args.out, "w") as f:
        f.write(r["source"])
    print(f"saved to {args.out}")


def cmd_archive():
    parser = argparse.ArgumentParser(prog="bench-archive", description="List run improvement history")
    parser.add_argument("--run-id", default=_RUN_ID)
    args = parser.parse_args()

    if not args.run_id:
        print("error: --run-id or BENCH_RUN_ID required", file=sys.stderr)
        raise SystemExit(1)

    r = _client().get_json(f"/scaffold/archive{_qs(run_id=args.run_id)}")
    print(f"Archive for {args.run_id}: {len(r['entries'])} improvements")
    for e in r["entries"]:
        agent = e.get("agent_index")
        agent_str = str(agent) if agent is not None else "-"
        speedup = e["baseline_us"] / e["kernel_us"] if e["kernel_us"] > 0 else 0
        print(f"  {e['job_id']}  {e['kernel_us']:.1f}us  {speedup:.3f}x  agent={agent_str}  at={e['achieved_at']}")


def cmd_agent_bests():
    parser = argparse.ArgumentParser(prog="bench-agent-bests", description="Show per-agent best leaderboard")
    parser.add_argument("--run-id", default=_RUN_ID)
    args = parser.parse_args()

    if not args.run_id:
        print("error: --run-id or BENCH_RUN_ID required", file=sys.stderr)
        raise SystemExit(1)

    r = _client().get_json(f"/scaffold/agent-bests{_qs(run_id=args.run_id)}")
    print(f"Per-agent bests for {args.run_id}: {len(r['agent_bests'])} agents")
    for e in r["agent_bests"]:
        agent = e.get("agent_index")
        agent_str = str(agent) if agent is not None else "-"
        print(f"  agent={agent_str}  {e['kernel_us']:.1f}us  {e['speedup']:.2f}x  job={e['job_id']}")


def cmd_kernel_source():
    parser = argparse.ArgumentParser(prog="bench-kernel-source", description="Fetch kernel source by job_id")
    parser.add_argument("job_id")
    parser.add_argument("--out", metavar="FILE", help="Save source to FILE instead of printing")
    args = parser.parse_args()

    source = _client().get_text(f"/scaffold/kernel-source/{args.job_id}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(source)
        print(f"saved to {args.out}")
    else:
        print(source, end="")


def cmd_artifact():
    parser = argparse.ArgumentParser(prog="bench-artifact", description="Fetch a job artifact")
    parser.add_argument("job_id")
    parser.add_argument("artifact_kind", help="Artifact kind (e.g., cuobjdump, ptx)")
    parser.add_argument("--out", metavar="FILE")
    args = parser.parse_args()

    try:
        data = _client().get_bytes(f"/evaluation/jobs/{args.job_id}/artifacts/{args.artifact_kind}")
    except EvaluatorError as e:
        if e.status == 404:
            print(f"artifact not found: {args.artifact_kind}", file=sys.stderr)
            raise SystemExit(1)
        raise
    if args.out:
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"saved to {args.out}")
    else:
        print(data.decode(), end="")
