import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from scaffold.client.api import EvaluatorClient
from scaffold.client.config import ScaffoldConfig
from scaffold.client.orchestrator import Orchestrator, RunInfo


def _orchestrator(args) -> Orchestrator:
    scaffold_dir = Path(args.scaffold_dir).resolve()
    config = ScaffoldConfig.load(scaffold_dir)
    api = EvaluatorClient.from_env()
    return Orchestrator(config=config, api=api)


def _resolve(value, fallback):
    return fallback if value is None else value


def _parse_shapes(raw: str) -> list[dict]:
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, list) else [parsed]


def _cmd_launch(args) -> int:
    orch = _orchestrator(args)
    rc = orch.config.run

    plugin = _resolve(args.plugin, rc.plugin)
    target = _resolve(args.target, rc.target)
    shapes = _parse_shapes(args.shapes) if args.shapes is not None else rc.shapes
    model = _resolve(args.model, rc.model)
    n_agents = _resolve(args.n_agents, rc.n_agents)
    start_index = _resolve(args.start_index, rc.start_index)
    target_speedup = _resolve(args.target_speedup, rc.target_speedup)
    starter_mode = _resolve(args.starter_mode, rc.starter_mode)
    preset_path = _resolve(args.preset_path, rc.preset_path)
    auto_terminate = _resolve(args.auto_terminate, rc.auto_terminate)
    poll_interval = _resolve(args.poll_interval, rc.poll_interval)
    max_iterations = _resolve(args.max_iterations, rc.max_iterations)

    missing = [name for name, val in
               (("plugin", plugin), ("target", target), ("shapes", shapes), ("model", model)) if not val]
    if missing:
        print(f"error: missing required run config (set in scaffold.yaml or via flags): {', '.join(missing)}",
              file=sys.stderr)
        return 1
    if auto_terminate and target_speedup is None:
        print("error: auto-terminate requires target_speedup", file=sys.stderr)
        return 1

    if not args.no_build:
        orch.build_image()
    run = orch.create_run(plugin, target, shapes, target_speedup)
    print(f"run: {run.run_id}  task: {run.task_slug}  shapes: {len(run.task_slugs)}")
    handles = orch.launch(
        run, model=model, n_agents=n_agents,
        starter_mode=starter_mode, preset_path=preset_path, start_index=start_index,
        max_iterations=max_iterations,
    )
    for h in handles:
        print(f"agent {h.agent_index} -> {h.container_name} ({h.workdir})")
    print(f"logs: {orch.runtime.logs_command(handles[0].container_name)}")

    if auto_terminate:
        run_dir = orch.runs_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "wait_for_speedup.log"
        log = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "scaffold.client.admin_cli", "watch", run.run_id,
             "--interval", str(poll_interval)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        print(f"auto-terminate: watcher pid {proc.pid} (log: {log_path})")
    else:
        print(f"finish: python3 -m scaffold.client.admin_cli finish {run.run_id}")
    return 0


def _cmd_start_agent(args) -> int:
    orch = _orchestrator(args)
    rc = orch.config.run
    if not args.no_build:
        orch.build_image()
    run = RunInfo(run_id=args.run_id, task_slug=args.task_slug,
                  plugin=_resolve(args.plugin, rc.plugin), target=_resolve(args.target, rc.target))
    handle = orch.start_agent(
        run, agent_index=args.agent_index, model=_resolve(args.model, rc.model),
        starter_mode=_resolve(args.starter_mode, rc.starter_mode),
        preset_path=_resolve(args.preset_path, rc.preset_path),
        max_iterations=_resolve(args.max_iterations, rc.max_iterations),
    )
    print(f"agent {handle.agent_index} -> {handle.container_name} ({handle.workdir})")
    print(f"logs: {orch.runtime.logs_command(handle.container_name)}")
    return 0


def _cmd_stop(args) -> int:
    orch = _orchestrator(args)
    orch.stop_agent(args.container_name)
    print(f"stopped {args.container_name}")
    return 0


def _cmd_stop_run(args) -> int:
    orch = _orchestrator(args)
    stopped = orch.stop_run(args.run_id)
    print(f"stopped {len(stopped)} containers: {' '.join(stopped)}")
    return 0


def _cmd_finish(args) -> int:
    orch = _orchestrator(args)
    result = orch.finish_run(args.run_id, postprocess=not args.no_postprocess)
    print(json.dumps(result))
    return 0


_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _fmt_span(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _bar(value: float, maxv: float, width: int = 28) -> str:
    filled = 0 if maxv <= 0 else round(value / maxv * width)
    return "█" * filled + "·" * (width - filled)


def _sparkline(series: list[float]) -> str:
    lo, hi = min(series), max(series)
    rng = hi - lo or 1.0
    out = []
    for v in series:
        ch = _SPARK[min(len(_SPARK) - 1, int((v - lo) / rng * (len(_SPARK) - 1)))]
        out.append(_c(ch, "32") if v >= 1.0 else _c(ch, "36"))
    return "".join(out)


def _cumbest_series(entries: list[dict], times: list[datetime], width: int = 50) -> list[float]:
    t0 = times[0]
    span = (times[-1] - t0).total_seconds() or 1.0
    cum, hi = [], 0.0
    for e, t in zip(entries, times):
        hi = max(hi, e["speedup"])
        cum.append((t, hi))
    series = []
    for i in range(width):
        col_end = t0.timestamp() + span * (i + 1) / width
        val = 0.0
        for t, c in cum:
            if t.timestamp() <= col_end:
                val = c
            else:
                break
        series.append(val)
    return series


def _print_status(orch: Orchestrator, run_id: str) -> None:
    api = orch.api
    run = api.get_run(run_id)
    archive = api.archive(run_id)
    agent_bests = api.agent_bests(run_id)
    best = api.best_kernel(run_id)
    running = orch.runtime.list_by_prefix(f"kernel-agent-{run_id}-")

    target = run["target"]
    speed_target = run["target_speedup"] if "target_speedup" in run and run["target_speedup"] else "-"
    job_states = Counter(j["state"] for j in run.get("jobs", []))
    inflight = " ".join(f"{k}={v}" for k, v in sorted(job_states.items())) or "none"

    print(_c("━" * 64, "90"))
    print(_c(run_id, "1;37"))
    print(f"  target={_c(target, '36')}  target_speedup={speed_target}  "
          f"agents_running={_c(str(len(running)), '32' if running else '90')}  jobs[{inflight}]")

    if not archive:
        print(_c("  no correct submissions yet", "33"))
        return

    entries = sorted(archive, key=lambda e: e["achieved_at"])
    times = [datetime.fromisoformat(e["achieved_at"]) for e in entries]
    span = (times[-1] - times[0]).total_seconds()
    per_agent = Counter(e["agent_index"] for e in entries)

    if best and best.get("kernel_us"):
        tag = _c(f"{best['speedup']:.4f}x", "1;32" if best["speedup"] >= 1.0 else "1;33")
        print(f"  best {tag}  {best['kernel_us']:.2f}us vs {best['baseline_us']:.2f}us "
              f"(agent {best['agent_index']})   submissions={len(entries)}  span={_fmt_span(span)}")

    # cumulative best speedup over time
    series = _cumbest_series(entries, times)
    print()
    print(f"  best speedup over time   {series[0]:.2f}x {_sparkline(series)} {series[-1]:.2f}x")
    print(_c(f"  (green = ≥1.0x beats baseline · {times[0]:%H:%M:%S} → {times[-1]:%H:%M:%S})", "90"))

    # submissions per agent
    print()
    print(_c("  submissions / agent", "1;37"))
    smax = max(per_agent.values())
    for a in sorted(per_agent):
        print(f"    a{a}  {_c(_bar(per_agent[a], smax), '36')} {per_agent[a]}")

    # best speedup per agent
    if agent_bests:
        print()
        print(_c("  best speedup / agent", "1;37"))
        bmax = max(e["speedup"] for e in agent_bests)
        for e in sorted(agent_bests, key=lambda e: e["agent_index"]):
            col = "32" if e["speedup"] >= 1.0 else "33"
            print(f"    a{e['agent_index']}  {_c(_bar(e['speedup'], bmax), col)} {e['speedup']:.3f}x")

    # record-setting progression
    records, hi = [], 0.0
    for e in entries:
        if e["speedup"] > hi:
            hi = e["speedup"]
            records.append(e)
    print()
    print(_c(f"  progression ({len(records)} new records)", "1;37"))
    for e in records[-8:]:
        ts = datetime.fromisoformat(e["achieved_at"])
        mark = _c("▲", "32") if e["speedup"] >= 1.0 else _c("△", "33")
        print(f"    {ts:%H:%M:%S} {mark} {e['speedup']:.4f}x  a{e['agent_index']}")


def _cmd_status(args) -> int:
    orch = _orchestrator(args)
    while True:
        if args.interval and _USE_COLOR:
            print("\033[2J\033[H", end="")
        _print_status(orch, args.run_id)
        if not args.interval:
            return 0
        time.sleep(args.interval)


def _cmd_watch(args) -> int:
    orch = _orchestrator(args)
    result = orch.watch_speedup(args.run_id, poll_interval=args.interval,
                                postprocess=not args.no_postprocess)
    print(json.dumps(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernel-orchestrator")
    parser.add_argument("--scaffold-dir", default=str(Path(__file__).resolve().parents[1]))
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch")
    launch.add_argument("--plugin", default=None)
    launch.add_argument("--target", default=None)
    launch.add_argument("--shapes", default=None, help="JSON: one shape object or a list of them")
    launch.add_argument("--model", default=None)
    launch.add_argument("--n-agents", type=int, default=None)
    launch.add_argument("--start-index", type=int, default=None)
    launch.add_argument("--target-speedup", type=float, default=None)
    launch.add_argument("--starter-mode", default=None)
    launch.add_argument("--preset-path", default=None)
    launch.add_argument("--auto-terminate", action=argparse.BooleanOptionalAction, default=None)
    launch.add_argument("--poll-interval", type=float, default=None)
    launch.add_argument("--max-iterations", type=int, default=None,
                        help="Max agent sessions per container before it exits (default: scaffold.yaml or 1)")
    launch.add_argument("--no-build", action="store_true")
    launch.set_defaults(func=_cmd_launch)

    start = sub.add_parser("start-agent")
    start.add_argument("--run-id", required=True)
    start.add_argument("--task-slug", required=True)
    start.add_argument("--plugin", default=None)
    start.add_argument("--target", default=None)
    start.add_argument("--model", default=None)
    start.add_argument("--agent-index", type=int, required=True)
    start.add_argument("--starter-mode", default=None)
    start.add_argument("--preset-path", default=None)
    start.add_argument("--max-iterations", type=int, default=None,
                       help="Max agent sessions for this container before it exits (default: scaffold.yaml or 1)")
    start.add_argument("--no-build", action="store_true")
    start.set_defaults(func=_cmd_start_agent)

    stop = sub.add_parser("stop")
    stop.add_argument("container_name")
    stop.set_defaults(func=_cmd_stop)

    stop_run = sub.add_parser("stop-run")
    stop_run.add_argument("run_id")
    stop_run.set_defaults(func=_cmd_stop_run)

    finish = sub.add_parser("finish")
    finish.add_argument("run_id")
    finish.add_argument("--no-postprocess", action="store_true")
    finish.set_defaults(func=_cmd_finish)

    watch = sub.add_parser("watch")
    watch.add_argument("run_id")
    watch.add_argument("--interval", type=float, default=60.0)
    watch.add_argument("--no-postprocess", action="store_true")
    watch.set_defaults(func=_cmd_watch)

    status = sub.add_parser("status")
    status.add_argument("run_id")
    status.add_argument("--interval", type=float, default=0.0,
                        help="repeat every N seconds (0 = one-shot)")
    status.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
