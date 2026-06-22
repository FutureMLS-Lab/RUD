import csv
import subprocess
from collections import defaultdict
from pathlib import Path

from kernel_evaluator.services.evaluation.profiling.types import ProfilePolicy


SUMMARY_SECTIONS = (
    (
        "Kernel",
        (
            ("Name", "Kernel Name", ""),
            ("GPU Time Avg", "gpu__time_duration.avg", "us"),
            ("GPU Time Max", "gpu__time_duration.max", "us"),
            ("GPU Time Min", "gpu__time_duration.min", "us"),
            ("Replay Passes", "profiler__replayer_passes", "passes"),
        ),
    ),
    (
        "Launch",
        (
            ("Block Size", "launch__block_size", "threads"),
            ("Grid Size", "launch__grid_size", "blocks"),
            ("Registers / Thread", "launch__registers_per_thread", ""),
            ("Shared Mem / Block", "launch__shared_mem_per_block", "Kbyte"),
        ),
    ),
    (
        "Speed Of Light",
        (
            ("SM Throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "%"),
            ("Compute Memory Throughput", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "%"),
            ("DRAM Throughput", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "%"),
            ("L1/TEX Throughput", "l1tex__throughput.avg.pct_of_peak_sustained_active", "%"),
            ("L2 Throughput", "lts__throughput.avg.pct_of_peak_sustained_elapsed", "%"),
            ("Instruction Throughput", "sm__inst_executed.sum.pct_of_peak_sustained_elapsed", "%"),
        ),
    ),
)


class NcuCli:
    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s

    def profile(
        self,
        runner_cmd: list[str],
        report_base: Path,
        env: dict[str, str],
        work_dir: Path,
        policy: ProfilePolicy,
        extra_args: tuple[str, ...],
    ) -> tuple[Path, str]:
        cmd = [
            "ncu",
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "-f",
            "--set",
            policy.set_name,
            "-o",
            str(report_base),
            *extra_args,
            *policy.extra_ncu_args,
            *runner_cmd,
        ]
        self._run(cmd, env, work_dir)
        report_path = report_base.with_suffix(".ncu-rep")
        if not report_path.exists():
            raise RuntimeError(f"ncu did not produce report: {report_path}")
        return report_path, _format_summary(self._raw_summary(report_path, env, work_dir, policy))

    def _raw_summary(self, report_path: Path, env: dict[str, str], work_dir: Path, policy: ProfilePolicy) -> str:
        cmd = ["ncu", "-i", str(report_path), "--page", policy.summary_page, "--csv"]
        return self._run(cmd, env, work_dir)

    def _run(self, cmd: list[str], env: dict[str, str], work_dir: Path) -> str:
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work_dir, timeout=self.timeout_s)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr + completed.stdout).strip())
        return completed.stdout


def _format_value(value: str, unit: str) -> str:
    if value == "":
        return "n/a"
    if unit == "":
        return value
    return f"{value} {unit}"


def _summary_rows(summary_text: str) -> list[defaultdict[str, str]]:
    lines = summary_text.splitlines()
    if len(lines) < 3:
        return []
    rows = list(csv.reader(lines))
    if len(rows) < 3:
        return []
    headers = rows[0]
    return [defaultdict(str, zip(headers, row)) for row in rows[2:]]


def _format_summary(summary_text: str) -> str:
    rows = _summary_rows(summary_text)
    if len(rows) == 0:
        return "ncu_summary: unavailable\n"
    parts = []
    for kernel_index, row in enumerate(rows):
        parts.append(f"kernel_index: {kernel_index}")
        for section, fields in SUMMARY_SECTIONS:
            parts.append(f"{section}:")
            for label, field, unit in fields:
                parts.append(f"  {label}: {_format_value(row[field], unit)}")
    return "\n".join(parts) + "\n"
