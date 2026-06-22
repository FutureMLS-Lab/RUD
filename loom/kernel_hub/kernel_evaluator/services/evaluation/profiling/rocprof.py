import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from kernel_evaluator.services.evaluation.profiling.types import ProfilePolicy


COLUMN_ALIASES = {
    "Counter_Name": ("Counter_Name", "CounterName"),
    "Counter_Value": ("Counter_Value", "CounterValue", "Value"),
    "Kernel_Name": ("Kernel_Name", "KernelName", "Name"),
    "Dispatch_Id": ("Dispatch_Id", "DispatchID", "DispatchId", "Dispatch Id"),
}

DERIVED_METRICS = (
    ("Icache_Hit_Rate", "SQC_ICACHE_HITS", "SQC_ICACHE_REQ"),
    ("Icache_Miss_Rate", "SQC_ICACHE_MISSES", "SQC_ICACHE_REQ"),
    ("Scache_Hit_Rate", "SQC_DCACHE_HITS", "SQC_DCACHE_REQ"),
    ("Frac_INSTS_VALU", "SQ_INSTS_VALU", "SQ_INSTS"),
    ("Frac_INSTS_SALU", "SQ_INSTS_SALU", "SQ_INSTS"),
    ("Frac_INSTS_SMEM", "SQ_INSTS_SMEM", "SQ_INSTS"),
    ("Frac_INSTS_VMEM", "SQ_INSTS_VMEM", "SQ_INSTS"),
    ("Frac_Active_VMEM", "SQ_ACTIVE_INST_VMEM", "SQ_ACTIVE_INST_ANY"),
    ("SQ_Busy_Ratio", "SQ_BUSY_CYCLES", "SQ_CYCLES"),
    ("Waves_per_WG", "SPI_CSN_WAVE", "SPI_CSN_NUM_THREADGROUPS"),
    ("LDS_Conflict_Ratio", "SQ_LDS_BANK_CONFLICT", "SQ_INSTS_LDS"),
    ("L2_Hit_Rate", "TCC_HIT_sum", "TCC_REQ_sum"),
    ("L2_Miss_Rate", "TCC_MISS_sum", "TCC_REQ_sum"),
    ("vL1_Read_Frac", "TCP_TOTAL_READ_sum", "TCP_TOTAL_ACCESSES_sum"),
    ("vL1_Write_Frac", "TCP_TOTAL_WRITE_sum", "TCP_TOTAL_ACCESSES_sum"),
)

RAW_TOTALS = ("SQ_WAVES", "SPI_CSN_NUM_THREADGROUPS", "SQ_INSTS", "SQ_CYCLES")


class RocprofCli:
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
        report_dir = report_base
        report_dir.mkdir(parents=True, exist_ok=True)
        pmc_dir = report_dir / "pmc"
        pmc_dir.mkdir(parents=True, exist_ok=True)
        for group_index, counters in enumerate(policy.pmc_counter_groups):
            out_dir = pmc_dir / f"group_{group_index:02d}"
            cmd = [
                "rocprofv3",
                "--pmc",
                ",".join(counters),
                "--output-format",
                "csv",
                "--output-file",
                f"profiles_{group_index}",
                "-d",
                str(out_dir),
                *extra_args,
                "--",
                *runner_cmd,
            ]
            self._run(cmd, env, work_dir)
        if policy.att_enabled:
            self._collect_att(runner_cmd, report_dir, env, work_dir, policy)
        summary_text = _summarize_pmc(sorted(pmc_dir.rglob("*counter_collection*.csv")))
        return report_dir, summary_text

    def _collect_att(
        self,
        runner_cmd: list[str],
        report_dir: Path,
        env: dict[str, str],
        work_dir: Path,
        policy: ProfilePolicy,
    ) -> None:
        att_dir = report_dir / "att"
        att_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rocprofv3",
            "--att",
            "true",
            "--att-library-path",
            policy.att_library_path,
            "-d",
            str(att_dir),
            "--",
            *runner_cmd,
        ]
        self._run(cmd, env, work_dir)
        for code_json in att_dir.rglob("code.json"):
            _extract_asm(code_json, code_json.with_suffix(".s"))

    def _run(self, cmd: list[str], env: dict[str, str], work_dir: Path) -> None:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=work_dir, timeout=self.timeout_s
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr + completed.stdout).strip())


def _resolve_column(headers: list[str], canon: str) -> str:
    for alias in COLUMN_ALIASES[canon]:
        if alias in headers:
            return alias
    raise KeyError(f"missing column {canon} in {headers}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _summarize_pmc(csv_paths: list[Path]) -> str:
    counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if len(rows) < 2:
            continue
        headers = rows[0]
        name_col = headers.index(_resolve_column(headers, "Counter_Name"))
        value_col = headers.index(_resolve_column(headers, "Counter_Value"))
        kernel_col = headers.index(_resolve_column(headers, "Kernel_Name"))
        for row in rows[1:]:
            kernel = row[kernel_col]
            counter = row[name_col]
            counters[kernel][counter] += float(row[value_col])

    if len(counters) == 0:
        return "rocprof_pmc: no counters collected\n"

    parts: list[str] = []
    for kernel_index, kernel in enumerate(sorted(counters)):
        values = counters[kernel]
        parts.append(f"kernel_index: {kernel_index}")
        parts.append(f"  Kernel Name: {kernel}")
        for label in RAW_TOTALS:
            if label in values:
                parts.append(f"  {label}: {values[label]:.0f}")
        for label, numerator, denominator in DERIVED_METRICS:
            if numerator in values and denominator in values:
                parts.append(f"  {label}: {_safe_ratio(values[numerator], values[denominator]):.4f}")
    return "\n".join(parts) + "\n"


def _extract_asm(code_json: Path, output_path: Path) -> None:
    data = json.loads(code_json.read_text(encoding="utf-8"))
    code = data.get("code")
    if not code:
        return
    lines = []
    for item in code:
        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], str) and item[0].strip():
            lines.append(item[0])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
