from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PMC_COUNTER_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "SQC_ICACHE_MISSES", "SQC_DCACHE_REQ", "SQC_DCACHE_HITS", "SQC_DCACHE_MISSES",
        "SQ_ACTIVE_INST_ANY", "SQ_ACTIVE_INST_VMEM", "SQ_WAIT_INST_ANY",
    ),
    (
        "SQC_ICACHE_REQ", "SQC_ICACHE_HITS", "SQ_INSTS", "SQ_INSTS_VALU",
        "SQ_INSTS_SALU", "SQ_INSTS_SMEM", "SQ_INSTS_VMEM",
    ),
    (
        "SPI_CSN_WAVE", "SQ_WAVES", "SQ_CYCLES", "SQ_BUSY_CYCLES", "SPI_CSN_BUSY",
        "SPI_CSN_NUM_THREADGROUPS", "SQ_LDS_BANK_CONFLICT", "SQ_INSTS_LDS",
        "TCC_REQ_sum", "TCC_HIT_sum", "TCC_MISS_sum", "TCC_ATOMIC_sum",
        "TCP_TOTAL_ACCESSES_sum", "TCP_TOTAL_READ_sum", "TCP_TOTAL_WRITE_sum",
    ),
)


@dataclass(frozen=True)
class ProfilePolicy:
    set_name: str = "basic"
    summary_page: str = "raw"
    launch_count: int = 1
    warmup_launches: int = 1
    extra_ncu_args: tuple[str, ...] = ()
    pmc_counter_groups: tuple[tuple[str, ...], ...] = PMC_COUNTER_GROUPS
    att_enabled: bool = True
    att_library_path: str = "/opt/rocm/lib"


@dataclass(frozen=True)
class ProfileShapeResult:
    shape_index: int
    task_slug: str
    report_path: Path
    summary_text: str


class ProfilerCli(Protocol):
    def profile(
        self,
        runner_cmd: list[str],
        report_base: Path,
        env: dict[str, str],
        work_dir: Path,
        policy: ProfilePolicy,
        extra_args: tuple[str, ...],
    ) -> tuple[Path, str]:
        ...
