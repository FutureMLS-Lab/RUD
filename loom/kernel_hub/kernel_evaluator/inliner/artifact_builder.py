"""On-demand artifact builder for kernel compilation.

Compiles .cu source to .so and .ptx files, caching results by source hash.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

ARTIFACT_CACHE_DIR = os.environ.get("TKCC_ARTIFACT_CACHE", "/tmp/tkcc_artifacts")

NVCC_FLAGS = [
    "-std=c++20", "-O3", "--use_fast_math",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-Xcompiler", "-fPIC",
    "-diag-suppress", "20013,20015,20091,2809,177",
]


def get_artifact_paths(kernel_source: str, function_name: str) -> tuple[str, str, str]:
    """Return (so_path, ptx_path, cubin_path) for a kernel, building if needed.

    Args:
        kernel_source: The .cu source code.
        function_name: The function name prefix (used for file naming).

    Returns:
        Tuple of (so_path, ptx_path, cubin_path).
    """
    source_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]
    cache_dir = Path(ARTIFACT_CACHE_DIR) / source_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    cu_path = cache_dir / f"{function_name}.cu"
    so_path = cache_dir / f"{function_name}.so"
    ptx_path = cache_dir / f"{function_name}.ptx"
    cubin_path = cache_dir / f"{function_name}.cubin"

    if not so_path.exists() or not ptx_path.exists():
        cu_path.write_text(kernel_source)
        _compile_kernel(cu_path, so_path, ptx_path)

    return str(so_path), str(ptx_path), str(cubin_path)


def _compile_kernel(cu_path: Path, so_path: Path, ptx_path: Path) -> None:
    """Compile .cu to .so and .ptx."""
    # Compile to .so
    subprocess.run([
        "nvcc", *NVCC_FLAGS, "-shared",
        "-o", str(so_path), str(cu_path),
        "-lcuda", "-lcudart",
    ], check=True)

    # Compile to .ptx
    subprocess.run([
        "nvcc", *NVCC_FLAGS, "-ptx",
        "-o", str(ptx_path), str(cu_path),
    ], check=True)


def find_kernel_symbol(ptx_path: str, must_contain: str | None = None) -> str | None:
    """Find the kernel symbol in a PTX file.

    Args:
        ptx_path: Path to the .ptx file.
        must_contain: If provided, only match symbols containing this string.

    Returns:
        The kernel symbol name, or None if not found.
    """
    with open(ptx_path) as f:
        ptx_text = f.read()

    pattern = r'\.visible\s+\.entry\s+(\w+)'
    matches = re.findall(pattern, ptx_text)

    if not matches:
        return None

    if must_contain:
        for m in matches:
            if must_contain in m:
                return m

    return matches[0]


def build_pdl_artifacts(ptx_path: str, output_dir: Path) -> tuple[str, str]:
    """Inject PDL griddepcontrol and build cubin for PDL-enabled dispatch.

    Args:
        ptx_path: Path to the clean .ptx file.
        output_dir: Directory to write the PDL artifacts.

    Returns:
        Tuple of (pdl_ptx_path, pdl_cubin_path).
    """
    from kernel_evaluator.inliner.inject_ptx import inject_pdl_into_ptx, assemble_ptx_to_cubin

    with open(ptx_path) as f:
        clean_ptx = f.read()

    pdl_ptx = inject_pdl_into_ptx(clean_ptx, early_launch=False)

    pdl_ptx_path = output_dir / "kernel_pdl.ptx"
    pdl_cubin_path = output_dir / "kernel_pdl.cubin"

    pdl_ptx_path.write_text(pdl_ptx)
    assemble_ptx_to_cubin(pdl_ptx, str(pdl_cubin_path))

    return str(pdl_ptx_path), str(pdl_cubin_path)
