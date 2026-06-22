import re
import os


def _is_ptx_declaration(stripped: str) -> bool:
    """Return True if a stripped PTX line is a declaration or blank (not an instruction)"""
    return (
        stripped.startswith('.reg ')
        or stripped.startswith('.local ')
        or stripped.startswith('.shared ')
        or stripped.startswith('.param ')
        or not stripped
        or stripped.startswith('//')
    )


def inject_pdl_into_ptx(ptx_text: str, early_launch: bool = False) -> str:
    """
    Inject PDL griddepcontrol instructions into every .visible .entry.

    When early_launch=False (default): griddepcontrol.wait before first
    instruction, griddepcontrol.launch_dependents before each ret;.

    When early_launch=True: both wait and launch_dependents before first
    instruction (successor can start immediately while this kernel runs).
    """
    lines = ptx_text.split('\n')
    out = []
    in_entry = False
    brace_depth = 0
    wait_inserted = False

    for line in lines:
        stripped = line.strip()

        if '.visible .entry' in line: # Start of a kernel entry definition.
            in_entry = True
            brace_depth = 0
            wait_inserted = False

        if in_entry:
            if stripped == '{': # Enter a new scope level.
                brace_depth += 1
                out.append(line)
                continue

            if stripped == '}': # Exit the scope
                brace_depth -= 1
                out.append(line)
                if brace_depth == 0:
                    in_entry = False
                continue

            if brace_depth >= 1: # Inside the kernel body
                if not wait_inserted: # Inject wait before the first non-declaration instruction.
                    if not _is_ptx_declaration(stripped):
                        indent = re.match(r'^(\s*)', line).group(1)
                        out.append(f'{indent}griddepcontrol.wait;')
                        if early_launch:
                            out.append(f'{indent}griddepcontrol.launch_dependents;')
                        wait_inserted = True

                if not early_launch and stripped == 'ret;':
                    indent = re.match(r'^(\s*)', line).group(1)
                    out.append(f'{indent}griddepcontrol.launch_dependents;')

        out.append(line)

    return '\n'.join(out)


def inject_timing_into_ptx(ptx_text: str) -> str:
    """
    Inject smid-indexed start/end globaltimer writes into every .visible .entry in PTX.
    """
    lines = ptx_text.split('\n')
    out: list[str] = []

    in_entry = False
    in_header = False
    brace_depth = 0
    header_start_out_idx = 0
    regs_injected = False

    for line in lines:
        stripped = line.strip()
        indent = re.match(r'^(\s*)', line).group(1)

        if '.visible .entry' in line: # Start of a kernel entry definition.
            in_entry = True
            in_header = True
            brace_depth = 0
            header_start_out_idx = len(out)
            regs_injected = False

        if not in_entry:
            out.append(line)
            continue

        if in_header:
            if stripped == ')':
                # Back-patch: add trailing comma to the last .param line in the header.
                for rev_idx in range(len(out) - 1, header_start_out_idx - 1, -1):
                    prev = out[rev_idx].rstrip()
                    if prev.strip().startswith('.param '):
                        if not prev.endswith(','):
                            out[rev_idx] = prev + ','
                        break
                out.append(f"{indent}\t.param .u64 tkcc_timing_buf_param")
                out.append(line)   # the ')' line
                continue

            if '{' in stripped:
                in_header = False
                brace_depth += stripped.count('{') - stripped.count('}')
                if brace_depth <= 0:
                    brace_depth = 0
                    in_entry = False
            out.append(line)
            continue

        if stripped == '{':
            brace_depth += 1
            out.append(line)
            continue

        if stripped == '}':
            brace_depth -= 1
            out.append(line)
            if brace_depth == 0:
                in_entry = False
            continue

        if not regs_injected: # Inject register declarations + start-write before first instruction
            if not _is_ptx_declaration(stripped):
                out.extend([
                    f"{indent}.reg .pred  %tkcc_is_tid0;",
                    f"{indent}.reg .u32   %tkcc_tid0, %tkcc_smid;",
                    f"{indent}.reg .u64   %tkcc_tb, %tkcc_gt, %tkcc_slot_addr, %tkcc_discard;",
                ])
                regs_injected = True

                out.extend([
                    f"{indent}mov.u32        %tkcc_tid0, %tid.x;",
                    f"{indent}setp.eq.u32    %tkcc_is_tid0, %tkcc_tid0, 0;",
                    f"@%tkcc_is_tid0 mov.u32        %tkcc_smid, %smid;",
                    f"@%tkcc_is_tid0 ld.param.u64   %tkcc_tb, [tkcc_timing_buf_param];",
                    f"@%tkcc_is_tid0 mul.wide.u32   %tkcc_slot_addr, %tkcc_smid, 16;",
                    f"@%tkcc_is_tid0 add.u64        %tkcc_slot_addr, %tkcc_slot_addr, %tkcc_tb;",
                    f"@%tkcc_is_tid0 mov.u64        %tkcc_gt, %globaltimer;",
                    f"@%tkcc_is_tid0 atom.global.cas.b64 %tkcc_discard, [%tkcc_slot_addr], 0, %tkcc_gt;",
                ])

        if stripped == 'ret;':
            out.extend([
                f"@%tkcc_is_tid0 mov.u64        %tkcc_gt, %globaltimer;",
                f"@%tkcc_is_tid0 red.global.max.u64 [%tkcc_slot_addr+8], %tkcc_gt;",
            ])

        out.append(line)

    return '\n'.join(out)


def assemble_ptx_to_cubin(ptx_text: str, cubin_path: str) -> None:
    """Assemble PTX text to a cubin via ptxas.

    The target architecture is extracted from the ``.target`` directive in the PTX.
    """
    import subprocess

    m = re.search(r'\.target\s+(sm_\w+)', ptx_text)
    if not m:
        raise ValueError("PTX is missing .target directive; cannot determine architecture")
    arch = m.group(1)

    ptx_path = cubin_path.replace('.cubin', '.ptx')
    with open(ptx_path, 'w') as f:
        f.write(ptx_text)

    subprocess.run(
        ['ptxas', f'-arch={arch}', '-o', cubin_path, ptx_path],
        check=True,
    )
