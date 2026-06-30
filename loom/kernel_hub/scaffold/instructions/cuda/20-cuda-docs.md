# CUDA reference docs (offline, grep-able)

You have a **`cuda-docs`** skill with NVIDIA's full reference docs converted to
searchable markdown — read `.agents/skills/cuda-docs/SKILL.md` first, then
`grep`/`rg` the `references/` dirs instead of guessing:

- **PTX ISA 9.1** — `.agents/skills/cuda-docs/references/ptx-docs/` — instruction-level
  detail: WGMMA/WMMA register fragments, TMA, `mbarrier`, swizzle modes, async copy.
- **CUDA Runtime API** — `.agents/skills/cuda-docs/references/cuda-runtime-docs/` — error
  codes, API params, `cudaDeviceProp`, memory/stream behavior.
- **CUDA Driver API** — `.agents/skills/cuda-docs/references/cuda-driver-docs/` — contexts,
  module loading, virtual memory, `CUDA_ERROR_*`.

Each area has a short search guide (`references/ptx-isa.md`, `references/cuda-runtime.md`,
`references/cuda-driver.md`) with example queries. Use this for correct intrinsics, ABIs,
and instruction semantics before writing inline PTX or low-level CUDA — don't rely on memory.
