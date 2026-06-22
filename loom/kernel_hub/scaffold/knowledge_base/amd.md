## Knowledge Base

A read-only knowledge base is mounted at `/kb`. **Consult the KB before searching online** — it contains curated AMD/ROCm documentation directly relevant to kernel optimization. Use heavily when brainstorming, debugging, or stuck.

**Contents (directly under `/kb`):**
- `aiter/` — AITER kernel library (the production reference stack you must beat)
- `HIP/`, `HIP-rendered/` — HIP programming guide and API reference
- `rocm-libraries/` — ROCm library sources (rocBLAS, Composable Kernel, ...)
- `AGENTS.md` — index / entry point

Use `rg` to search across the KB.

Not in KB (search online): MI300/MI355 ISA quirks, the latest aiter changes.
