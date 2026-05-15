# Task: Write a 4-Page MLSys Paper for Claudeloop

## Overview

Write a 4-page academic paper (in LaTeX, using the MLSys or NeurIPS style) presenting **claudeloop**: a self-improving agentic loop framework for autonomous software development built on top of Claude Code.

The paper should be placed in a directory called `paper/` with the main file `paper/main.tex` and compiled to `paper/main.pdf`.

---

## Paper Structure

### Title
"Claudeloop: A Self-Improving Agentic Loop Framework for Autonomous Software Development"

### Abstract (~150 words)
- Summarize the motivation: autonomous coding agents need iterative refinement with clear success criteria.
- Summarize the system: worker-evaluator loop with hybrid evaluation, cost tracking, and auto-commit.
- Summarize key results: a calculus engine with 110+ tests built in 1 iteration ($1.69, 414s).

### 1. Introduction (~0.5 page)
- Motivation: LLM-based coding agents are powerful but lack structured iteration and evaluation.
- Problem: How to build an autonomous system that iteratively improves code until success criteria are met, with cost and quality controls.
- Contribution summary (3 bullets):
  1. A worker-evaluator agentic loop with hybrid evaluation (subprocess tests + LLM judgment).
  2. A modular backend system supporting both CLI (with tool use) and SDK (text-only) modes.
  3. Comprehensive cost tracking, budget caps, and per-iteration logging for reproducibility.

### 2. System Design (~1.5 pages)

This section MUST include a **system overview figure** (use TikZ or pgfplots). The figure should show the full loop:

```
User defines: TASK_PROMPT.md + SUCCESS_CONDITION.md + PLAN.md
         |
         v
    [Worker Agent] -- reads task + plan --> Claude Code CLI (with tools: Bash, Edit, Write, Read, Glob, Grep)
         |
         | writes/edits code, runs tests
         v
    [Git Auto-Commit] -- "claudeloop: iteration N"
         |
         v
    [Evaluator Agent]
         |--- (1) Extract & run bash test commands from SUCCESS_CONDITION.md (subprocess)
         |--- (2) Send test results + criteria to Claude for qualitative judgment
         |--- Returns JSON: {success, reason, suggestions}
         |
    success? --yes--> EXIT (code 0)
         |
         no
         |
    Append suggestions to PLAN.md
         |
         v
    Next iteration (or exit if cost/iter cap reached)
```

#### 2.1 Worker Agent
- Builds prompt from TASK_PROMPT.md + PLAN.md + iteration number.
- Invokes Claude Code CLI backend with full tool access.
- Streams real-time events (tool_use, tool_result, text, error, result).
- ~70 lines of Python; relies on the backend abstraction.

#### 2.2 Evaluator Agent (Hybrid Evaluation)
- **Stage 1 (Deterministic):** Extracts bash commands from SUCCESS_CONDITION.md, runs each via subprocess, captures exit code + stdout + stderr + duration.
- **Stage 2 (LLM-based):** Sends test results + success criteria + current plan to Claude, which returns a structured JSON judgment: `{success: bool, reason: str, suggestions: str}`.
- Appends evaluator suggestions to PLAN.md for the next iteration.
- ~190 lines of Python.

#### 2.3 Backend System
- **Abstract interface:** `Backend` base class with `BackendResponse` and `StreamEvent`.
- **CLI Backend (326 lines):** Wraps Claude Code CLI as a subprocess; supports streaming (`stream-json`) and quiet (`json`) modes; parses tool use events.
- **SDK Backend (77 lines):** Direct Anthropic API calls; text-only (no tool use); used for evaluator when tool access is unnecessary.
- Factory pattern in `__init__.py` for backend selection.

#### 2.4 Configuration & Orchestration
- `RunConfig` dataclass: prompt_path, success_path, plan_path, max_iters, model, backend_name, log_dir, max_cost, auto_commit, verbose, additional_prompt, effort_level, fast_mode.
- Runner (245 lines): manages the full loop, cost accumulation, early exit conditions (success / cost cap / max iterations).
- Rich terminal UI with tables and panels for real-time feedback.

#### 2.5 Git Integration & Logging
- Auto-commits after each iteration with message "claudeloop: iteration N".
- Per-iteration log files: `agent_logs/agent_iter_N_<hash>.log` with cost, duration, output.
- Main log: `agent_logs/claudeloop.log` for cross-iteration tracking.

### 3. Experiments (~1.5 pages)

#### 3.1 Case Study: Calculus Calculator
Describe the task: build a production-grade calculus engine with 110+ tests covering:
- Symbolic differentiation, integration, limits, Taylor/Laurent series
- Numerical integration (Simpson's, Gauss-Legendre, adaptive)
- Multivariable calculus (partial derivatives, gradient, Jacobian)
- ODE solver, expression caching, LaTeX output, CLI interface

**Table 1: Task Completion Results**

| Metric | Value |
|--------|-------|
| Iterations to success | 1 |
| Total cost | $1.69 |
| Wall-clock time | 414s |
| Files created | 21 (10 modules + 11 test files) |
| Tests passing | 110/110 |
| Lines of code generated | ~2,500 |

**Table 2: Performance Benchmarks (from the generated code)**

| Benchmark | Requirement | Achieved | Status |
|-----------|------------|----------|--------|
| Degree-50 polynomial diff | < 2s | 0.083s | PASS |
| Cache speedup | > 10x | 757x | PASS |
| Gauss-Legendre exactness | < 1e-10 | ~1e-14 | PASS |

#### 3.2 System Efficiency Analysis

**Table 3: Cost Breakdown by Component**

| Component | Backend | Approx. Cost | Tokens (est.) |
|-----------|---------|-------------|---------------|
| Worker (iter 1) | CLI | ~$1.50 | ~100K |
| Evaluator (iter 1) | SDK | ~$0.19 | ~15K |
| **Total** | | **$1.69** | **~115K** |

**Figure 2: Token Usage Distribution** (bar chart or pie chart)
- Show input vs output tokens for worker and evaluator.
- Show tool use breakdown if possible.

#### 3.3 Comparison with Manual Development
- Discuss time savings: 414s automated vs estimated hours for manual implementation.
- Discuss cost efficiency: $1.69 for a complete, tested library.

#### 3.4 Ablation: Hybrid Evaluation
- Discuss the importance of combining deterministic tests with LLM judgment.
- Without deterministic tests: LLM might hallucinate success.
- Without LLM judgment: Cannot assess qualitative aspects (code quality, architecture).

### 4. Related Work (~0.3 page)
- SWE-bench, SWE-agent, OpenHands, Devin, Aider
- Self-improving agents (Reflexion, etc.)
- Position claudeloop as lightweight (<1,400 lines), open, and built on Claude Code's native tool ecosystem.

### 5. Conclusion (~0.2 page)
- Summarize contributions.
- Future work: multi-agent parallelism, human-in-the-loop mode, benchmark suite.

---

## Requirements

1. **LaTeX**: Use a standard ML conference style (mlsys2024 or neurips_2024). Place all files in `paper/`.
2. **Figures**: Create at least 2 figures:
   - Figure 1: System overview diagram (TikZ).
   - Figure 2: Token usage / cost analysis chart (pgfplots or TikZ).
3. **Tables**: Create at least 3 tables as described above.
4. **References**: Include at least 8 references (Claude, SWE-bench, SWE-agent, Reflexion, OpenHands, Aider, LangChain, etc.).
5. **Compilation**: The paper must compile with `pdflatex` (run `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`).
6. **Page limit**: 4 pages of content (references may overflow to page 5).
7. **Quality**: The writing should be clear, concise, and suitable for a workshop or short paper submission.

---

## File Structure

```
paper/
├── main.tex          # Main LaTeX file
├── references.bib    # BibTeX references
├── figures/          # Any external figure files (if needed)
└── main.pdf          # Compiled output
```

## Additional Notes

- Use `\usepackage{tikz}` and `\usepackage{pgfplots}` for figures.
- All numbers and metrics should match the actual system (see data above).
- The total codebase is ~1,370 lines of Python across 12 modules.
- The system supports Claude Opus 4.6, Sonnet 4.5, and Haiku 4.5 models.
