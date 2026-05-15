# Calculus Calculator (calccli) — Implementation Plan

## Overview
Build a production-grade Python calculus engine with CLI interface using sympy.

## Architecture
- **parser.py**: Expression parsing with implicit multiplication (`2x` → `2*x`) and caret notation (`x^2` → `x**2`)
- **cache.py**: LRU cache with expression canonicalization for O(1) repeated computations
- **engine.py**: Core symbolic operations — differentiation, integration, limits, Taylor/Laurent series
- **numerical.py**: Numerical integration — Simpson's rule, Gauss-Legendre 5-point, adaptive quadrature
- **multivariable.py**: Partial derivatives, gradient vectors, Jacobian matrices
- **ode_solver.py**: Symbolic ODE solving (separable, linear, exact, constant-coefficient)
- **latex_render.py**: LaTeX output formatting
- **cli.py / __main__.py**: CLI argument parsing and dispatch

## Tasks

### Iteration 1 — Full Implementation (COMPLETED)
- [x] Create PLAN.md
- [x] Set up project structure (all module files, `__init__.py`, `__main__.py`)
- [x] Implement `parser.py` — expression parsing with implicit multiplication
- [x] Implement `cache.py` — LRU expression cache
- [x] Implement `engine.py` — differentiate, integrate, limit, taylor_expand
- [x] Implement `numerical.py` — Simpson, Gauss-Legendre, adaptive quadrature
- [x] Implement `multivariable.py` — partial_derivative, gradient, jacobian
- [x] Implement `ode_solver.py` — solve_ode
- [x] Implement `latex_render.py` — to_latex
- [x] Implement `cli.py` + `__main__.py` — CLI interface
- [x] Write all 11 test files (110 tests total)
- [x] Run tests and fix issues (fixed adaptive quadrature bug)
- [x] Verify all SUCCESS_CONDITION checks pass

## Completed Work (Iteration 1)

### Implementation
All 10 modules implemented:
- `parser.py`: Handles implicit multiplication (`2x`, `3sin(x)`) and caret notation (`x^2`)
- `cache.py`: LRU cache with canonicalization, achieves 750x+ speedup
- `engine.py`: differentiate (nth-order), integrate (definite/indefinite), limit (one-sided, infinity), taylor_expand (handles Laurent series at poles)
- `numerical.py`: Simpson's rule (1000 subintervals), Gauss-Legendre 5-point (exact for degree ≤ 9), adaptive Simpson's quadrature (1e-12 tolerance)
- `multivariable.py`: partial_derivative, gradient, jacobian
- `ode_solver.py`: Parses ODE notation (y', y''), solves via sympy.dsolve
- `latex_render.py`: Converts sympy expressions/lists to LaTeX
- `cli.py`: 9 subcommands (diff, integrate, numint, limit, taylor, partial, gradient, jacobian, ode) with --latex flag
- `__main__.py`: Entry point for `python -m calccli`
- `__init__.py`: Public API re-exports

### Bug Fixed
- Adaptive quadrature had incorrect interval length calculation in recursive subdivision (used full interval `(b-a)` instead of half-interval `(b-a)/2` for left/right sub-estimates)

### Test Results
- **110 tests, all passing** in 6.28 seconds
- All 13 CLI grep checks from SUCCESS_CONDITION pass
- All 3 error handling checks pass
- Performance: degree-50 polynomial diff in 0.083s (< 2s requirement)
- Accuracy: adaptive sin integral accurate to 1e-10, Gauss-Legendre exact to 1e-14 for degree-9
- Cache: 757x speedup (> 10x requirement)

## Next Steps
- Project is feature-complete and all tests pass
- No remaining work needed unless additional features are requested

## Status
**Current**: COMPLETE — All features implemented, all tests passing
