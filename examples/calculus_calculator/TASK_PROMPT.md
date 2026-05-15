# Agent Prompt

## Role
You are a software development agent. Your goal is to build a high-performance,
production-grade calculus engine as a Python library with a CLI interface. The
system must be both mathematically rigorous and computationally efficient.

## Project Description
Build `calccli`, a Python package that performs symbolic and numerical calculus
operations. The library must support advanced mathematical operations, handle
edge cases robustly, and meet strict performance benchmarks.

### Core Features

1. **Symbolic differentiation** — compute derivatives of arbitrary expressions
   including polynomials, trigonometric, exponential, logarithmic, hyperbolic
   functions, and compositions via chain rule. Support nth-order derivatives.
2. **Symbolic integration** — compute antiderivatives for standard forms
   (polynomials, trig, exponential, 1/x, hyperbolic). Support definite integrals
   with symbolic bounds.
3. **Numerical integration** — approximate definite integrals using multiple
   methods: Simpson's rule, Gauss-Legendre quadrature (5-point), and adaptive
   quadrature. The user selects the method via a parameter (default: adaptive).
4. **Limits** — evaluate limits of expressions at a point (including ±infinity),
   with support for one-sided limits (left/right).
5. **Taylor / Laurent series** — expand a function around a point to n terms.
   Detect and handle poles by producing Laurent series with negative-power terms
   when appropriate.
6. **Multivariable calculus** — partial derivatives, gradient vectors, and the
   Jacobian matrix for vector-valued functions of multiple variables.
7. **ODE solver** — solve first-order and second-order ordinary differential
   equations symbolically (separable, linear, exact, and constant-coefficient
   types). Return the general solution with arbitrary constants.
8. **Expression simplification & caching** — implement an LRU cache for parsed
   expressions so that repeated computations on the same expression are O(1).
   Canonicalize expressions before caching.
9. **LaTeX output** — every operation supports an optional `--latex` flag (CLI)
   or `latex=True` parameter (API) that renders the result in LaTeX notation.
10. **CLI** — a `python -m calccli` interface that accepts an operation, an
    expression, and parameters, then prints the result.

### Expression format
Use a simple string representation. Variables are single letters (default `x`).
Support standard math notation: `x^3 + 2*x`, `sin(x)`, `cos(x)`, `tan(x)`,
`exp(x)`, `log(x)`, `sqrt(x)`, `sinh(x)`, `cosh(x)`, `tanh(x)`, `asin(x)`,
`acos(x)`, `atan(x)`, `abs(x)`.

Expressions must be parsed correctly even with missing multiplication signs
(e.g., `2x` should be interpreted as `2*x`).

You may use `sympy` as the symbolic math engine.

### CLI examples
```
python -m calccli diff "x^3 + 2*x" x
# => 3*x**2 + 2

python -m calccli diff "sin(x)*exp(x)" x --order 2
# => 2*exp(x)*cos(x)

python -m calccli integrate "sin(x)" x
# => -cos(x)

python -m calccli integrate "x^2" x --definite 0 1
# => 1/3

python -m calccli numint "x^2" x 0 1 --method simpson
# => 0.3333...

python -m calccli numint "exp(-x^2)" x 0 3 --method gauss
# => 0.8862...

python -m calccli limit "sin(x)/x" x 0
# => 1

python -m calccli limit "1/x" x 0 --side right
# => oo

python -m calccli taylor "exp(x)" x 0 5
# => 1 + x + x**2/2 + x**3/6 + x**4/24

python -m calccli taylor "1/x" x 1 4
# => 1 - (x - 1) + (x - 1)**2 - (x - 1)**3

python -m calccli partial "x^2*y + y^3" x
# => 2*x*y

python -m calccli gradient "x^2*y + y^3" x y
# => [2*x*y, x**2 + 3*y**2]

python -m calccli jacobian "x^2+y, x*y" "x, y"
# => [[2*x, 1], [y, x]]

python -m calccli ode "y' - 2*y = 0"
# => y(x) = C1*exp(2*x)

python -m calccli ode "y'' + y = 0"
# => y(x) = C1*sin(x) + C2*cos(x)

python -m calccli diff "x^3" x --latex
# => 3 x^{2}
```

## Project structure
```
calccli/
    __init__.py          # public API re-exports
    engine.py            # core symbolic: diff, integrate, limit, taylor/laurent
    numerical.py         # numerical integration (simpson, gauss-legendre, adaptive)
    multivariable.py     # partial derivatives, gradient, jacobian
    ode_solver.py        # symbolic ODE solving
    parser.py            # expression parsing & implicit multiplication
    cache.py             # LRU expression cache with canonicalization
    latex_render.py       # LaTeX output formatting
    cli.py               # CLI argument parsing & dispatch
    __main__.py          # python -m calccli entry point
tests/
    __init__.py
    test_diff.py
    test_integrate.py
    test_numerical.py
    test_limits.py
    test_taylor.py
    test_multivariable.py
    test_ode.py
    test_parser.py
    test_cache.py
    test_latex.py
    test_cli.py
    test_performance.py  # performance benchmark tests
```

## Constraints
- Use Python 3.10+
- Use `sympy` for symbolic math
- All public functions must have type hints
- Include comprehensive tests using `pytest`
- Handle invalid input gracefully with clear error messages and non-zero exit codes
- Always update PLAN.md with your progress

### Performance requirements
- Parsing and differentiating a degree-50 polynomial must complete in < 2 seconds.
- Numerical integration (adaptive) of `sin(x)` over [0, pi] must be accurate to
  1e-10 and complete in < 1 second.
- The LRU cache must demonstrate at least 10x speedup on repeated identical
  computations (measured via `test_performance.py`).
- Gauss-Legendre 5-point quadrature must match exact integrals of polynomials up
  to degree 9 within machine epsilon (~1e-14).

## Tools Available
You have access to Claude Code tools: Bash, Edit, Write, Read, Glob, Grep.
Use them to explore the codebase, write code, and run tests.
