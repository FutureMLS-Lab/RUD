# Success Conditions

## Criteria

### Core Calculus (1–6)
1. The `calccli` package is importable and exposes `differentiate`, `integrate`, `numerical_integrate`, `limit`, `taylor_expand`, `partial_derivative`, `gradient`, `jacobian`, and `solve_ode` functions.
2. Symbolic differentiation handles polynomials, trig (sin, cos, tan), exponential, logarithmic, hyperbolic functions, and chain rule compositions. Nth-order derivatives work correctly.
3. Symbolic integration handles polynomials, trig, exponential, 1/x, and hyperbolic functions. Definite integrals with numeric bounds return simplified numeric results.
4. Limits correctly evaluate standard cases including `sin(x)/x -> 1`, limits at infinity, and one-sided limits (e.g., `1/x` from the right at 0 gives `+oo`).
5. Taylor expansion produces correct coefficients up to the requested order. Laurent series is produced when expanding around a pole (e.g., `1/x` around 0).
6. Numerical integration (Simpson's, Gauss-Legendre, adaptive) all produce accurate results. Gauss-Legendre is exact for polynomials up to degree 9.

### Advanced Features (7–10)
7. Partial derivatives, gradient, and Jacobian produce correct symbolic results for multivariate expressions.
8. The ODE solver handles first-order separable/linear ODEs and second-order constant-coefficient ODEs, returning general solutions with arbitrary constants.
9. The expression parser handles implicit multiplication (`2x` => `2*x`, `3sin(x)` => `3*sin(x)`) and caret notation (`x^2` => `x**2`).
10. The LRU cache correctly caches parsed expressions and demonstrates measurable speedup on repeated computations.

### Quality & Performance (11–15)
11. LaTeX output mode produces valid LaTeX for all operations (derivatives, integrals, limits, series, etc.).
12. The CLI (`python -m calccli`) supports all subcommands: `diff`, `integrate`, `numint`, `limit`, `taylor`, `partial`, `gradient`, `jacobian`, `ode`.
13. Invalid input (bad expressions, missing args, unknown operations) produces a helpful error message and non-zero exit code.
14. Differentiating a degree-50 polynomial completes in under 2 seconds.
15. All tests pass with `pytest`, including performance benchmarks.

## Test Commands
The following commands must all pass (exit code 0):

```bash
pip install sympy pytest -q

# --- Import checks ---
python -c "
from calccli.engine import differentiate, integrate, limit, taylor_expand
from calccli.numerical import numerical_integrate
from calccli.multivariable import partial_derivative, gradient, jacobian
from calccli.ode_solver import solve_ode
from calccli.parser import parse_expr
from calccli.cache import ExpressionCache
from calccli.latex_render import to_latex
print('all imports ok')
"

# --- Differentiation ---
python -c "
from calccli.engine import differentiate
assert '3*x**2 + 2' == str(differentiate('x^3 + 2*x', 'x'))
# nth-order derivative
assert '6*x' == str(differentiate('x^3 + 2*x', 'x', order=2))
assert '6' == str(differentiate('x^3 + 2*x', 'x', order=3))
# chain rule with trig
r = str(differentiate('sin(x^2)', 'x'))
assert '2*x*cos(x**2)' == r
print('differentiation ok')
"

# --- Integration ---
python -c "
from calccli.engine import integrate
r = str(integrate('x^2', 'x'))
assert 'x**3/3' in r
# definite integral
from sympy import Rational
r = integrate('x^2', 'x', lower=0, upper=1)
assert str(r) == '1/3'
print('integration ok')
"

# --- Numerical integration: Simpson ---
python -c "
from calccli.numerical import numerical_integrate
import math
v = numerical_integrate('x^2', 'x', 0, 1, method='simpson')
assert abs(v - 1/3) < 1e-6, f'simpson got {v}'
print('simpson ok')
"

# --- Numerical integration: Gauss-Legendre exactness ---
python -c "
from calccli.numerical import numerical_integrate
# Gauss-Legendre 5-point must be exact for polynomials up to degree 9
import math
# degree 9 polynomial: x^9 on [0,1] => exact = 0.1
v = numerical_integrate('x^9', 'x', 0, 1, method='gauss')
assert abs(v - 0.1) < 1e-13, f'gauss-legendre degree-9 got {v}, expected 0.1'
print('gauss-legendre ok')
"

# --- Numerical integration: adaptive accuracy ---
python -c "
from calccli.numerical import numerical_integrate
import math
v = numerical_integrate('sin(x)', 'x', 0, math.pi, method='adaptive')
assert abs(v - 2.0) < 1e-10, f'adaptive got {v}'
print('adaptive ok')
"

# --- Limits ---
python -c "
from calccli.engine import limit
from sympy import oo
assert str(limit('sin(x)/x', 'x', 0)) == '1'
# one-sided
assert str(limit('1/x', 'x', 0, side='right')) in ['oo', '+oo']
assert str(limit('1/x', 'x', 0, side='left')) in ['-oo']
# limit at infinity
assert str(limit('1/x', 'x', 'oo')) == '0'
print('limits ok')
"

# --- Taylor / Laurent series ---
python -c "
from calccli.engine import taylor_expand
r = str(taylor_expand('exp(x)', 'x', 0, 5))
assert 'x**2/2' in r
assert 'x**4/24' in r
# Laurent series around a pole
r = str(taylor_expand('1/x', 'x', 0, 3))
assert 'x**(-1)' in r or '1/x' in r
print('taylor/laurent ok')
"

# --- Multivariable ---
python -c "
from calccli.multivariable import partial_derivative, gradient, jacobian
# partial derivative
r = str(partial_derivative('x^2*y + y^3', 'x'))
assert '2*x*y' in r
# gradient
g = gradient('x^2*y + y^3', ['x', 'y'])
gs = [str(c) for c in g]
assert '2*x*y' in gs[0]
assert 'x**2 + 3*y**2' in gs[1]
# jacobian
J = jacobian(['x^2 + y', 'x*y'], ['x', 'y'])
assert str(J[0][0]) == '2*x'
assert str(J[0][1]) == '1'
assert str(J[1][0]) == 'y'
assert str(J[1][1]) == 'x'
print('multivariable ok')
"

# --- ODE solver ---
python -c "
from calccli.ode_solver import solve_ode
# first order: y' - 2y = 0 => C1*exp(2x)
r = str(solve_ode(\"y' - 2*y = 0\"))
assert 'exp(2*x)' in r or 'exp(2*x)' in r.replace(' ', '')
# second order: y'' + y = 0 => C1*sin(x) + C2*cos(x)
r = str(solve_ode(\"y'' + y = 0\"))
assert 'sin' in r and 'cos' in r
print('ode ok')
"

# --- Parser: implicit multiplication ---
python -c "
from calccli.parser import parse_expr
from sympy import symbols, sin
x = symbols('x')
# 2x => 2*x
assert parse_expr('2x') == 2*x
# 3sin(x) => 3*sin(x)
assert parse_expr('3sin(x)') == 3*sin(x)
# caret notation
assert parse_expr('x^2') == x**2
print('parser ok')
"

# --- Cache speedup ---
python -c "
import time
from calccli.cache import ExpressionCache
from calccli.engine import differentiate
cache = ExpressionCache(maxsize=128)
expr = 'x^10 + 3*x^7 + sin(x^3)*exp(x)'
# cold run
t0 = time.perf_counter()
for _ in range(100):
    r1 = cache.get_or_compute('diff', expr, lambda: differentiate(expr, 'x'))
cold = time.perf_counter() - t0
# warm run (all cached)
t0 = time.perf_counter()
for _ in range(100):
    r2 = cache.get_or_compute('diff', expr, lambda: differentiate(expr, 'x'))
warm = time.perf_counter() - t0
speedup = cold / warm if warm > 0 else float('inf')
assert speedup > 10, f'cache speedup only {speedup:.1f}x, expected >10x'
print(f'cache speedup: {speedup:.1f}x — ok')
"

# --- LaTeX output ---
python -c "
from calccli.latex_render import to_latex
from calccli.engine import differentiate, integrate
from sympy import sympify
r = differentiate('x^3', 'x')
ltx = to_latex(r)
assert 'x' in ltx and '{2}' in ltx  # 3x^{2}
r2 = integrate('1/x', 'x')
ltx2 = to_latex(r2)
assert 'log' in ltx2 or 'ln' in ltx2
print('latex ok')
"

# --- CLI: all subcommands ---
python -m calccli diff "x^3 + 2*x" x | grep -q "3\*x\*\*2 + 2"
python -m calccli diff "x^3" x --order 3 | grep -q "6"
python -m calccli integrate "sin(x)" x | grep -q "cos"
python -m calccli integrate "x^2" x --definite 0 1 | grep -q "1/3"
python -m calccli numint "x^2" x 0 1 --method simpson | python -c "import sys; v=float(sys.stdin.read().strip()); assert abs(v - 1/3) < 1e-4, f'got {v}'"
python -m calccli numint "x^9" x 0 1 --method gauss | python -c "import sys; v=float(sys.stdin.read().strip()); assert abs(v - 0.1) < 1e-12, f'got {v}'"
python -m calccli limit "sin(x)/x" x 0 | grep -q "1"
python -m calccli limit "1/x" x 0 --side right | grep -q "oo"
python -m calccli taylor "exp(x)" x 0 5 | grep -q "x\*\*2/2"
python -m calccli partial "x^2*y + y^3" x | grep -q "2\*x\*y"
python -m calccli gradient "x^2*y + y^3" x y | grep -q "2\*x\*y"
python -m calccli ode "y' - 2*y = 0" | grep -q "exp"
python -m calccli diff "x^3" x --latex | grep -q "x"

# --- CLI: error handling ---
python -m calccli diff 2>&1; test $? -ne 0
python -m calccli unknowncmd "x" x 2>&1; test $? -ne 0
python -m calccli diff "###invalid###" x 2>&1; test $? -ne 0

# --- Performance: degree-50 polynomial ---
python -c "
import time
from calccli.engine import differentiate
poly = ' + '.join(f'{i}*x^{i}' for i in range(1, 51))
t0 = time.perf_counter()
differentiate(poly, 'x')
elapsed = time.perf_counter() - t0
assert elapsed < 2.0, f'degree-50 diff took {elapsed:.2f}s, must be < 2s'
print(f'degree-50 diff: {elapsed:.3f}s — ok')
"

# --- Full test suite ---
pytest tests/ -v
```

## Notes
- All test commands must exit with code 0 for success.
- The evaluator will also check the qualitative criteria above.
- Performance tests are critical — the system must meet the stated speed and accuracy thresholds.
- Laurent series detection is a stretch goal but expected for full marks.
