"""Numerical integration methods: Simpson's rule, Gauss-Legendre, adaptive quadrature."""

from __future__ import annotations

import math
from typing import Callable

from sympy import Symbol, lambdify

from .parser import parse_expr


def _make_function(expr_str: str, var: str = "x") -> Callable[[float], float]:
    """Convert an expression string to a callable numerical function."""
    sym = Symbol(var)
    expr = parse_expr(expr_str)
    return lambdify(sym, expr, modules=["math"])


def _simpson(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Simpson's 1/3 rule for numerical integration.

    Args:
        f: The function to integrate.
        a: Lower bound.
        b: Upper bound.
        n: Number of subintervals (must be even).

    Returns:
        Approximate integral value.
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            result += 2 * f(x_i)
        else:
            result += 4 * f(x_i)

    return result * h / 3


# Gauss-Legendre 5-point quadrature nodes and weights on [-1, 1]
_GL5_NODES = [
    -math.sqrt(5 + 2 * math.sqrt(10 / 7)) / 3,
    -math.sqrt(5 - 2 * math.sqrt(10 / 7)) / 3,
    0.0,
    math.sqrt(5 - 2 * math.sqrt(10 / 7)) / 3,
    math.sqrt(5 + 2 * math.sqrt(10 / 7)) / 3,
]

_GL5_WEIGHTS = [
    (322 - 13 * math.sqrt(70)) / 900,
    (322 + 13 * math.sqrt(70)) / 900,
    128 / 225,
    (322 + 13 * math.sqrt(70)) / 900,
    (322 - 13 * math.sqrt(70)) / 900,
]


def _gauss_legendre(f: Callable[[float], float], a: float, b: float) -> float:
    """Gauss-Legendre 5-point quadrature.

    Exact for polynomials up to degree 2*5-1 = 9.

    Args:
        f: The function to integrate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Approximate integral value.
    """
    # Transform from [-1,1] to [a,b]: x = (b-a)/2 * t + (a+b)/2
    mid = (a + b) / 2
    half = (b - a) / 2

    result = 0.0
    for node, weight in zip(_GL5_NODES, _GL5_WEIGHTS):
        x = half * node + mid
        result += weight * f(x)

    return result * half


def _adaptive_quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_depth: int = 50,
) -> float:
    """Adaptive Simpson's quadrature.

    Recursively subdivides intervals until the desired tolerance is met.

    Args:
        f: The function to integrate.
        a: Lower bound.
        b: Upper bound.
        tol: Error tolerance.
        max_depth: Maximum recursion depth.

    Returns:
        Approximate integral value.
    """

    def _quad_recursive(
        a: float, b: float, fa: float, fb: float, whole: float,
        tol: float, depth: int,
    ) -> float:
        mid = (a + b) / 2
        fmid = f(mid)
        h = (b - a) / 2  # half-interval length

        left = h / 6 * (fa + 4 * f((a + mid) / 2) + fmid)
        right = h / 6 * (fmid + 4 * f((mid + b) / 2) + fb)
        combined = left + right

        if depth >= max_depth or abs(combined - whole) <= 15 * tol:
            return combined + (combined - whole) / 15

        return (
            _quad_recursive(a, mid, fa, fmid, left, tol / 2, depth + 1)
            + _quad_recursive(mid, b, fmid, fb, right, tol / 2, depth + 1)
        )

    fa = f(a)
    fb = f(b)
    mid = (a + b) / 2
    fmid = f(mid)
    whole = (b - a) / 6 * (fa + 4 * fmid + fb)

    return _quad_recursive(a, b, fa, fb, whole, tol, 0)


def numerical_integrate(
    expr_str: str,
    var: str = "x",
    a: float = 0.0,
    b: float = 1.0,
    method: str = "adaptive",
    n: int = 1000,
    tol: float = 1e-12,
) -> float:
    """Numerically integrate an expression over [a, b].

    Args:
        expr_str: The expression to integrate.
        var: The variable of integration.
        a: Lower bound.
        b: Upper bound.
        method: Integration method — 'simpson', 'gauss', or 'adaptive' (default).
        n: Number of subintervals for Simpson's rule.
        tol: Tolerance for adaptive quadrature.

    Returns:
        The approximate integral value.

    Raises:
        ValueError: If an unknown method is specified.
    """
    f = _make_function(expr_str, var)
    a = float(a)
    b = float(b)

    if method == "simpson":
        return _simpson(f, a, b, n)
    elif method == "gauss":
        return _gauss_legendre(f, a, b)
    elif method == "adaptive":
        return _adaptive_quad(f, a, b, tol)
    else:
        raise ValueError(f"Unknown integration method: '{method}'. Use 'simpson', 'gauss', or 'adaptive'.")
