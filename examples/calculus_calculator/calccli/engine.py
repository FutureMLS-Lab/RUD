"""Core symbolic calculus engine: differentiation, integration, limits, Taylor/Laurent series."""

from __future__ import annotations

from typing import Any

import sympy
from sympy import (
    Symbol, oo, series, O, Rational,
    diff as sym_diff,
    integrate as sym_integrate,
    limit as sym_limit,
)

from .parser import parse_expr


def differentiate(
    expr_str: str,
    var: str = "x",
    order: int = 1,
) -> sympy.Expr:
    """Compute the nth-order derivative of an expression.

    Args:
        expr_str: The expression to differentiate.
        var: The variable to differentiate with respect to.
        order: The order of the derivative (default 1).

    Returns:
        The derivative as a sympy expression.

    Raises:
        ValueError: If the expression cannot be parsed.
    """
    sym = Symbol(var)
    expr = parse_expr(expr_str)
    result = sym_diff(expr, sym, order)
    return result


def integrate(
    expr_str: str,
    var: str = "x",
    lower: Any = None,
    upper: Any = None,
) -> sympy.Expr:
    """Compute the indefinite or definite integral of an expression.

    Args:
        expr_str: The expression to integrate.
        var: The variable of integration.
        lower: Lower bound for definite integral (optional).
        upper: Upper bound for definite integral (optional).

    Returns:
        The integral as a sympy expression.
    """
    sym = Symbol(var)
    expr = parse_expr(expr_str)

    if lower is not None and upper is not None:
        # Parse bounds if they are strings
        if isinstance(lower, str):
            lower = parse_expr(lower)
        else:
            lower = sympy.sympify(lower)
        if isinstance(upper, str):
            upper = parse_expr(upper)
        else:
            upper = sympy.sympify(upper)
        result = sym_integrate(expr, (sym, lower, upper))
    else:
        result = sym_integrate(expr, sym)

    return result


def limit(
    expr_str: str,
    var: str = "x",
    point: Any = 0,
    side: str | None = None,
) -> sympy.Expr:
    """Evaluate the limit of an expression at a point.

    Args:
        expr_str: The expression.
        var: The variable.
        point: The point to evaluate the limit at. Can be a number, 'oo', '-oo'.
        side: 'left', 'right', or None for two-sided.

    Returns:
        The limit as a sympy expression.
    """
    sym = Symbol(var)
    expr = parse_expr(expr_str)

    # Handle string point values
    if isinstance(point, str):
        if point in ("oo", "+oo", "inf", "+inf"):
            point = oo
        elif point in ("-oo", "-inf"):
            point = -oo
        else:
            point = sympy.sympify(point)
    else:
        point = sympy.sympify(point)

    # Map side parameter to sympy's dir parameter
    if side == "right":
        direction = "+"
    elif side == "left":
        direction = "-"
    else:
        direction = "+"  # sympy default for two-sided

    if side is not None:
        result = sym_limit(expr, sym, point, dir=direction)
    else:
        result = sym_limit(expr, sym, point)

    return result


def taylor_expand(
    expr_str: str,
    var: str = "x",
    point: Any = 0,
    n: int = 6,
) -> sympy.Expr:
    """Expand a function as a Taylor or Laurent series.

    If the function has a pole at the expansion point, a Laurent series
    with negative-power terms is produced.

    Args:
        expr_str: The expression to expand.
        var: The variable.
        point: The point to expand around.
        n: Number of terms.

    Returns:
        The series expansion as a sympy expression (without O term).
    """
    sym = Symbol(var)
    expr = parse_expr(expr_str)

    if isinstance(point, str):
        point = sympy.sympify(point)
    else:
        point = sympy.sympify(point)

    # Use sympy series which automatically handles Laurent series for poles
    s = series(expr, sym, point, n=n)
    # Remove the O(...) term
    result = s.removeO()

    return result
