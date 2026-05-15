"""Multivariable calculus: partial derivatives, gradient, Jacobian."""

from __future__ import annotations

from typing import Sequence

import sympy
from sympy import Symbol, diff as sym_diff

from .parser import parse_expr


def partial_derivative(
    expr_str: str,
    var: str,
    order: int = 1,
) -> sympy.Expr:
    """Compute the partial derivative of an expression with respect to a variable.

    Args:
        expr_str: The expression.
        var: The variable to differentiate with respect to.
        order: The order of the derivative.

    Returns:
        The partial derivative as a sympy expression.
    """
    sym = Symbol(var)
    expr = parse_expr(expr_str)
    return sym_diff(expr, sym, order)


def gradient(
    expr_str: str,
    variables: Sequence[str],
) -> list[sympy.Expr]:
    """Compute the gradient vector of a scalar expression.

    Args:
        expr_str: The scalar expression.
        variables: List of variable names.

    Returns:
        A list of partial derivatives (the gradient components).
    """
    expr = parse_expr(expr_str)
    result = []
    for var in variables:
        sym = Symbol(var)
        result.append(sym_diff(expr, sym))
    return result


def jacobian(
    expr_strs: Sequence[str],
    variables: Sequence[str],
) -> list[list[sympy.Expr]]:
    """Compute the Jacobian matrix of a vector-valued function.

    Args:
        expr_strs: List of expression strings (the component functions).
        variables: List of variable names.

    Returns:
        A 2D list representing the Jacobian matrix.
        J[i][j] = d(expr_i)/d(var_j)
    """
    result = []
    for expr_str in expr_strs:
        expr = parse_expr(expr_str)
        row = []
        for var in variables:
            sym = Symbol(var)
            row.append(sym_diff(expr, sym))
        result.append(row)
    return result
