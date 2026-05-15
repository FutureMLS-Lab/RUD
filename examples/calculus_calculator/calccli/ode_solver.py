"""Symbolic ODE solver for first-order and second-order ODEs."""

from __future__ import annotations

import re

import sympy
from sympy import (
    Symbol, Function, Eq, dsolve, sympify,
    Derivative,
)


def _parse_ode(ode_str: str) -> tuple[Eq, Function, Symbol]:
    """Parse an ODE string into a sympy equation.

    Supports notation like:
    - "y' - 2*y = 0"
    - "y'' + y = 0"
    - "y' + x*y = x"

    Args:
        ode_str: The ODE as a string.

    Returns:
        Tuple of (equation, dependent function, independent variable).
    """
    x = Symbol("x")
    y = Function("y")

    s = ode_str.strip()

    # Split on '='
    if "=" in s:
        lhs_str, rhs_str = s.split("=", 1)
    else:
        lhs_str = s
        rhs_str = "0"

    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()

    def _replace_derivatives(s: str) -> str:
        """Replace y'', y' with sympy Derivative notation."""
        # Replace y'' with Derivative(y(x), x, x)
        s = re.sub(r"y\s*''", "Derivative(y(x), x, x)", s)
        # Replace y' with Derivative(y(x), x)
        s = re.sub(r"y\s*'", "Derivative(y(x), x)", s)
        # Replace standalone y (not part of y(x)) with y(x)
        # Avoid replacing y inside Derivative(y(x)...) or y(x)
        s = re.sub(r'(?<!\w)y(?!\s*[\(\'])', "y(x)", s)
        return s

    lhs_str = _replace_derivatives(lhs_str)
    rhs_str = _replace_derivatives(rhs_str)

    # Replace ^ with **
    lhs_str = lhs_str.replace("^", "**")
    rhs_str = rhs_str.replace("^", "**")

    local_dict = {"x": x, "y": y, "Derivative": Derivative}

    lhs_expr = sympify(lhs_str, locals=local_dict)
    rhs_expr = sympify(rhs_str, locals=local_dict)

    eq = Eq(lhs_expr, rhs_expr)
    return eq, y, x


def solve_ode(ode_str: str) -> sympy.Expr:
    """Solve an ordinary differential equation symbolically.

    Supports first-order (separable, linear, exact) and second-order
    (constant-coefficient) ODEs.

    Args:
        ode_str: The ODE as a string (e.g., "y' - 2*y = 0").

    Returns:
        The general solution as a sympy expression or Eq.

    Raises:
        ValueError: If the ODE cannot be parsed or solved.
    """
    try:
        eq, y, x = _parse_ode(ode_str)
        sol = dsolve(eq, y(x))
        return sol
    except Exception as e:
        raise ValueError(f"Cannot solve ODE: '{ode_str}' — {e}") from e
