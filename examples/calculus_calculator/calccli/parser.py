"""Expression parsing with implicit multiplication and caret notation."""

from __future__ import annotations

import re
from sympy import sympify, Symbol
from sympy.parsing.sympy_parser import (
    parse_expr as sympy_parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


# Functions that should be recognized
_KNOWN_FUNCS = (
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "sinh", "cosh", "tanh", "asin", "acos", "atan", "abs",
)


def _preprocess(expr_str: str) -> str:
    """Preprocess expression string before sympy parsing.

    Handles:
    - Caret notation: x^2 -> x**2
    - Implicit multiplication before functions: 3sin(x) -> 3*sin(x)
    - Implicit multiplication: 2x -> 2*x
    """
    s = expr_str.strip()

    # Replace ^ with ** for exponentiation
    s = s.replace("^", "**")

    # Insert * before known function names when preceded by a digit or closing paren
    for fn in _KNOWN_FUNCS:
        # digit immediately before function name: 3sin(x) -> 3*sin(x)
        s = re.sub(rf'(\d)({fn}\s*\()', rf'\1*\2', s)
        # closing paren before function name: )sin(x) -> )*sin(x)
        s = re.sub(rf'(\))({fn}\s*\()', rf'\1*\2', s)
        # variable letter before function name: xsin(x) -> x*sin(x)
        s = re.sub(rf'([a-zA-Z])({fn}\s*\()', rf'\1*\2', s)

    return s


def parse_expr(expr_str: str, local_dict: dict | None = None) -> sympify:
    """Parse a mathematical expression string into a sympy expression.

    Supports:
    - Standard math notation: x^3 + 2*x, sin(x), exp(x), etc.
    - Implicit multiplication: 2x -> 2*x, 3sin(x) -> 3*sin(x)
    - Caret notation: x^2 -> x**2

    Args:
        expr_str: The expression string to parse.
        local_dict: Optional dictionary of local symbols.

    Returns:
        A sympy expression.

    Raises:
        ValueError: If the expression cannot be parsed.
    """
    preprocessed = _preprocess(expr_str)

    transformations = standard_transformations + (
        implicit_multiplication_application,
    )

    try:
        result = sympy_parse_expr(
            preprocessed,
            transformations=transformations,
            local_dict=local_dict or {},
        )
        return result
    except Exception as e:
        raise ValueError(f"Cannot parse expression: '{expr_str}' — {e}") from e
