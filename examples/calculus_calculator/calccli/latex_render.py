"""LaTeX output formatting for symbolic expressions."""

from __future__ import annotations

from typing import Any

import sympy
from sympy import latex as sym_latex


def to_latex(expr: Any) -> str:
    """Convert a sympy expression to LaTeX notation.

    Args:
        expr: A sympy expression, equation, or list of expressions.

    Returns:
        A LaTeX string representation.
    """
    if isinstance(expr, list):
        # Handle lists (e.g., gradient vectors)
        items = [sym_latex(e) for e in expr]
        return r"\left[" + ", ".join(items) + r"\right]"

    if isinstance(expr, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in expr):
        # Handle matrices (e.g., Jacobian)
        rows = []
        for row in expr:
            rows.append(" & ".join(sym_latex(e) for e in row))
        return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"

    return sym_latex(expr)
