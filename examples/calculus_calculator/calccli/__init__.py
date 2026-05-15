"""calccli — A calculus calculator library with CLI interface."""

from .engine import differentiate, integrate, limit, taylor_expand
from .numerical import numerical_integrate
from .multivariable import partial_derivative, gradient, jacobian
from .ode_solver import solve_ode
from .parser import parse_expr
from .cache import ExpressionCache
from .latex_render import to_latex

__all__ = [
    "differentiate",
    "integrate",
    "limit",
    "taylor_expand",
    "numerical_integrate",
    "partial_derivative",
    "gradient",
    "jacobian",
    "solve_ode",
    "parse_expr",
    "ExpressionCache",
    "to_latex",
]
