"""CLI argument parsing and dispatch for calccli."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .engine import differentiate, integrate, limit, taylor_expand
from .numerical import numerical_integrate
from .multivariable import partial_derivative, gradient, jacobian
from .ode_solver import solve_ode
from .latex_render import to_latex


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="calccli",
        description="Calculus calculator — symbolic and numerical operations.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # diff
    p_diff = subparsers.add_parser("diff", help="Differentiate an expression")
    p_diff.add_argument("expression", help="The expression to differentiate")
    p_diff.add_argument("variable", help="The variable")
    p_diff.add_argument("--order", type=int, default=1, help="Order of derivative")
    p_diff.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # integrate
    p_int = subparsers.add_parser("integrate", help="Integrate an expression")
    p_int.add_argument("expression", help="The expression to integrate")
    p_int.add_argument("variable", help="The variable")
    p_int.add_argument("--definite", nargs=2, metavar=("LOWER", "UPPER"), help="Definite integral bounds")
    p_int.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # numint
    p_numint = subparsers.add_parser("numint", help="Numerical integration")
    p_numint.add_argument("expression", help="The expression to integrate")
    p_numint.add_argument("variable", help="The variable")
    p_numint.add_argument("a", type=float, help="Lower bound")
    p_numint.add_argument("b", type=float, help="Upper bound")
    p_numint.add_argument("--method", default="adaptive", choices=["simpson", "gauss", "adaptive"])
    p_numint.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # limit
    p_limit = subparsers.add_parser("limit", help="Evaluate a limit")
    p_limit.add_argument("expression", help="The expression")
    p_limit.add_argument("variable", help="The variable")
    p_limit.add_argument("point", help="The point (number, oo, -oo)")
    p_limit.add_argument("--side", choices=["left", "right"], help="One-sided limit")
    p_limit.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # taylor
    p_taylor = subparsers.add_parser("taylor", help="Taylor/Laurent series expansion")
    p_taylor.add_argument("expression", help="The expression")
    p_taylor.add_argument("variable", help="The variable")
    p_taylor.add_argument("point", help="The expansion point")
    p_taylor.add_argument("n", type=int, help="Number of terms")
    p_taylor.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # partial
    p_partial = subparsers.add_parser("partial", help="Partial derivative")
    p_partial.add_argument("expression", help="The expression")
    p_partial.add_argument("variable", help="The variable")
    p_partial.add_argument("--order", type=int, default=1, help="Order")
    p_partial.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # gradient
    p_grad = subparsers.add_parser("gradient", help="Gradient vector")
    p_grad.add_argument("expression", help="The expression")
    p_grad.add_argument("variables", nargs="+", help="The variables")
    p_grad.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # jacobian
    p_jac = subparsers.add_parser("jacobian", help="Jacobian matrix")
    p_jac.add_argument("expressions", help="Comma-separated expressions")
    p_jac.add_argument("variables", help="Comma-separated variable names")
    p_jac.add_argument("--latex", action="store_true", help="Output in LaTeX")

    # ode
    p_ode = subparsers.add_parser("ode", help="Solve an ODE")
    p_ode.add_argument("equation", help="The ODE (e.g., \"y' - 2*y = 0\")")
    p_ode.add_argument("--latex", action="store_true", help="Output in LaTeX")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        result = _dispatch(args)
        use_latex = getattr(args, "latex", False)

        if use_latex:
            if isinstance(result, list):
                print(to_latex(result))
            elif isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
                print(to_latex(result))
            else:
                print(to_latex(result))
        else:
            if isinstance(result, list):
                print(result)
            else:
                print(result)

        return 0

    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace):
    """Dispatch to the appropriate function based on the command."""
    cmd = args.command

    if cmd == "diff":
        return differentiate(args.expression, args.variable, order=args.order)

    elif cmd == "integrate":
        if args.definite:
            lower, upper = args.definite
            return integrate(args.expression, args.variable, lower=lower, upper=upper)
        return integrate(args.expression, args.variable)

    elif cmd == "numint":
        return numerical_integrate(
            args.expression, args.variable, args.a, args.b, method=args.method,
        )

    elif cmd == "limit":
        return limit(args.expression, args.variable, args.point, side=args.side)

    elif cmd == "taylor":
        return taylor_expand(args.expression, args.variable, args.point, n=args.n)

    elif cmd == "partial":
        return partial_derivative(args.expression, args.variable, order=args.order)

    elif cmd == "gradient":
        return gradient(args.expression, args.variables)

    elif cmd == "jacobian":
        # Parse comma-separated expressions and variables
        expr_list = [e.strip() for e in args.expressions.split(",")]
        var_list = [v.strip() for v in args.variables.split(",")]
        return jacobian(expr_list, var_list)

    elif cmd == "ode":
        return solve_ode(args.equation)

    else:
        raise ValueError(f"Unknown command: '{cmd}'")
