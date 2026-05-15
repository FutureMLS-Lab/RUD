"""Tests for LaTeX output."""

import pytest
from sympy import symbols, sympify

from calccli.latex_render import to_latex
from calccli.engine import differentiate, integrate


x = symbols("x")


class TestLatex:
    def test_derivative_latex(self):
        r = differentiate("x^3", "x")
        ltx = to_latex(r)
        assert "x" in ltx and "{2}" in ltx  # 3x^{2}

    def test_integral_log_latex(self):
        r = integrate("1/x", "x")
        ltx = to_latex(r)
        assert "log" in ltx or "ln" in ltx

    def test_list_latex(self):
        from calccli.multivariable import gradient
        g = gradient("x^2 + y^2", ["x", "y"])
        ltx = to_latex(g)
        assert r"\left[" in ltx
        assert "2" in ltx

    def test_simple_expression(self):
        ltx = to_latex(sympify("x**2 + 1"))
        assert "x" in ltx

    def test_fraction(self):
        ltx = to_latex(sympify("1/x"))
        assert "frac" in ltx or "x" in ltx
