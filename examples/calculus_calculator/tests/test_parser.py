"""Tests for expression parser."""

import pytest
from sympy import symbols, sin, cos, exp, sqrt

from calccli.parser import parse_expr


x, y = symbols("x y")


class TestParser:
    def test_implicit_multiplication_digit_var(self):
        assert parse_expr("2x") == 2 * x

    def test_implicit_multiplication_digit_func(self):
        assert parse_expr("3sin(x)") == 3 * sin(x)

    def test_caret_notation(self):
        assert parse_expr("x^2") == x**2

    def test_standard_notation(self):
        assert parse_expr("x**2") == x**2

    def test_complex_expression(self):
        result = parse_expr("x^3 + 2*x + 1")
        assert result == x**3 + 2 * x + 1

    def test_trig_functions(self):
        assert parse_expr("sin(x)") == sin(x)
        assert parse_expr("cos(x)") == cos(x)

    def test_exp_function(self):
        assert parse_expr("exp(x)") == exp(x)

    def test_sqrt_function(self):
        assert parse_expr("sqrt(x)") == sqrt(x)

    def test_multivariate(self):
        assert parse_expr("x^2*y") == x**2 * y

    def test_invalid_expression(self):
        with pytest.raises(ValueError, match="Cannot parse expression"):
            parse_expr("###invalid###")
