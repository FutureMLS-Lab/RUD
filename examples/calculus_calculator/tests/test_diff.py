"""Tests for symbolic differentiation."""

import pytest
from sympy import symbols, cos, sin, exp, log, sqrt, cosh, sinh

from calccli.engine import differentiate


x = symbols("x")


class TestDifferentiate:
    def test_polynomial(self):
        assert str(differentiate("x^3 + 2*x", "x")) == "3*x**2 + 2"

    def test_polynomial_second_order(self):
        assert str(differentiate("x^3 + 2*x", "x", order=2)) == "6*x"

    def test_polynomial_third_order(self):
        assert str(differentiate("x^3 + 2*x", "x", order=3)) == "6"

    def test_polynomial_fourth_order_zero(self):
        assert str(differentiate("x^3 + 2*x", "x", order=4)) == "0"

    def test_sin(self):
        assert str(differentiate("sin(x)", "x")) == "cos(x)"

    def test_cos(self):
        assert str(differentiate("cos(x)", "x")) == "-sin(x)"

    def test_exp(self):
        assert str(differentiate("exp(x)", "x")) == "exp(x)"

    def test_log(self):
        assert str(differentiate("log(x)", "x")) == "1/x"

    def test_chain_rule_sin_x2(self):
        assert str(differentiate("sin(x^2)", "x")) == "2*x*cos(x**2)"

    def test_chain_rule_exp_sin(self):
        result = differentiate("exp(sin(x))", "x")
        assert result == exp(sin(x)) * cos(x)

    def test_product_rule(self):
        result = differentiate("sin(x)*exp(x)", "x")
        expected = sin(x) * exp(x) + exp(x) * cos(x)
        assert result.equals(expected)

    def test_product_rule_second_order(self):
        result = differentiate("sin(x)*exp(x)", "x", order=2)
        assert result.equals(2 * exp(x) * cos(x))

    def test_sqrt(self):
        result = differentiate("sqrt(x)", "x")
        assert result == 1 / (2 * sqrt(x))

    def test_hyperbolic_sinh(self):
        assert str(differentiate("sinh(x)", "x")) == "cosh(x)"

    def test_hyperbolic_cosh(self):
        assert str(differentiate("cosh(x)", "x")) == "sinh(x)"

    def test_constant(self):
        assert str(differentiate("5", "x")) == "0"

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            differentiate("###invalid###", "x")
