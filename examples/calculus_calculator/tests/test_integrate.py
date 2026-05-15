"""Tests for symbolic integration."""

import pytest
from sympy import symbols, Rational, cos, sin, exp, log

from calccli.engine import integrate


x = symbols("x")


class TestIntegrate:
    def test_polynomial(self):
        r = str(integrate("x^2", "x"))
        assert "x**3/3" in r

    def test_definite_integral(self):
        r = integrate("x^2", "x", lower=0, upper=1)
        assert str(r) == "1/3"

    def test_sin(self):
        r = integrate("sin(x)", "x")
        assert r == -cos(x)

    def test_cos(self):
        r = integrate("cos(x)", "x")
        assert r == sin(x)

    def test_exp(self):
        r = integrate("exp(x)", "x")
        assert r == exp(x)

    def test_one_over_x(self):
        r = integrate("1/x", "x")
        assert r == log(x)

    def test_definite_sin_0_pi(self):
        import math
        r = integrate("sin(x)", "x", lower=0, upper="pi")
        assert str(r) == "2"

    def test_polynomial_definite(self):
        r = integrate("x^3", "x", lower=0, upper=2)
        assert str(r) == "4"

    def test_constant(self):
        r = integrate("5", "x")
        assert r == 5 * x

    def test_hyperbolic_sinh(self):
        from sympy import sinh, cosh
        r = integrate("sinh(x)", "x")
        assert r == cosh(x)
