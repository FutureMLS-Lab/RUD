"""Tests for multivariable calculus operations."""

import pytest
from sympy import symbols

from calccli.multivariable import partial_derivative, gradient, jacobian


x, y, z = symbols("x y z")


class TestPartialDerivative:
    def test_basic(self):
        r = str(partial_derivative("x^2*y + y^3", "x"))
        assert "2*x*y" in r

    def test_with_respect_to_y(self):
        r = partial_derivative("x^2*y + y^3", "y")
        assert r == x**2 + 3 * y**2

    def test_second_order(self):
        r = partial_derivative("x^2*y + y^3", "y", order=2)
        assert r == 6 * y

    def test_constant_with_respect_to_other_var(self):
        r = partial_derivative("x^2", "y")
        assert str(r) == "0"


class TestGradient:
    def test_basic(self):
        g = gradient("x^2*y + y^3", ["x", "y"])
        gs = [str(c) for c in g]
        assert "2*x*y" in gs[0]
        assert "x**2 + 3*y**2" in gs[1]

    def test_three_variables(self):
        g = gradient("x*y*z", ["x", "y", "z"])
        assert g[0] == y * z
        assert g[1] == x * z
        assert g[2] == x * y


class TestJacobian:
    def test_basic(self):
        J = jacobian(["x^2 + y", "x*y"], ["x", "y"])
        assert str(J[0][0]) == "2*x"
        assert str(J[0][1]) == "1"
        assert str(J[1][0]) == "y"
        assert str(J[1][1]) == "x"

    def test_single_function(self):
        J = jacobian(["x^2 + y^2"], ["x", "y"])
        assert str(J[0][0]) == "2*x"
        assert str(J[0][1]) == "2*y"
