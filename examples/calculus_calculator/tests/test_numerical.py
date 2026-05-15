"""Tests for numerical integration methods."""

import math
import pytest

from calccli.numerical import numerical_integrate


class TestSimpson:
    def test_x_squared(self):
        v = numerical_integrate("x^2", "x", 0, 1, method="simpson")
        assert abs(v - 1 / 3) < 1e-6

    def test_sin(self):
        v = numerical_integrate("sin(x)", "x", 0, math.pi, method="simpson")
        assert abs(v - 2.0) < 1e-6

    def test_constant(self):
        v = numerical_integrate("5", "x", 0, 2, method="simpson")
        assert abs(v - 10.0) < 1e-10

    def test_linear(self):
        v = numerical_integrate("x", "x", 0, 1, method="simpson")
        assert abs(v - 0.5) < 1e-10


class TestGaussLegendre:
    def test_polynomial_degree_1(self):
        v = numerical_integrate("x", "x", 0, 1, method="gauss")
        assert abs(v - 0.5) < 1e-14

    def test_polynomial_degree_2(self):
        v = numerical_integrate("x^2", "x", 0, 1, method="gauss")
        assert abs(v - 1 / 3) < 1e-14

    def test_polynomial_degree_5(self):
        v = numerical_integrate("x^5", "x", 0, 1, method="gauss")
        assert abs(v - 1 / 6) < 1e-14

    def test_polynomial_degree_9(self):
        """Gauss-Legendre 5-point must be exact for polynomials up to degree 9."""
        v = numerical_integrate("x^9", "x", 0, 1, method="gauss")
        assert abs(v - 0.1) < 1e-13

    def test_constant(self):
        v = numerical_integrate("3", "x", 0, 2, method="gauss")
        assert abs(v - 6.0) < 1e-14


class TestAdaptive:
    def test_sin_accuracy(self):
        """Adaptive quadrature of sin(x) over [0, pi] must be accurate to 1e-10."""
        v = numerical_integrate("sin(x)", "x", 0, math.pi, method="adaptive")
        assert abs(v - 2.0) < 1e-10

    def test_exp(self):
        v = numerical_integrate("exp(x)", "x", 0, 1, method="adaptive")
        assert abs(v - (math.e - 1)) < 1e-10

    def test_gaussian(self):
        v = numerical_integrate("exp(-x^2)", "x", 0, 3, method="adaptive")
        assert abs(v - 0.8862073482595214) < 1e-6


class TestInvalidMethod:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown integration method"):
            numerical_integrate("x", "x", 0, 1, method="unknown")
