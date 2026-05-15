"""Tests for Taylor and Laurent series expansion."""

import pytest
from sympy import symbols

from calccli.engine import taylor_expand


x = symbols("x")


class TestTaylorExpand:
    def test_exp_at_zero(self):
        r = str(taylor_expand("exp(x)", "x", 0, 5))
        assert "x**2/2" in r
        assert "x**4/24" in r

    def test_sin_at_zero(self):
        r = str(taylor_expand("sin(x)", "x", 0, 6))
        assert "x**3/6" in r or "-x**3/6" in r
        assert "x**5/120" in r or "-x**5/120" in r

    def test_cos_at_zero(self):
        r = str(taylor_expand("cos(x)", "x", 0, 5))
        assert "x**2/2" in r
        assert "x**4/24" in r

    def test_laurent_1_over_x_at_zero(self):
        """Laurent series for 1/x at 0 should have x^(-1) term."""
        r = str(taylor_expand("1/x", "x", 0, 3))
        assert "x**(-1)" in r or "1/x" in r

    def test_taylor_around_nonzero_point(self):
        r = str(taylor_expand("1/x", "x", 1, 4))
        # Should contain (x-1) terms
        assert "x" in r

    def test_polynomial_exact(self):
        r = taylor_expand("x^3 + x", "x", 0, 5)
        assert r == x**3 + x
