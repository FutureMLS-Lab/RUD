"""Tests for limit evaluation."""

import pytest
from sympy import oo

from calccli.engine import limit


class TestLimits:
    def test_sin_x_over_x(self):
        assert str(limit("sin(x)/x", "x", 0)) == "1"

    def test_one_over_x_right(self):
        r = str(limit("1/x", "x", 0, side="right"))
        assert r in ["oo", "+oo"]

    def test_one_over_x_left(self):
        assert str(limit("1/x", "x", 0, side="left")) == "-oo"

    def test_at_infinity(self):
        assert str(limit("1/x", "x", "oo")) == "0"

    def test_at_negative_infinity(self):
        assert str(limit("1/x", "x", "-oo")) == "0"

    def test_polynomial_at_infinity(self):
        r = limit("x^2", "x", "oo")
        assert r == oo

    def test_exp_at_negative_infinity(self):
        assert str(limit("exp(x)", "x", "-oo")) == "0"

    def test_constant_limit(self):
        assert str(limit("5", "x", 3)) == "5"

    def test_indeterminate_form(self):
        # (x^2 - 1)/(x - 1) -> 2 as x -> 1
        assert str(limit("(x^2 - 1)/(x - 1)", "x", 1)) == "2"
