"""Tests for ODE solver."""

import pytest

from calccli.ode_solver import solve_ode


class TestODESolver:
    def test_first_order_exponential(self):
        r = str(solve_ode("y' - 2*y = 0"))
        assert "exp(2*x)" in r or "exp(2*x)" in r.replace(" ", "")

    def test_second_order_harmonic(self):
        r = str(solve_ode("y'' + y = 0"))
        assert "sin" in r and "cos" in r

    def test_first_order_linear(self):
        r = str(solve_ode("y' + y = 0"))
        assert "exp" in r

    def test_second_order_real_roots(self):
        # y'' - 3y' + 2y = 0 => C1*exp(x) + C2*exp(2*x)
        r = str(solve_ode("y'' - 3*y' + 2*y = 0"))
        assert "exp" in r

    def test_invalid_ode(self):
        with pytest.raises(ValueError):
            solve_ode("###invalid###")
