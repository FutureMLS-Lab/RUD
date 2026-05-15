"""Tests for CLI interface."""

import pytest
import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run calccli CLI with given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "calccli", *args],
        capture_output=True,
        text=True,
    )


class TestCLIDiff:
    def test_basic_diff(self):
        r = run_cli("diff", "x^3 + 2*x", "x")
        assert r.returncode == 0
        assert "3*x**2 + 2" in r.stdout

    def test_nth_order(self):
        r = run_cli("diff", "x^3", "x", "--order", "3")
        assert r.returncode == 0
        assert "6" in r.stdout

    def test_latex(self):
        r = run_cli("diff", "x^3", "x", "--latex")
        assert r.returncode == 0
        assert "x" in r.stdout


class TestCLIIntegrate:
    def test_indefinite(self):
        r = run_cli("integrate", "sin(x)", "x")
        assert r.returncode == 0
        assert "cos" in r.stdout

    def test_definite(self):
        r = run_cli("integrate", "x^2", "x", "--definite", "0", "1")
        assert r.returncode == 0
        assert "1/3" in r.stdout


class TestCLINumint:
    def test_simpson(self):
        r = run_cli("numint", "x^2", "x", "0", "1", "--method", "simpson")
        assert r.returncode == 0
        v = float(r.stdout.strip())
        assert abs(v - 1 / 3) < 1e-4

    def test_gauss(self):
        r = run_cli("numint", "x^9", "x", "0", "1", "--method", "gauss")
        assert r.returncode == 0
        v = float(r.stdout.strip())
        assert abs(v - 0.1) < 1e-12


class TestCLILimit:
    def test_basic(self):
        r = run_cli("limit", "sin(x)/x", "x", "0")
        assert r.returncode == 0
        assert "1" in r.stdout

    def test_one_sided(self):
        r = run_cli("limit", "1/x", "x", "0", "--side", "right")
        assert r.returncode == 0
        assert "oo" in r.stdout


class TestCLITaylor:
    def test_basic(self):
        r = run_cli("taylor", "exp(x)", "x", "0", "5")
        assert r.returncode == 0
        assert "x**2/2" in r.stdout


class TestCLIMultivariable:
    def test_partial(self):
        r = run_cli("partial", "x^2*y + y^3", "x")
        assert r.returncode == 0
        assert "2*x*y" in r.stdout

    def test_gradient(self):
        r = run_cli("gradient", "x^2*y + y^3", "x", "y")
        assert r.returncode == 0
        assert "2*x*y" in r.stdout


class TestCLIODE:
    def test_first_order(self):
        r = run_cli("ode", "y' - 2*y = 0")
        assert r.returncode == 0
        assert "exp" in r.stdout


class TestCLIErrors:
    def test_no_args(self):
        r = run_cli("diff")
        assert r.returncode != 0

    def test_unknown_command(self):
        r = run_cli("unknowncmd", "x", "x")
        assert r.returncode != 0

    def test_invalid_expression(self):
        r = run_cli("diff", "###invalid###", "x")
        assert r.returncode != 0
