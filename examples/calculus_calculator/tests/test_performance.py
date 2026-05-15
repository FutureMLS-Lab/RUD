"""Performance benchmark tests."""

import time
import math
import pytest

from calccli.engine import differentiate
from calccli.numerical import numerical_integrate
from calccli.cache import ExpressionCache


class TestPerformance:
    def test_degree_50_polynomial_diff(self):
        """Differentiating a degree-50 polynomial must complete in < 2 seconds."""
        poly = " + ".join(f"{i}*x^{i}" for i in range(1, 51))
        t0 = time.perf_counter()
        differentiate(poly, "x")
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"degree-50 diff took {elapsed:.2f}s, must be < 2s"

    def test_adaptive_accuracy(self):
        """Adaptive quadrature of sin(x) over [0, pi] accurate to 1e-10."""
        v = numerical_integrate("sin(x)", "x", 0, math.pi, method="adaptive")
        assert abs(v - 2.0) < 1e-10, f"adaptive got {v}"

    def test_adaptive_speed(self):
        """Adaptive quadrature of sin(x) over [0, pi] in < 1 second."""
        t0 = time.perf_counter()
        numerical_integrate("sin(x)", "x", 0, math.pi, method="adaptive")
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"adaptive took {elapsed:.2f}s, must be < 1s"

    def test_gauss_legendre_exactness(self):
        """Gauss-Legendre 5-point exact for polynomials up to degree 9."""
        v = numerical_integrate("x^9", "x", 0, 1, method="gauss")
        assert abs(v - 0.1) < 1e-14, f"gauss got {v}"

    def test_cache_speedup(self):
        """LRU cache must demonstrate at least 10x speedup on repeated computations."""
        cache = ExpressionCache(maxsize=128)
        expr = "x^10 + 3*x^7 + sin(x^3)*exp(x)"

        # Cold run
        t0 = time.perf_counter()
        for _ in range(100):
            cache.get_or_compute("diff", expr, lambda: differentiate(expr, "x"))
        cold = time.perf_counter() - t0

        # Warm run (all cached)
        t0 = time.perf_counter()
        for _ in range(100):
            cache.get_or_compute("diff", expr, lambda: differentiate(expr, "x"))
        warm = time.perf_counter() - t0

        speedup = cold / warm if warm > 0 else float("inf")
        assert speedup > 10, f"cache speedup only {speedup:.1f}x, expected >10x"
