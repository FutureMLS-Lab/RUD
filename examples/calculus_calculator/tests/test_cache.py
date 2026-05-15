"""Tests for expression cache."""

import pytest

from calccli.cache import ExpressionCache


class TestExpressionCache:
    def test_cache_stores_result(self):
        cache = ExpressionCache(maxsize=10)
        result = cache.get_or_compute("diff", "x^2", lambda: "2*x")
        assert result == "2*x"

    def test_cache_returns_cached(self):
        cache = ExpressionCache(maxsize=10)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "2*x"

        cache.get_or_compute("diff", "x^2", compute)
        cache.get_or_compute("diff", "x^2", compute)
        assert call_count == 1

    def test_cache_different_operations(self):
        cache = ExpressionCache(maxsize=10)
        r1 = cache.get_or_compute("diff", "x^2", lambda: "2*x")
        r2 = cache.get_or_compute("integrate", "x^2", lambda: "x^3/3")
        assert r1 == "2*x"
        assert r2 == "x^3/3"

    def test_cache_eviction(self):
        cache = ExpressionCache(maxsize=2)
        cache.get_or_compute("a", "1", lambda: "r1")
        cache.get_or_compute("b", "2", lambda: "r2")
        cache.get_or_compute("c", "3", lambda: "r3")
        # First entry should be evicted
        assert len(cache) == 2

    def test_cache_canonicalization(self):
        cache = ExpressionCache(maxsize=10)
        r1 = cache.get_or_compute("diff", "x^2", lambda: "2*x")
        r2 = cache.get_or_compute("diff", " x^2 ", lambda: "WRONG")
        assert r2 == "2*x"  # Should use cached result

    def test_cache_clear(self):
        cache = ExpressionCache(maxsize=10)
        cache.get_or_compute("diff", "x^2", lambda: "2*x")
        cache.clear()
        assert len(cache) == 0
