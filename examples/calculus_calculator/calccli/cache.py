"""LRU expression cache with canonicalization."""

from __future__ import annotations

import functools
from typing import Any, Callable


class ExpressionCache:
    """LRU cache for computed symbolic results.

    Keys are canonicalized from (operation, expression_string) pairs.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._order: list[str] = []  # LRU order tracking

    @staticmethod
    def _canonicalize(operation: str, expr_str: str) -> str:
        """Canonicalize an expression for cache key generation.

        Strips whitespace and normalizes the expression string.
        """
        normalized = expr_str.strip().replace(" ", "")
        return f"{operation}:{normalized}"

    def get_or_compute(
        self, operation: str, expr_str: str, compute_fn: Callable[[], Any]
    ) -> Any:
        """Get a cached result or compute and cache it.

        Args:
            operation: The operation name (e.g., 'diff', 'integrate').
            expr_str: The expression string.
            compute_fn: A callable that computes the result if not cached.

        Returns:
            The computed or cached result.
        """
        key = self._canonicalize(operation, expr_str)

        if key in self._cache:
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]

        # Compute the result
        result = compute_fn()

        # Evict oldest if at capacity
        if len(self._cache) >= self._maxsize:
            oldest_key = self._order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = result
        self._order.append(key)
        return result

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._cache)
