"""Fallback content provider for degraded pipeline layers.

Modes:
  'none'   — always return None (layer is skipped).
  'static' — return a configured static string for any provider.
  'cached' — return the last known content for the named provider,
             or None if nothing has been cached yet.
"""

from __future__ import annotations

from typing import Optional

_VALID_MODES = frozenset({"none", "static", "cached"})


class FallbackProvider:
    """Selects fallback content based on the configured mode.

    Args:
        mode:         One of 'none', 'static', or 'cached'.
        static_value: Required when mode == 'static'. Ignored otherwise.
    """

    def __init__(self, mode: str, static_value: str | None = None) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown fallback mode {mode!r}. Expected one of {sorted(_VALID_MODES)}."
            )
        self._mode = mode
        self._static_value = static_value
        self._cache: dict[str, str] = {}

    def get_fallback(self, provider_name: str) -> str | None:
        """Return fallback content for *provider_name*, or None."""
        if self._mode == "none":
            return None
        if self._mode == "static":
            return self._static_value
        # cached
        return self._cache.get(provider_name)

    def update_cache(self, provider_name: str, content: str) -> None:
        """Store *content* as the cached fallback for *provider_name*.

        Only meaningful when mode == 'cached', but safe to call in any mode.
        """
        self._cache[provider_name] = content
