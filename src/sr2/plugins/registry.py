"""Generic, lazy-discovery plugin registry for SR2.

Each registry instance manages one entry-point group. Discovery is deferred
until the first call to ``get()`` or ``names()``, and is performed at most
once per instance.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Generic, TypeVar

from sr2.plugins.errors import PluginCollisionError, PluginNotFoundError

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Lazy entry-point registry for plugin types.

    Parameters
    ----------
    group:
        The entry-point group to scan (e.g. ``"sr2.stores"``).
    protocol:
        The runtime-checkable Protocol class that discovered classes must satisfy.
    """

    def __init__(self, group: str, protocol: type | None = None) -> None:
        self._group = group
        self._protocol = protocol

        # Populated on first _discover() call
        self._discovered: bool = False
        self._classes: dict[str, type] = {}          # name -> class (non-colliding)
        self._collisions: dict[str, list[str]] = {}  # name -> [dist, dist, ...]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> type[T]:
        """Return the class registered under *name*.

        Raises
        ------
        PluginCollisionError
            If two distributions register the same name.
        PluginNotFoundError
            If no entry point with *name* exists.
        TypeError
            If the loaded class does not satisfy the registry's protocol.
        """
        self._ensure_discovered()

        if name in self._collisions:
            raise PluginCollisionError(name, self._collisions[name])

        if name not in self._classes:
            raise PluginNotFoundError(name, list(self._classes.keys()))

        cls = self._classes[name]
        self._validate_protocol(name, cls)
        return cls  # type: ignore[return-value]

    def names(self) -> list[str]:
        """Return all non-colliding plugin names available in this group."""
        self._ensure_discovered()
        return list(self._classes.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_discovered(self) -> None:
        if not self._discovered:
            self._discover()
            self._discovered = True

    def _discover(self) -> None:
        """Scan the entry-point group and populate internal caches."""
        eps = entry_points(group=self._group)

        # Map name -> list[EntryPoint] to detect collisions
        grouped: dict[str, list] = {}
        for ep in eps:
            grouped.setdefault(ep.name, []).append(ep)

        for name, eps_for_name in grouped.items():
            if len(eps_for_name) > 1:
                dist_names = [ep.dist.name for ep in eps_for_name]
                self._collisions[name] = dist_names
            else:
                self._classes[name] = eps_for_name[0].load()

    def _validate_protocol(self, name: str, cls: type) -> None:
        """Raise TypeError if *cls* does not satisfy the registry protocol."""
        if self._protocol is None:
            return
        if not isinstance(cls, self._protocol):
            # Identify what's missing for a helpful message
            missing = []
            for attr in getattr(self._protocol, "__protocol_attrs__", []):
                if not hasattr(cls, attr):
                    missing.append(attr)

            if missing:
                detail = f"missing required member(s): {', '.join(missing)}"
            else:
                detail = f"does not satisfy protocol {self._protocol.__name__!r}"

            raise TypeError(
                f"Plugin {name!r} loaded class {cls.__name__!r} which {detail}."
            )
