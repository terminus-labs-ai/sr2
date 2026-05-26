"""
Tests for PluginRegistry[T] and the plugin error hierarchy.

Covers:
- Happy path: get() returns the registered class; names() returns all names.
- PluginNotFoundError: unknown name, message lists available names.
- PluginCollisionError: two entry points with the same name, message names the distributions.
- Protocol validation: class missing required members raises a clear error.
- Lazy discovery: entry points scanned at most once per registry instance.
- Empty/absent group: graceful — names() returns [], get() raises PluginNotFoundError.
"""

from __future__ import annotations

import importlib.metadata
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, patch

import pytest

from sr2.plugins.errors import PluginCollisionError, PluginError, PluginNotFoundError
from sr2.plugins.registry import PluginRegistry


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@runtime_checkable
class FakeProtocol(Protocol):
    @classmethod
    def build(cls, config: dict, deps: dict) -> "FakeProtocol": ...


class ConformingClass:
    """Satisfies FakeProtocol."""

    @classmethod
    def build(cls, config: dict, deps: dict) -> "ConformingClass":
        return cls()


class NonConformingClass:
    """Does NOT satisfy FakeProtocol — missing build()."""
    pass


def _make_entry_point(name: str, cls: type, dist_name: str = "some-dist") -> MagicMock:
    """Build a mock importlib.metadata.EntryPoint."""
    ep = MagicMock(spec=importlib.metadata.EntryPoint)
    ep.name = name
    ep.load.return_value = cls
    dist = MagicMock()
    dist.name = dist_name
    ep.dist = dist
    return ep


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    def test_plugin_error_is_import_error(self):
        assert issubclass(PluginError, ImportError)

    def test_plugin_not_found_error_is_plugin_error(self):
        assert issubclass(PluginNotFoundError, PluginError)

    def test_plugin_collision_error_is_plugin_error(self):
        assert issubclass(PluginCollisionError, PluginError)

    def test_plugin_not_found_error_is_import_error(self):
        """Backward compat: callers catching ImportError still catch it."""
        assert issubclass(PluginNotFoundError, ImportError)

    def test_plugin_collision_error_is_import_error(self):
        assert issubclass(PluginCollisionError, ImportError)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestGetHappyPath:
    def test_get_returns_registered_class(self):
        ep = _make_entry_point("my_resolver", ConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            result = registry.get("my_resolver")
        assert result is ConformingClass
        ep.load.assert_called_once()

    def test_get_is_idempotent(self):
        ep = _make_entry_point("my_resolver", ConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            assert registry.get("my_resolver") is registry.get("my_resolver")

    def test_names_returns_all_registered_names(self):
        eps = [
            _make_entry_point("alpha", ConformingClass),
            _make_entry_point("beta", ConformingClass),
        ]
        with patch("sr2.plugins.registry.entry_points", return_value=eps):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            result = registry.names()
        assert sorted(result) == ["alpha", "beta"]

    def test_multiple_groups_are_independent(self):
        ep_a = _make_entry_point("thing", ConformingClass)
        ep_b = _make_entry_point("other", ConformingClass)

        def side_effect(group):
            if group == "sr2.resolvers":
                return [ep_a]
            if group == "sr2.stores":
                return [ep_b]
            return []

        with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
            reg_a = PluginRegistry("sr2.resolvers", FakeProtocol)
            reg_b = PluginRegistry("sr2.stores", FakeProtocol)
            assert reg_a.get("thing") is ConformingClass
            assert reg_b.get("other") is ConformingClass


# ---------------------------------------------------------------------------
# PluginNotFoundError
# ---------------------------------------------------------------------------

class TestPluginNotFoundError:
    def test_get_unknown_name_raises_plugin_not_found_error(self):
        ep = _make_entry_point("known", ConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginNotFoundError):
                registry.get("unknown")

    def test_error_message_lists_available_names(self):
        eps = [
            _make_entry_point("alpha", ConformingClass),
            _make_entry_point("beta", ConformingClass),
        ]
        with patch("sr2.plugins.registry.entry_points", return_value=eps):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginNotFoundError) as exc_info:
                registry.get("missing")
        message = str(exc_info.value)
        assert "alpha" in message
        assert "beta" in message

    def test_error_message_includes_requested_name(self):
        ep = _make_entry_point("alpha", ConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginNotFoundError) as exc_info:
                registry.get("missing_thing")
        assert "missing_thing" in str(exc_info.value)


# ---------------------------------------------------------------------------
# PluginCollisionError
# ---------------------------------------------------------------------------

class TestPluginCollisionError:
    def test_duplicate_name_raises_plugin_collision_error(self):
        ep1 = _make_entry_point("my_resolver", ConformingClass, dist_name="dist-a")
        ep2 = _make_entry_point("my_resolver", ConformingClass, dist_name="dist-b")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginCollisionError):
                registry.get("my_resolver")

    def test_collision_error_names_both_distributions(self):
        ep1 = _make_entry_point("shared", ConformingClass, dist_name="dist-alpha")
        ep2 = _make_entry_point("shared", ConformingClass, dist_name="dist-beta")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginCollisionError) as exc_info:
                registry.get("shared")
        message = str(exc_info.value)
        assert "dist-alpha" in message
        assert "dist-beta" in message

    def test_collision_error_names_the_conflicting_key(self):
        ep1 = _make_entry_point("shared", ConformingClass, dist_name="dist-alpha")
        ep2 = _make_entry_point("shared", ConformingClass, dist_name="dist-beta")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginCollisionError) as exc_info:
                registry.get("shared")
        assert "shared" in str(exc_info.value)

    def test_non_colliding_names_in_same_scan_still_resolve(self):
        ep1 = _make_entry_point("clash", ConformingClass, dist_name="dist-a")
        ep2 = _make_entry_point("clash", ConformingClass, dist_name="dist-b")
        ep3 = _make_entry_point("unique", ConformingClass, dist_name="dist-c")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2, ep3]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            result = registry.get("unique")
        assert result is ConformingClass


# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------

class TestProtocolValidation:
    def test_non_conforming_class_raises_error_not_not_found(self):
        ep = _make_entry_point("bad_plugin", NonConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(Exception) as exc_info:
                registry.get("bad_plugin")
        # Must NOT be PluginNotFoundError — the name was found, the class is wrong
        assert not isinstance(exc_info.value, PluginNotFoundError)

    def test_non_conforming_class_error_is_actionable(self):
        """The error message should reference the missing member or the class."""
        ep = _make_entry_point("bad_plugin", NonConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(Exception) as exc_info:
                registry.get("bad_plugin")
        message = str(exc_info.value)
        # Should mention the class or the missing method so the user knows what to fix
        assert "NonConformingClass" in message or "build" in message or "bad_plugin" in message

    def test_conforming_class_does_not_raise_on_get(self):
        ep = _make_entry_point("good_plugin", ConformingClass)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            # No exception expected
            result = registry.get("good_plugin")
        assert result is ConformingClass


# ---------------------------------------------------------------------------
# Lazy discovery
# ---------------------------------------------------------------------------

class TestLazyDiscovery:
    def test_entry_points_not_called_before_first_get_or_names(self):
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            _registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            mock_ep.assert_not_called()

    def test_entry_points_called_on_first_get(self):
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginNotFoundError):
                registry.get("anything")
            mock_ep.assert_called_once()

    def test_entry_points_called_on_names(self):
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            registry.names()
            mock_ep.assert_called_once()

    def test_entry_points_scanned_at_most_once_across_multiple_gets(self):
        ep = _make_entry_point("my_plugin", ConformingClass)
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = [ep]
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            registry.get("my_plugin")
            registry.get("my_plugin")
            with pytest.raises(PluginNotFoundError):
                registry.get("nonexistent")
        # Entry points should be scanned exactly once regardless of how many get() calls
        mock_ep.assert_called_once()

    def test_entry_points_scanned_at_most_once_when_names_then_get(self):
        ep = _make_entry_point("my_plugin", ConformingClass)
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = [ep]
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            registry.names()
            registry.get("my_plugin")
            registry.names()
        mock_ep.assert_called_once()

    def test_discovery_is_per_registry_instance(self):
        """Each registry instance has its own discovery cache."""
        ep = _make_entry_point("plugin", ConformingClass)
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = [ep]
            reg_a = PluginRegistry("sr2.resolvers", FakeProtocol)
            reg_b = PluginRegistry("sr2.resolvers", FakeProtocol)
            reg_a.get("plugin")
            reg_b.get("plugin")
        # Two registry instances → two discovery calls
        assert mock_ep.call_count == 2


# ---------------------------------------------------------------------------
# Empty / absent group
# ---------------------------------------------------------------------------

class TestEmptyGroup:
    def test_names_returns_empty_list_for_absent_group(self):
        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            result = registry.names()
        assert result == []

    def test_get_on_empty_group_raises_plugin_not_found_error(self):
        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginNotFoundError):
                registry.get("anything")

    def test_empty_group_is_not_an_import_error_subtype_mismatch(self):
        """Creating a registry for an empty group does not raise at construction."""
        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            # Construction must not raise
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            assert registry is not None

    def test_not_found_message_shows_empty_list_for_empty_group(self):
        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            registry = PluginRegistry("sr2.resolvers", FakeProtocol)
            with pytest.raises(PluginNotFoundError) as exc_info:
                registry.get("anything")
        # Available names are empty; message should not crash and should include the requested name
        message = str(exc_info.value)
        assert "anything" in message


# ---------------------------------------------------------------------------
# entry_points() call contract
# ---------------------------------------------------------------------------

class TestEntryPointsCallContract:
    def test_registry_passes_group_name_to_entry_points(self):
        """entry_points must be called with the correct group."""
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            registry = PluginRegistry("sr2.my_custom_group", FakeProtocol)
            registry.names()
        mock_ep.assert_called_once_with(group="sr2.my_custom_group")
