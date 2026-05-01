"""SR2 v2 error hierarchy.

All SR2 errors inherit from SR2Error. Application-level errors vs.
internal errors are separated so callers can catch the right granularity.
"""

from __future__ import annotations


class SR2Error(Exception):
    """Base exception for all SR2 errors."""


class ConfigError(SR2Error):
    """Invalid or missing configuration."""


class PluginError(SR2Error):
    """Plugin lifecycle error (discovery, activation, licensing)."""


class PluginNotFoundError(PluginError):
    """Requested plugin not found in entry points."""


class PluginLicenseError(PluginError):
    """Paid plugin activated without valid license."""


class PipelineError(SR2Error):
    """Pipeline execution error (budget exceeded, resolution failure)."""


class ProviderError(SR2Error):
    """ContentProvider resolution failure."""


class ReducerError(SR2Error):
    """ContentReducer execution failure."""


class MemoryError(SR2Error):
    """Memory store or retrieval error."""
