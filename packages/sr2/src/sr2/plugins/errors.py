"""Plugin error types."""


class PluginNotFoundError(ImportError):
    """Raised when a requested plugin is not registered or discoverable."""

    pass


class PluginLicenseError(ImportError):
    """Raised when a plugin was discovered but blocked by license validation."""

    pass
