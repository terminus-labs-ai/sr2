from sr2.config.models import PipelineConfig
from sr2.config.loader import ConfigLoader


class InterfaceRouter:
    """Routes triggers to the correct pipeline config."""

    def __init__(self, interfaces: dict[str, str | dict], loader: ConfigLoader):
        """
        Args:
            interfaces: mapping of interface_name -> config_path or config_dict
            loader: ConfigLoader instance for resolving configs
        """
        self._interfaces = interfaces
        self._loader = loader
        self._cache: dict[str, PipelineConfig] = {}

    def route(self, interface_type: str) -> PipelineConfig:
        """Get the resolved config for an interface type.

        Caches resolved configs (they don't change at runtime).
        Raises KeyError if interface_type is not registered.
        """
        if interface_type not in self._interfaces:
            raise KeyError(f"Unknown interface type: {interface_type}")
        if interface_type not in self._cache:
            source = self._interfaces[interface_type]
            if isinstance(source, dict):
                self._cache[interface_type] = self._loader.load_from_dict(source)
            else:
                self._cache[interface_type] = self._loader.load(source)
        return self._cache[interface_type]

    def invalidate(self, name: str) -> None:
        """Remove the named interface from cache, forcing a re-read on next route().

        No-op if the name isn't cached or isn't a registered interface.
        """
        self._cache.pop(name, None)

    def reload_interface(self, name: str) -> PipelineConfig:
        """Invalidate cache for *name*, then re-route (re-read from disk).

        Returns the freshly loaded PipelineConfig. If validation fails,
        the exception propagates and the cache stays empty (no stale data).
        """
        self.invalidate(name)
        return self.route(name)

    @property
    def registered_interfaces(self) -> list[str]:
        return list(self._interfaces.keys())
