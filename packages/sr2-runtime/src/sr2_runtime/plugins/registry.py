"""Plugin registry — maps plugin names to plugin classes."""


class PluginRegistry:
    """Maps plugin names to plugin classes."""

    def __init__(self):
        self._plugins: dict[str, type] = {}

    def register(self, name: str, plugin_class: type) -> None:
        self._plugins[name] = plugin_class

    def get(self, name: str) -> type:
        if name not in self._plugins:
            raise KeyError(f"Unknown plugin: {name}. Available: {list(self._plugins.keys())}")
        return self._plugins[name]

    @property
    def available(self) -> list[str]:
        return list(self._plugins.keys())


def create_default_registry() -> PluginRegistry:
    """Create registry with all built-in plugins."""
    from sr2_runtime.plugins.telegram import TelegramPlugin
    from sr2_runtime.plugins.timer import TimerPlugin
    from sr2_runtime.plugins.http import HTTPPlugin
    from sr2_runtime.plugins.a2a import A2APlugin
    from sr2_runtime.plugins.single_shot import SingleShotPlugin

    reg = PluginRegistry()
    reg.register("telegram", TelegramPlugin)
    reg.register("timer", TimerPlugin)
    reg.register("http", HTTPPlugin)
    reg.register("a2a", A2APlugin)
    reg.register("single_shot", SingleShotPlugin)
    return reg
