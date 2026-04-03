"""Plugin protocol and trigger context for interface plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sr2_runtime.llm.streaming import StreamCallback


@dataclass
class TriggerContext:
    """Context passed from a plugin to the agent when a trigger fires."""

    interface_name: str  # Name of the interface in YAML
    plugin_name: str  # Plugin type (telegram, timer, http, a2a)
    session_name: str  # Resolved session name
    session_lifecycle: str  # persistent | ephemeral | rolling
    input_data: Any  # The trigger payload (message text, HTTP body, timer tick, etc.)
    metadata: dict | None = None  # Plugin-specific metadata (user_id, request_id, etc.)
    respond_callback: StreamCallback | None = None  # async callback for streaming events


class InterfacePlugin(Protocol):
    """Protocol that all interface plugins must implement.

    Lifecycle:
    1. __init__(interface_name, config, agent_callback) - plugin created with its YAML config
    2. start() - begin listening/polling
    3. [agent_callback called when triggers arrive]
    4. stop() - graceful shutdown
    """

    async def start(self) -> None:
        """Start the plugin (begin listening, connect, start polling)."""
        ...

    async def stop(self) -> None:
        """Stop the plugin gracefully."""
        ...

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """Send a message through this interface.

        Used by post_to_session when the target session is owned by this plugin.
        Plugins that don't support proactive sending can no-op.
        """
        ...
