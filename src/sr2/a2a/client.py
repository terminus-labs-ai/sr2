"""A2A client tool — allows agents to call remote agents via A2A protocol."""

import logging
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class A2AToolConfig:
    """Configuration for an A2A client tool."""

    name: str
    target_url: str
    skill_id: str | None = None
    timeout_seconds: float = 120.0
    description: str = ""


class A2AClientTool:
    """Tool for calling remote agents via A2A protocol.

    The agent's LLM calls this like any other tool. The runtime executes it
    by sending an HTTP request to the remote agent.
    """

    def __init__(
        self,
        config: A2AToolConfig,
        http_callable=None,
    ):
        """
        Args:
            config: Tool configuration
            http_callable: async function(url: str, payload: dict, timeout: float) -> dict
                           Wraps httpx or any HTTP client. For testing, use a mock.
        """
        self._config = config
        self._http = http_callable

    @property
    def tool_definition(self) -> dict:
        """Return the tool definition for the LLM (OpenAI function schema)."""
        return {
            "name": self._config.name,
            "description": self._config.description
            or f"Call remote agent at {self._config.target_url}",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message/task to send to the remote agent",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata to include with the request",
                        "default": {},
                    },
                },
                "required": ["message"],
            },
        }

    # Parameter names the LLM commonly uses instead of "message"
    _MESSAGE_ALIASES = {
        "message",
        "prompt",
        "query",
        "input",
        "text",
        "request",
        "task",
        "description",
        "content",
    }

    def _extract_message(self, kwargs: dict) -> str | None:
        """Extract the message from kwargs, tolerating LLM-chosen param names."""
        # 1. Exact match on known aliases
        for alias in self._MESSAGE_ALIASES:
            if alias in kwargs and isinstance(kwargs[alias], str):
                return kwargs[alias]
        # 2. Fallback: first string value
        for value in kwargs.values():
            if isinstance(value, str):
                return value
        return None

    async def execute(self, **kwargs) -> str:
        """Execute the A2A tool call.

        Accepts **kwargs to be resilient to LLMs that ignore the schema
        and use parameter names like 'prompt', 'query', etc. instead of 'message'.
        """
        if not self._http:
            return "Error: No HTTP client configured for A2A tool"

        message = self._extract_message(kwargs)
        if message is None:
            return f"Error: No message found in arguments: {list(kwargs.keys())}"

        metadata = kwargs.get("metadata")
        task_id = kwargs.get("task_id")
        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        payload: dict = {
            "task_id": tid,
            "message": message,
        }
        if metadata:
            payload["metadata"] = metadata
        if self._config.skill_id:
            payload["skill_id"] = self._config.skill_id

        try:
            logger.info(
                f"A2A call to {self._config.target_url} with task_id={tid}, "
                f"message_length={len(message)}"
            )
            response = await self._http(
                url=f"{self._config.target_url}/a2a/message",
                payload=payload,
                timeout=self._config.timeout_seconds,
            )

            status = response.get("status", "unknown")
            result = response.get("result", "No result returned")

            logger.info(
                f"A2A response from {self._config.target_url}: "
                f"status={status}, result_length={len(str(result))}"
            )
            logger.debug(f"A2A response result: {str(result)[:500]}")

            if status == "completed":
                return result
            else:
                logger.warning(f"A2A call returned non-completed status '{status}': {result}")
                return f"Remote agent returned status '{status}': {result}"

        except TimeoutError:
            logger.error(
                f"A2A call to {self._config.target_url} timed out "
                f"after {self._config.timeout_seconds}s"
            )
            return f"A2A call to {self._config.target_url} timed out after {self._config.timeout_seconds}s"
        except Exception as e:
            logger.error(f"A2A call to {self._config.target_url} failed: {e}", exc_info=True)
            return f"A2A call failed: {e}"

    async def fetch_agent_card(self) -> dict | None:
        """Fetch the remote agent's Agent Card for capability discovery."""
        if not self._http:
            return None
        try:
            response = await self._http(
                url=f"{self._config.target_url}/.well-known/agent.json",
                payload={},
                timeout=10.0,
            )
            return response
        except Exception as e:
            logger.error("Failed to fetch agent card", exc_info=True)
            return None
