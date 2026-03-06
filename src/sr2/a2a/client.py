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

    async def execute(
        self,
        message: str,
        metadata: dict | None = None,
        task_id: str | None = None,
    ) -> str:
        """Execute the A2A tool call.

        1. Build A2A request payload
        2. Send HTTP request to target agent
        3. Parse response
        4. Return result string for the LLM context
        """
        if not self._http:
            return "Error: No HTTP client configured for A2A tool"

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
                logger.warning(
                    f"A2A call returned non-completed status '{status}': {result}"
                )
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
            logger.warning(f"Failed to fetch agent card: {e}")
            return None
