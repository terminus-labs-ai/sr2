"""A2A server adapter — routes inbound A2A messages through the pipeline."""

import json
import logging
from dataclasses import dataclass

from sr2.a2a.card import AgentCardGenerator
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.router import InterfaceRouter
from sr2.resolvers.registry import ResolverContext

logger = logging.getLogger(__name__)


@dataclass
class A2ARequest:
    """Parsed inbound A2A request."""

    task_id: str
    message: str
    metadata: dict | None = None


@dataclass
class A2AResponse:
    """Outbound A2A response."""

    task_id: str
    status: str  # "completed", "failed"
    result: str
    metadata: dict | None = None


class A2AServerAdapter:
    """Receives A2A messages and routes them through the pipeline.

    This is NOT a full A2A server implementation (that's the a2a-sdk's job).
    This is the glue between the A2A SDK's request handling and our pipeline.
    """

    def __init__(
        self,
        pipeline_engine: PipelineEngine,
        interface_router: InterfaceRouter,
        card_generator: AgentCardGenerator,
        agent_config: dict | None = None,
        interface_name: str = "a2a_inbound",
    ):
        self._engine = pipeline_engine
        self._router = interface_router
        self._card = card_generator
        self._agent_config = agent_config or {}
        self._interface = interface_name

    def get_agent_card(self) -> dict:
        """Return the agent card for discovery."""
        return self._card.generate()

    async def handle_message(self, request: A2ARequest) -> A2AResponse:
        """Handle an inbound A2A message.

        1. Get the pipeline config for the a2a_inbound interface
        2. Build resolver context from the request
        3. Run the pipeline to compile context
        4. Return the compiled context as the "result"
        """
        try:
            config = self._router.route(self._interface)

            context = ResolverContext(
                agent_config=self._agent_config,
                trigger_input={"message": request.message, **(request.metadata or {})},
                session_id=request.task_id,
                interface_type=self._interface,
            )

            compiled = await self._engine.compile(config, context)

            return A2AResponse(
                task_id=request.task_id,
                status="completed",
                result=compiled.content,
                metadata={
                    "tokens": compiled.tokens,
                    "pipeline_status": compiled.pipeline_result.overall_status,
                },
            )
        except Exception as e:
            logger.error("A2A message handling failed", exc_info=True)
            return A2AResponse(
                task_id=request.task_id,
                status="failed",
                result=str(e),
            )

    async def handle_raw_json(self, raw: str) -> str:
        """Handle raw JSON string (for direct HTTP integration).

        Parses the JSON, extracts task_id and message, calls handle_message,
        returns JSON response.
        """
        try:
            data = json.loads(raw)
            request = A2ARequest(
                task_id=data.get("task_id", "unknown"),
                message=data.get("message", ""),
                metadata=data.get("metadata"),
            )
            response = await self.handle_message(request)
            return json.dumps(
                {
                    "task_id": response.task_id,
                    "status": response.status,
                    "result": response.result,
                    "metadata": response.metadata,
                }
            )
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "task_id": "unknown",
                    "status": "failed",
                    "result": f"Invalid JSON: {e}",
                }
            )
