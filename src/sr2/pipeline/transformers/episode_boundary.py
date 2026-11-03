"""LLM judge for autonomous episode boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic

from sr2.config.models import ConfigError, TransformerConfig
from sr2.models import ContentBlock, Message, TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.protocols import TokenCounter
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionRequest, LLMCallable

_DEFAULT_PROMPT = (
    "Decide whether the newest turn starts a new topic. Return JSON: "
    '{"decision":"continue"} or '
    '{"decision":"close","title":"...","summary":"..."}.'
)


class EpisodeBoundaryFailure(RuntimeError):
    """A judge transport failure carrying the continuity capability."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.capability = "consolidation"
        self.retryable = True


@dataclass(frozen=True)
class EpisodeJudgement:
    decision: str
    title: str = ""
    summary: str = ""
    warning_code: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class EpisodeBoundaryTransformer:
    """Judge completed turns without changing compiled layer content."""

    name = "episode_boundary"

    def __init__(
        self,
        config: TransformerConfig,
        llm: LLMCallable,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._token_counter = token_counter or CharacterTokenCounter()
        self.max_executions = config.max_executions
        self.execution_count = 0
        self.subscriptions = build_subscriptions(
            config.subscriptions,
            {**PHASE_MAP, "started": EventPhase.STARTING},
            [EventSubscription("assistant_response", EventPhase.COMPLETED)],
        )
        self._observed_events: list[Event] = []
        self.episode_start: int | None = None

    @classmethod
    def build(
        cls,
        config: TransformerConfig,
        deps: Dependencies,
        token_counter: TokenCounter | None = None,
    ) -> "EpisodeBoundaryTransformer":
        if deps.llm is None:
            raise ConfigError("transformer 'episode_boundary' requires an LLM")
        key = config.config.get("judge_model", "default")
        if key not in deps.llm:
            raise ConfigError(
                f"transformer 'episode_boundary' configured judge_model={key!r} "
                f"but that key is absent from deps.llm"
            )
        return cls(config, deps.llm[key], token_counter)

    def observe(self, event: Event) -> None:
        """Record a real bus event when its configured phase also matches."""
        if any(
            subscription.event_name == event.name
            and (subscription.phase is None or subscription.phase is event.phase)
            for subscription in self.subscriptions
        ):
            self._observed_events.append(event)

    def consume_observed_events(self) -> list[Event]:
        """Return this turn's triggers once for deferred judge execution."""
        events = self._observed_events
        self._observed_events = []
        return events

    @property
    def min_episode_turns(self) -> int:
        return self._config.config.get("min_episode_turns", 2)

    async def judge(self, turns: list[tuple[int, str]]) -> EpisodeJudgement:
        request = self._build_request(turns)
        started = monotonic()
        try:
            response = await self._llm.complete(request)
        except Exception as exc:
            raise EpisodeBoundaryFailure(f"episode boundary judge failed: {exc}") from exc
        latency_ms = (monotonic() - started) * 1000
        raw = next(
            (block.text for block in response.content if isinstance(block, TextBlock)),
            "",
        )
        try:
            payload = json.loads(raw)
            decision = payload.get("decision")
            if decision == "continue":
                title = summary = ""
            elif decision == "close":
                title = payload.get("title")
                summary = payload.get("summary")
                if not isinstance(title, str) or not title or not isinstance(summary, str) or not summary:
                    raise ValueError("close requires title and summary")
            else:
                raise ValueError("unknown decision")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            decision, title, summary = "continue", "", ""
            warning_code = "episode_boundary.unparseable"
        else:
            warning_code = None
        return EpisodeJudgement(
            decision=decision,
            title=title,
            summary=summary,
            warning_code=warning_code,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _build_request(self, turns: list[tuple[int, str]]) -> CompletionRequest:
        prompt = self._config.config.get("judge_prompt") or _DEFAULT_PROMPT
        budget = self._config.config.get("judge_max_tokens", 4000)
        system = [TextBlock(text=prompt)]
        while self._request_tokens(CompletionRequest(system=system, messages=[])) > budget:
            prompt = prompt[: max(0, len(prompt) - max(1, len(prompt) // 4))]
            system = [TextBlock(text=prompt)] if prompt else []
        selected: list[Message] = []
        for seq, text in reversed(turns):
            candidate = Message(
                role="user",
                content=[TextBlock(text=f"Turn {seq}:\n{text}")],
            )
            request = CompletionRequest(
                system=system,
                messages=[candidate, *selected],
            )
            if self._request_tokens(request) <= budget:
                selected.insert(0, candidate)
            else:
                break
        if not selected and turns:
            seq, text = turns[-1]
            while text:
                candidate = Message(
                    role="user",
                    content=[TextBlock(text=f"Turn {seq}:\n{text}")],
                )
                request = CompletionRequest(system=system, messages=[candidate])
                if self._request_tokens(request) <= budget:
                    selected = [candidate]
                    break
                text = text[max(1, len(text) // 4) :]
        return CompletionRequest(system=system, messages=selected)

    def _request_tokens(self, request: CompletionRequest) -> int:
        blocks: list[ContentBlock] = list(request.system or [])
        blocks.extend(block for message in request.messages for block in message.content)
        return self._token_counter.count(blocks)

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult:
        return TransformationResult(
            transformer_name=self.name,
            source_layer="episode_boundary",
            content=None,
            events=[],
            entries=[],
        )
