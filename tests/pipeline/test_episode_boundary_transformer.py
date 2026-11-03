"""sr2-160: episode-boundary judge, close record, and YAML configuration."""

# ruff: noqa: E402 -- direct single-file pytest runs need tests/conftest on sys.path.

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, is_dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from conftest import make_minimal_config, make_user_input, tool_use_event
from sr2.cli import load_config
from sr2.config.models import PipelineConfig
from sr2.models import TextBlock, TokenUsage, ToolResultBlock
from sr2.orchestrator import SR2
from sr2.pipeline.provenance import InMemoryProvenanceStore, read_turn_archive
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

FIXTURES = Path(__file__).parent / "fixtures" / "episode_boundary"


class ScriptedLLM:
    def __init__(
        self,
        decisions: list[str] | None = None,
        *,
        usage: TokenUsage | None = None,
        tool_turn: bool = False,
        failure: Exception | None = None,
        store: RecordingContinuityStore | None = None,
    ) -> None:
        self.decisions = decisions or ['{"decision":"continue"}']
        self.usage = usage or TokenUsage()
        self.tool_turn = tool_turn
        self.failure = failure
        self.store = store
        self.complete_calls: list[CompletionRequest] = []
        self.stream_calls: list[CompletionRequest] = []
        self.records_seen_at_stream: list[int] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.complete_calls.append(request)
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        index = min(len(self.complete_calls) - 1, len(self.decisions) - 1)
        return CompletionResponse(
            id=f"judge-{index}",
            content=[TextBlock(text=self.decisions[index])],
            stop_reason="end_turn",
            usage=self.usage,
        )

    async def stream(self, request: CompletionRequest):
        self.stream_calls.append(request)
        if self.store is not None:
            self.records_seen_at_stream.append(len(await self.store.get_events()))
        if self.tool_turn and len(self.stream_calls) == 1:
            yield StreamEvent(type="text", text="checking")
            yield tool_use_event("episode-tool")
        else:
            yield StreamEvent(type="text", text=f"answer-{len(self.stream_calls)}")
        yield StreamEvent(type="end")


async def _tool_executor(block) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=block.id, content="tool result")


class RecordingContinuityStore:
    """The continuity-store boundary, with records directly readable by tests."""

    def __init__(self) -> None:
        self.records = []

    async def append(self, record) -> None:
        self.records.append(record)

    async def get_events(self):
        return list(self.records)

    async def close(self) -> None:
        return None


class CompileBoundaryTracer(CollectingTracer):
    def __init__(self, store: RecordingContinuityStore) -> None:
        super().__init__()
        self.store = store
        self.records_seen_at_compile: list[int] = []

    def on_compile(self, request: CompletionRequest) -> None:
        self.records_seen_at_compile.append(len(self.store.records))
        super().on_compile(request)


def _episode_config(
    *,
    judge_model: str | None = None,
    judge_max_tokens: int = 4000,
    min_episode_turns: int = 2,
    judge_prompt: str | None = None,
    subscriptions: list[dict[str, str]] | None = None,
) -> PipelineConfig:
    payload = make_minimal_config().model_dump()
    config: dict[str, Any] = {
        "judge_max_tokens": judge_max_tokens,
        "min_episode_turns": min_episode_turns,
    }
    if judge_model is not None:
        config["judge_model"] = judge_model
    if judge_prompt is not None:
        config["judge_prompt"] = judge_prompt
    transformer: dict[str, Any] = {"type": "episode_boundary", "config": config}
    if subscriptions is not None:
        transformer["subscriptions"] = subscriptions
    payload["transformers"] = [transformer]
    return PipelineConfig.model_validate(payload)


def _runtime(
    llm,
    *,
    config: PipelineConfig | None = None,
    store: RecordingContinuityStore | None = None,
    provenance=None,
    tracer: CollectingTracer | None = None,
    session_id: str = "episode-tests",
    tool_executor=None,
) -> SR2:
    return SR2(
        pipeline_config=config or _episode_config(),
        llm=llm,
        token_counter=CharacterTokenCounter(),
        continuity_store=store,
        provenance_store=provenance,
        tracer=tracer,
        session_id=session_id,
        tool_executor=tool_executor,
    )


async def _drain(runtime: SR2, text: str) -> list[StreamEvent]:
    events = [event async for event in runtime.turn(make_user_input(text))]
    assert events[-1].type == "end"
    return events


async def _open_store(tmp_path: Path, name: str = "continuity.db") -> RecordingContinuityStore:
    return RecordingContinuityStore()


async def _closed_episode(tmp_path: Path, *, tracer_factory=None):
    store = await _open_store(tmp_path)
    provenance = InMemoryProvenanceStore()
    session_id = "close-record"
    tracer = tracer_factory(store) if tracer_factory is not None else None

    seed = _runtime(
        {"default": ScriptedLLM()},
        config=make_minimal_config(),
        store=store,
        provenance=provenance,
        session_id=session_id,
    )
    await _drain(seed, "pre-feature turn zero")
    await seed.aclose()

    judge = ScriptedLLM(
        ['{"decision":"continue"}'] * 4
        + ['{"decision":"close","title":"Topic A","summary":"A was resolved."}'],
        store=store,
    )
    runtime = _runtime(
        {"default": judge},
        store=store,
        provenance=provenance,
        tracer=tracer,
        session_id=session_id,
    )
    await runtime.continue_archive()
    for seq in range(1, 5):
        await _drain(runtime, f"topic A turn {seq}")
        await runtime.aclose()
    await _drain(runtime, "topic B opens on turn 5")
    await runtime.aclose()
    return runtime, judge, store, provenance


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _contains_pair(value: Any, key: str, expected: Any) -> bool:
    value = _plain(value)
    if isinstance(value, dict):
        if value.get(key) == expected:
            return True
        return any(_contains_pair(item, key, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_pair(item, key, expected) for item in value)
    return False


def _metric_values(value: Any, words: tuple[str, ...]) -> list[float]:
    found: list[float] = []

    def visit(item: Any, path: tuple[str, ...] = ()) -> None:
        item = _plain(item)
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, path + (key.lower(),))
        elif isinstance(item, list):
            for child in item:
                visit(child, path)
        elif isinstance(item, (int, float)) and all(
            any(word in segment for segment in path) for word in words
        ):
            found.append(float(item))

    visit(value)
    return found


def _request_text(request: CompletionRequest) -> str:
    blocks = list(request.system or [])
    blocks.extend(block for message in request.messages for block in message.content)
    return "\n".join(getattr(block, "text", "") for block in blocks)


def _request_tokens(request: CompletionRequest) -> int:
    blocks = list(request.system or [])
    blocks.extend(block for message in request.messages for block in message.content)
    return CharacterTokenCounter().count(blocks)


@pytest.mark.asyncio
async def test_ac23_r1_close_appends_episode_with_exact_closed_turns(tmp_path):
    runtime, _, store, provenance = await _closed_episode(tmp_path)
    records = await store.get_events()
    expected_entries = []
    for seq in range(1, 5):
        expected_entries.extend(
            await read_turn_archive(provenance, runtime.session_id, seq)
        )
    expected_ids = tuple(entry.id for entry in expected_entries)
    assert len(records) == 1
    record = records[0]
    assert record.kind.value == "episode"
    assert record.event_kind.value == "formed"
    assert record.content == "A was resolved."
    assert record.meta["episode_state"] == "closed"
    assert record.meta["title"] == "Topic A"
    assert record.meta["turn_range"] == [1, 4]
    assert isinstance(record.meta["episode_id"], str) and record.meta["episode_id"]
    assert record.provenance_ids == expected_ids
    await store.close()


@pytest.mark.asyncio
async def test_ac23_r2_triggering_turn_is_not_in_closed_episode(tmp_path):
    runtime, _, store, provenance = await _closed_episode(tmp_path)
    record = (await store.get_events())[0]
    turn_five_ids = {
        entry.id for entry in await read_turn_archive(provenance, runtime.session_id, 5)
    }
    assert turn_five_ids
    assert turn_five_ids.isdisjoint(record.provenance_ids)
    await store.close()


@pytest.mark.asyncio
async def test_ac23_r3_close_is_stored_before_next_compile(tmp_path):
    tracers = []

    def build_tracer(continuity_store):
        tracer = CompileBoundaryTracer(continuity_store)
        tracers.append(tracer)
        return tracer

    runtime, _, store, _ = await _closed_episode(
        tmp_path,
        tracer_factory=build_tracer,
    )
    tracer = tracers[0]
    tracer.records_seen_at_compile.clear()
    assert len(await store.get_events()) == 1
    await _drain(runtime, "turn 6")
    assert tracer.records_seen_at_compile
    assert tracer.records_seen_at_compile[0] == 1
    await runtime.aclose()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "case"),
    [
        ("not json", "AC24-r1"),
        ('{"decision":"close"}', "AC24-r2"),
    ],
)
async def test_ac24_malformed_close_warns_and_appends_nothing(tmp_path, output, case):
    store = await _open_store(tmp_path, f"{case}.db")
    tracer = CollectingTracer()
    runtime = _runtime({"default": ScriptedLLM([output])}, store=store, tracer=tracer)
    await _drain(runtime, "one turn")
    await runtime.aclose()
    assert await store.get_events() == []
    assert _contains_pair(tracer.get_trace(), "code", "episode_boundary.unparseable")
    await store.close()


@pytest.mark.asyncio
async def test_ac24_r3_transport_failure_is_structured_consolidation_degradation(tmp_path):
    store = await _open_store(tmp_path)
    judge = ScriptedLLM(failure=RuntimeError("judge offline"))
    runtime = _runtime({"default": judge}, store=store)
    await _drain(runtime, "first turn")
    try:
        result: Any = await _drain(runtime, "next turn")
    except Exception as exc:  # ContinuityUnavailable is an allowed public carrier.
        result = exc
    assert _contains_pair(result, "capability", "consolidation")
    assert await store.get_events() == []
    await runtime.aclose()
    await store.close()


@pytest.mark.asyncio
async def test_ac24_r4_short_episode_never_closes(tmp_path):
    store = await _open_store(tmp_path)
    close = '{"decision":"close","title":"Too soon","summary":"One turn."}'
    judge = ScriptedLLM([close])
    runtime = _runtime(
        {"default": judge},
        config=_episode_config(min_episode_turns=2),
        store=store,
    )
    await _drain(runtime, "only turn")
    await runtime.aclose()
    assert len(judge.complete_calls) == 1
    assert await store.get_events() == []
    await store.close()


@pytest.mark.asyncio
async def test_ac25_r1_pipeline_without_transformer_matches_base_fixture():
    assert "transformers" in PipelineConfig.model_fields
    llm = ScriptedLLM()
    runtime = _runtime(
        {"default": llm},
        config=make_minimal_config(),
        session_id="episode-boundary-reference",
    )
    await _drain(runtime, "Reference input")
    expected = (FIXTURES / "compiled_request_base.json").read_bytes().rstrip(b"\n")
    assert llm.stream_calls[0].model_dump_json().encode() == expected
    assert llm.complete_calls == []
    await runtime.aclose()


def test_ac25_r2_unknown_judge_model_fails_during_load(tmp_path):
    path = tmp_path / "missing-judge.yaml"
    path.write_text((FIXTURES / "missing_judge_model.yaml").read_text())
    with pytest.raises(Exception, match="judge"):
        load_config(str(path))


@pytest.mark.asyncio
async def test_ac25_r3_omitted_model_uses_default(tmp_path):
    store = await _open_store(tmp_path)
    default = ScriptedLLM()
    runtime = _runtime({"default": default}, store=store)
    await _drain(runtime, "turn")
    await runtime.aclose()
    assert len(default.complete_calls) == 1
    await store.close()


@pytest.mark.asyncio
async def test_ac25_r4_yaml_transformer_judges_tool_turn_once(tmp_path):
    discovered = {
        entry_point.name: entry_point
        for entry_point in entry_points(group="sr2.transformers")
    }
    assert discovered["episode_boundary"].load()
    config = load_config(str(FIXTURES / "episode_boundary.yaml"))
    runtime_config = PipelineConfig.model_validate(config["pipeline"])
    store = await _open_store(tmp_path)
    default = ScriptedLLM(tool_turn=True)
    runtime = _runtime(
        {"default": default},
        config=runtime_config,
        store=store,
        tool_executor=_tool_executor,
    )
    await _drain(runtime, "use a tool")
    await runtime.aclose()
    assert len(default.stream_calls) == 2
    assert len(default.complete_calls) == 1
    await store.close()


@pytest.mark.asyncio
async def test_ac25_r5_explicit_subscription_matches_event_and_phase(tmp_path):
    completed = ScriptedLLM()
    runtime = _runtime(
        {"default": completed},
        config=_episode_config(
            subscriptions=[{"event": "turn_end", "phase": "completed"}]
        ),
    )
    await _drain(runtime, "matching non-default event")
    await runtime.aclose()
    assert len(completed.complete_calls) == 1

    wrong_event = ScriptedLLM()
    runtime = _runtime(
        {"default": wrong_event},
        config=_episode_config(
            subscriptions=[{"event": "tool_result", "phase": "completed"}]
        ),
    )
    await _drain(runtime, "unrelated event")
    await runtime.aclose()
    assert wrong_event.complete_calls == []

    wrong_phase = ScriptedLLM()
    runtime = _runtime(
        {"default": wrong_phase},
        config=_episode_config(
            subscriptions=[{"event": "turn_end", "phase": "started"}]
        ),
    )
    await _drain(runtime, "same event, unrelated phase")
    await runtime.aclose()
    assert wrong_phase.complete_calls == []


@pytest.mark.asyncio
async def test_ac25_r6_named_judge_model_routes_away_from_default(tmp_path):
    store = await _open_store(tmp_path)
    default = ScriptedLLM()
    judge = ScriptedLLM()
    runtime = _runtime(
        {"default": default, "judge": judge},
        config=_episode_config(judge_model="judge"),
        store=store,
    )
    await _drain(runtime, "turn")
    await runtime.aclose()
    assert len(judge.complete_calls) == 1
    assert default.complete_calls == []
    await store.close()


@pytest.mark.asyncio
async def test_ac25_r7_prompt_budget_keeps_newest_turns(tmp_path):
    store = await _open_store(tmp_path)
    llm = ScriptedLLM()
    runtime = _runtime(
        {"default": llm}, config=_episode_config(judge_max_tokens=260), store=store
    )
    for marker in ("OLDEST", "MIDDLE1", "MIDDLE2", "NEWEST"):
        await _drain(runtime, marker + " " + (marker.lower() + " ") * 20)
        await runtime.aclose()
    request = llm.complete_calls[-1]
    text = _request_text(request)
    assert _request_tokens(request) <= 260
    assert "NEWEST" in text
    assert "MIDDLE2" in text
    assert "OLDEST" not in text
    await store.close()


@pytest.mark.asyncio
async def test_ac25_r8_custom_judge_prompt_is_used(tmp_path):
    store = await _open_store(tmp_path)
    llm = ScriptedLLM()
    prompt = "DISTINCTIVE EPISODE OVERRIDE"
    runtime = _runtime(
        {"default": llm}, config=_episode_config(judge_prompt=prompt), store=store
    )
    await _drain(runtime, "turn")
    await runtime.aclose()
    assert prompt in _request_text(llm.complete_calls[0])
    await store.close()


@pytest.mark.asyncio
async def test_ac25_r9_trace_records_known_latency_and_usage(tmp_path, monkeypatch):
    module = importlib.import_module("sr2.pipeline.transformers.episode_boundary")
    ticks = iter((10.0, 10.125))
    if hasattr(module, "monotonic"):
        monkeypatch.setattr(module, "monotonic", lambda: next(ticks))
    else:
        monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    store = await _open_store(tmp_path)
    tracer = CollectingTracer()
    llm = ScriptedLLM(usage=TokenUsage(input_tokens=13, output_tokens=7))
    runtime = _runtime({"default": llm}, store=store, tracer=tracer)
    await _drain(runtime, "turn")
    await runtime.aclose()
    judge_records = [
        record
        for record in tracer.get_trace()
        if record.kind == "transformer" and record.component_name == "episode_boundary"
    ]
    assert len(judge_records) == 1
    judge_record = judge_records[0]
    assert 13 in _metric_values(judge_record, ("input", "token"))
    assert 7 in _metric_values(judge_record, ("output", "token"))
    latencies = _metric_values(judge_record, ("latency",))
    assert 125 in latencies or 0.125 in latencies
    await store.close()
