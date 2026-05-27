"""Tests for sr2.cli — load_config, build_llm_clients, chat_loop."""

from __future__ import annotations

import textwrap
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml

from sr2.protocols.llm import LLMCallable, StreamEvent


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """load_config(path: str) -> dict"""

    def test_returns_dict_for_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                models:
                  default:
                    model: anthropic/claude-sonnet-4-6

                pipeline:
                  token_budget: 200000
                  layers: []
            """)
        )

        from sr2.cli import load_config

        result = load_config(str(config_file))

        assert isinstance(result, dict)

    def test_returned_dict_contains_models_key(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                models:
                  default:
                    model: anthropic/claude-sonnet-4-6

                pipeline:
                  token_budget: 200000
                  layers: []
            """)
        )

        from sr2.cli import load_config

        result = load_config(str(config_file))

        assert "models" in result

    def test_returned_dict_contains_pipeline_key(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                models:
                  default:
                    model: anthropic/claude-sonnet-4-6

                pipeline:
                  token_budget: 200000
                  layers: []
            """)
        )

        from sr2.cli import load_config

        result = load_config(str(config_file))

        assert "pipeline" in result

    def test_models_section_content_matches_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                models:
                  default:
                    model: anthropic/claude-sonnet-4-6
                  fast:
                    model: anthropic/claude-haiku-4-5

                pipeline:
                  token_budget: 100000
                  layers: []
            """)
        )

        from sr2.cli import load_config

        result = load_config(str(config_file))

        assert result["models"]["default"]["model"] == "anthropic/claude-sonnet-4-6"
        assert result["models"]["fast"]["model"] == "anthropic/claude-haiku-4-5"

    def test_pipeline_section_content_matches_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                models:
                  default:
                    model: anthropic/claude-sonnet-4-6

                pipeline:
                  token_budget: 200000
                  layers: []
            """)
        )

        from sr2.cli import load_config

        result = load_config(str(config_file))

        assert result["pipeline"]["token_budget"] == 200000

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.yaml")

        from sr2.cli import load_config

        with pytest.raises(FileNotFoundError):
            load_config(missing)

    def test_raises_file_not_found_with_informative_path(self, tmp_path):
        missing = str(tmp_path / "nonexistent.yaml")

        from sr2.cli import load_config

        with pytest.raises(FileNotFoundError, match="nonexistent.yaml"):
            load_config(missing)

    def test_full_config_with_layers(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                models:
                  default:
                    model: anthropic/claude-sonnet-4-6
                    api_key: sk-ant-xxx

                pipeline:
                  token_budget: 200000
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: "You are a helpful assistant."
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
            """)
        )

        from sr2.cli import load_config

        result = load_config(str(config_file))

        assert result["models"]["default"]["api_key"] == "sk-ant-xxx"
        assert len(result["pipeline"]["layers"]) == 2
        assert result["pipeline"]["layers"][0]["name"] == "system"
        assert result["pipeline"]["layers"][1]["name"] == "conversation"


# ---------------------------------------------------------------------------
# build_llm_clients
# ---------------------------------------------------------------------------


class TestBuildLlmClients:
    """build_llm_clients(models: dict) -> dict[str, LLMCallable]"""

    def test_returns_dict(self):
        from sr2.cli import build_llm_clients

        models = {"default": {"model": "anthropic/claude-sonnet-4-6"}}

        result = build_llm_clients(models)

        assert isinstance(result, dict)

    def test_result_keyed_by_model_name(self):
        from sr2.cli import build_llm_clients

        models = {
            "default": {"model": "anthropic/claude-sonnet-4-6"},
            "fast": {"model": "anthropic/claude-haiku-4-5"},
        }

        result = build_llm_clients(models)

        assert set(result.keys()) == {"default", "fast"}

    def test_single_model_entry_produces_one_client(self):
        from sr2.cli import build_llm_clients

        models = {"default": {"model": "anthropic/claude-sonnet-4-6"}}

        result = build_llm_clients(models)

        assert len(result) == 1

    def test_multiple_model_entries_produce_multiple_clients(self):
        from sr2.cli import build_llm_clients

        models = {
            "default": {"model": "anthropic/claude-sonnet-4-6"},
            "fast": {"model": "anthropic/claude-haiku-4-5"},
            "local": {"model": "ollama/mistral"},
        }

        result = build_llm_clients(models)

        assert len(result) == 3

    def test_returned_clients_satisfy_llm_callable_protocol(self):
        from sr2.cli import build_llm_clients

        models = {"default": {"model": "anthropic/claude-sonnet-4-6"}}

        result = build_llm_clients(models)

        assert isinstance(result["default"], LLMCallable)

    def test_all_clients_satisfy_llm_callable_protocol(self):
        from sr2.cli import build_llm_clients

        models = {
            "default": {"model": "anthropic/claude-sonnet-4-6"},
            "fast": {"model": "anthropic/claude-haiku-4-5"},
        }

        result = build_llm_clients(models)

        for name, client in result.items():
            assert isinstance(client, LLMCallable), (
                f"Client for '{name}' does not satisfy LLMCallable protocol"
            )

    def test_uses_model_field_as_litellm_model_string(self):
        from sr2.cli import build_llm_clients
        from sr2.integrations.litellm import LiteLLMCallable

        models = {"default": {"model": "anthropic/claude-sonnet-4-6"}}

        result = build_llm_clients(models)

        client = result["default"]
        assert isinstance(client, LiteLLMCallable)
        assert client.model == "anthropic/claude-sonnet-4-6"

    def test_optional_kwargs_forwarded_to_litellm(self):
        """api_key and base_url from config are forwarded when the client makes LLM calls."""
        from unittest.mock import AsyncMock, patch

        from sr2.cli import build_llm_clients
        from sr2.config.models import PipelineConfig
        from sr2.models import Message, TextBlock
        from sr2.protocols.llm import CompletionRequest

        models = {
            "default": {
                "model": "anthropic/claude-sonnet-4-6",
                "api_key": "sk-ant-test-key",
                "base_url": "https://custom.example.com",
            }
        }

        result = build_llm_clients(models)
        client = result["default"]

        fake_response = AsyncMock()
        fake_response.id = "resp-1"
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.choices[0].message.tool_calls = None
        fake_response.choices[0].finish_reason = "end_turn"
        fake_response.usage.prompt_tokens = 5
        fake_response.usage.completion_tokens = 3

        import asyncio

        request = CompletionRequest(messages=[Message(role="user", content=[TextBlock(text="hi")])])

        with patch("sr2.integrations.litellm.litellm.acompletion", return_value=fake_response) as mock_call:
            asyncio.get_event_loop().run_until_complete(client.complete(request))
            call_kwargs = mock_call.call_args.kwargs
            assert call_kwargs.get("api_key") == "sk-ant-test-key"
            assert call_kwargs.get("base_url") == "https://custom.example.com"

    def test_empty_models_dict_returns_empty_dict(self):
        from sr2.cli import build_llm_clients

        result = build_llm_clients({})

        assert result == {}

    def test_model_without_optional_kwargs_works(self):
        from sr2.cli import build_llm_clients
        from sr2.integrations.litellm import LiteLLMCallable

        models = {"bare": {"model": "ollama/llama3"}}

        result = build_llm_clients(models)

        client = result["bare"]
        assert isinstance(client, LiteLLMCallable)
        assert client.model == "ollama/llama3"


# ---------------------------------------------------------------------------
# chat_loop
# ---------------------------------------------------------------------------


def _make_stream_events(*texts: str) -> list[StreamEvent]:
    """Helper: build a list of StreamEvents ending with 'end'."""
    events = [StreamEvent(type="text", text=t) for t in texts]
    events.append(StreamEvent(type="end"))
    return events


async def _async_gen_from_list(events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


class TestChatLoop:
    """chat_loop(sr2: SR2) -> None — interactive REPL over sr2.turn()"""

    @pytest.mark.asyncio
    async def test_exits_on_exit_command(self):
        """Typing 'exit' terminates the loop without error."""
        from sr2.cli import chat_loop

        mock_sr2 = MagicMock()

        with patch("builtins.input", return_value="exit"):
            with patch("builtins.print"):
                # Should return cleanly
                await chat_loop(mock_sr2)

    @pytest.mark.asyncio
    async def test_exits_on_quit_command(self):
        """Typing 'quit' terminates the loop without error."""
        from sr2.cli import chat_loop

        mock_sr2 = MagicMock()

        with patch("builtins.input", return_value="quit"):
            with patch("builtins.print"):
                await chat_loop(mock_sr2)

    @pytest.mark.asyncio
    async def test_exits_on_exit_case_insensitive(self):
        """Exit command is case-insensitive (EXIT, Exit, etc.)."""
        from sr2.cli import chat_loop

        mock_sr2 = MagicMock()

        with patch("builtins.input", return_value="EXIT"):
            with patch("builtins.print"):
                await chat_loop(mock_sr2)

    @pytest.mark.asyncio
    async def test_exits_on_quit_case_insensitive(self):
        """Quit command is case-insensitive (QUIT, Quit, etc.)."""
        from sr2.cli import chat_loop

        mock_sr2 = MagicMock()

        with patch("builtins.input", return_value="QUIT"):
            with patch("builtins.print"):
                await chat_loop(mock_sr2)

    @pytest.mark.asyncio
    async def test_calls_sr2_turn_with_user_input_as_text_block(self):
        """User input is passed to sr2.turn() as [TextBlock(text=user_input)]."""
        from sr2.cli import chat_loop
        from sr2 import TextBlock

        received_args = []

        async def fake_turn(content):
            received_args.append(content)
            yield StreamEvent(type="end")

        mock_sr2 = MagicMock()
        mock_sr2.turn = fake_turn

        inputs = iter(["hello world", "exit"])
        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print"):
                await chat_loop(mock_sr2)

        assert len(received_args) == 1
        assert received_args[0] == [TextBlock(text="hello world")]

    @pytest.mark.asyncio
    async def test_prints_text_events_with_no_newline(self):
        """StreamEvent(type='text') content is printed with end='' and flush=True."""
        from sr2.cli import chat_loop

        async def fake_turn(content):
            yield StreamEvent(type="text", text="Hello")
            yield StreamEvent(type="text", text=", world")
            yield StreamEvent(type="end")

        mock_sr2 = MagicMock()
        mock_sr2.turn = fake_turn

        print_calls = []
        inputs = iter(["say hello", "exit"])

        def capturing_print(*args, **kwargs):
            print_calls.append((args, kwargs))

        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print", side_effect=capturing_print):
                await chat_loop(mock_sr2)

        # Find print calls that came from text streaming (end="" flush=True)
        text_prints = [
            (args, kwargs)
            for args, kwargs in print_calls
            if kwargs.get("end") == "" and kwargs.get("flush") is True
        ]

        assert len(text_prints) == 2
        assert text_prints[0][0] == ("Hello",)
        assert text_prints[1][0] == (", world",)

    @pytest.mark.asyncio
    async def test_prints_newline_after_complete_response(self):
        """A newline is printed after each complete turn response."""
        from sr2.cli import chat_loop

        async def fake_turn(content):
            yield StreamEvent(type="text", text="Done")
            yield StreamEvent(type="end")

        mock_sr2 = MagicMock()
        mock_sr2.turn = fake_turn

        print_calls = []
        inputs = iter(["ask something", "exit"])

        def capturing_print(*args, **kwargs):
            print_calls.append((args, kwargs))

        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print", side_effect=capturing_print):
                await chat_loop(mock_sr2)

        # At least one bare print() call (newline) must exist after the text events
        bare_newline_calls = [
            (args, kwargs)
            for args, kwargs in print_calls
            if args == () and kwargs == {}
        ]
        assert len(bare_newline_calls) >= 1

    @pytest.mark.asyncio
    async def test_ignores_non_text_events(self):
        """Events with type='end' and type='usage' are not printed as text."""
        from sr2.cli import chat_loop
        from sr2.models import TokenUsage

        async def fake_turn(content):
            yield StreamEvent(type="text", text="result")
            yield StreamEvent(type="usage", usage=TokenUsage(input_tokens=10, output_tokens=5))
            yield StreamEvent(type="end")

        mock_sr2 = MagicMock()
        mock_sr2.turn = fake_turn

        text_stream_prints = []
        inputs = iter(["query", "exit"])

        def capturing_print(*args, **kwargs):
            if kwargs.get("end") == "" and kwargs.get("flush") is True:
                text_stream_prints.append(args)

        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print", side_effect=capturing_print):
                await chat_loop(mock_sr2)

        # Only the single "result" text should have been streamed
        assert len(text_stream_prints) == 1
        assert text_stream_prints[0] == ("result",)

    @pytest.mark.asyncio
    async def test_handles_keyboard_interrupt_cleanly(self):
        """KeyboardInterrupt exits without raising or printing a traceback."""
        from sr2.cli import chat_loop

        mock_sr2 = MagicMock()

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("builtins.print"):
                # Must not raise
                await chat_loop(mock_sr2)

    @pytest.mark.asyncio
    async def test_llm_error_prints_message_and_continues_loop(self):
        """An exception during sr2.turn() prints an error and returns to the prompt."""
        from sr2.cli import chat_loop

        call_count = 0

        async def fake_turn(content):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM unavailable")
                yield  # unreachable; makes this a true async generator
            yield StreamEvent(type="text", text="ok")
            yield StreamEvent(type="end")

        mock_sr2 = MagicMock()
        mock_sr2.turn = fake_turn

        print_calls = []
        inputs = iter(["bad request", "good request", "exit"])

        def capturing_print(*args, **kwargs):
            print_calls.append((args, kwargs))

        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print", side_effect=capturing_print):
                # Must not raise despite LLM error
                await chat_loop(mock_sr2)

        # An error message must have been printed
        all_printed_text = " ".join(
            str(arg) for args, _ in print_calls for arg in args
        ).lower()
        assert any(
            keyword in all_printed_text
            for keyword in ("error", "failed", "unavailable", "exception")
        ), f"Expected an error message in output, got: {all_printed_text!r}"

        # The loop must have continued — call_count should be 2 (bad + good)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_loop_continues_across_multiple_turns(self):
        """The loop handles multiple user messages before exiting."""
        from sr2.cli import chat_loop

        turn_count = 0

        async def fake_turn(content):
            nonlocal turn_count
            turn_count += 1
            yield StreamEvent(type="text", text=f"response {turn_count}")
            yield StreamEvent(type="end")

        mock_sr2 = MagicMock()
        mock_sr2.turn = fake_turn

        inputs = iter(["first", "second", "third", "exit"])

        with patch("builtins.input", side_effect=inputs):
            with patch("builtins.print"):
                await chat_loop(mock_sr2)

        assert turn_count == 3

    @pytest.mark.asyncio
    async def test_input_prompt_is_gt_symbol(self):
        """input() is called with '> ' as the prompt."""
        from sr2.cli import chat_loop

        prompt_args = []

        def capturing_input(prompt=""):
            prompt_args.append(prompt)
            return "exit"

        mock_sr2 = MagicMock()

        with patch("builtins.input", side_effect=capturing_input):
            with patch("builtins.print"):
                await chat_loop(mock_sr2)

        assert prompt_args[0] == "> "
