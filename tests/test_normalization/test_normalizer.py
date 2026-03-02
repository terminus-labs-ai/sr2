"""Tests for ResponseNormalizer."""

import pytest

from sr2.normalization.normalizer import ResponseNormalizer
from sr2.normalization.steps import (
    NormalizationInput,
    NormalizationOutput,
    StripMarkdownFencesStep,
    StripThinkingBlocksStep,
)


class TestResponseNormalizerDefaultChain:
    """Tests for ResponseNormalizer with the default step chain."""

    def test_clean_passthrough(self):
        """Clean JSON passes through unchanged."""
        normalizer = ResponseNormalizer()
        raw = '{"key": "value"}'
        assert normalizer.normalize(raw) == raw

    def test_think_block_and_preamble(self):
        """Strips <think> block and extracts JSON from preamble."""
        normalizer = ResponseNormalizer()
        raw = '<think>internal reasoning</think>\nHere is the answer:\n{"key_decisions": ["Deploy"], "facts": []}'
        result = normalizer.normalize(raw)
        assert result == '{"key_decisions": ["Deploy"], "facts": []}'

    def test_full_qwen3_style(self):
        """Full Qwen3/DeepSeek-R1 style: think block + markdown fence."""
        normalizer = ResponseNormalizer()
        raw = (
            "<thinking>\nStep 1: analyse context.\nStep 2: compose JSON.\n</thinking>\n"
            "```json\n{\"summary_of_turns\": \"1-5\", \"key_decisions\": [\"Use Redis\"]}\n```"
        )
        result = normalizer.normalize(raw)
        assert result == '{"summary_of_turns": "1-5", "key_decisions": ["Use Redis"]}'

    def test_fence_only(self):
        """Strips markdown fence with no thinking block."""
        normalizer = ResponseNormalizer()
        raw = '```json\n{"a": 1}\n```'
        assert normalizer.normalize(raw) == '{"a": 1}'

    def test_preamble_only(self):
        """Strips preamble text before JSON."""
        normalizer = ResponseNormalizer()
        raw = 'Sure, here is the JSON output:\n{"b": 2}'
        assert normalizer.normalize(raw) == '{"b": 2}'

    def test_non_json_returns_stripped_text(self):
        """Non-JSON text is returned with whitespace stripped (no braces → unchanged by ExtractJsonObject)."""
        normalizer = ResponseNormalizer()
        raw = "  plain text response  "
        result = normalizer.normalize(raw)
        assert result == "plain text response"


class TestResponseNormalizerCustomChain:
    """Tests for ResponseNormalizer with a custom step chain."""

    def test_single_step(self):
        """Custom chain with a single step works correctly."""
        normalizer = ResponseNormalizer(steps=[StripMarkdownFencesStep()])
        raw = "```json\n{\"x\": 1}\n```"
        assert normalizer.normalize(raw) == '{"x": 1}'

    def test_empty_steps(self):
        """Empty step chain returns raw text unchanged."""
        normalizer = ResponseNormalizer(steps=[])
        raw = "<think>ignored</think>{}"
        assert normalizer.normalize(raw) == raw

    def test_duck_typed_custom_step(self):
        """A duck-typed class matching the Protocol works as a custom step."""

        class UpperCaseStep:
            def normalize(self, inp: NormalizationInput) -> NormalizationOutput:
                result = inp.text.upper()
                return NormalizationOutput(text=result, was_modified=result != inp.text)

        normalizer = ResponseNormalizer(steps=[UpperCaseStep()])
        assert normalizer.normalize("hello") == "HELLO"

    def test_thinking_then_fence_chain(self):
        """Explicit two-step chain: strip thinking then strip fences."""
        normalizer = ResponseNormalizer(steps=[StripThinkingBlocksStep(), StripMarkdownFencesStep()])
        raw = "<think>thought</think>\n```json\n{\"ok\": true}\n```"
        assert normalizer.normalize(raw) == '{"ok": true}'
