"""Tests for normalization steps."""

import pytest

from sr2.normalization.steps import (
    ExtractJsonObjectStep,
    NormalizationInput,
    StripMarkdownFencesStep,
    StripThinkingBlocksStep,
    get_step,
)


class TestStripThinkingBlocksStep:
    """Tests for StripThinkingBlocksStep."""

    def _run(self, text: str) -> str:
        step = StripThinkingBlocksStep()
        return step.normalize(NormalizationInput(text=text)).text

    def test_removes_think_block(self):
        """Removes <think>…</think> and leaves the rest."""
        raw = "<think>internal reasoning here</think>\n{\"key\": \"value\"}"
        assert self._run(raw) == '{"key": "value"}'

    def test_removes_thinking_block(self):
        """Removes <thinking>…</thinking> variant."""
        raw = "<thinking>\nsome thought\n</thinking>\n{\"a\": 1}"
        assert self._run(raw) == '{"a": 1}'

    def test_case_insensitive(self):
        """Tag matching is case-insensitive."""
        raw = "<THINK>upper case</THINK>\n{\"x\": 2}"
        assert self._run(raw) == '{"x": 2}'

    def test_multiline_content(self):
        """Handles multi-line content inside tags."""
        raw = "<think>\nline one\nline two\nline three\n</think>\n{\"z\": 3}"
        assert self._run(raw) == '{"z": 3}'

    def test_no_op_when_absent(self):
        """Returns text unchanged when no think tags present."""
        raw = '{"clean": true}'
        result = StripThinkingBlocksStep().normalize(NormalizationInput(text=raw))
        assert result.text == raw
        assert result.was_modified is False

    def test_was_modified_true_when_stripped(self):
        """was_modified is True when a block was removed."""
        raw = "<think>x</think>{}"
        result = StripThinkingBlocksStep().normalize(NormalizationInput(text=raw))
        assert result.was_modified is True


class TestStripMarkdownFencesStep:
    """Tests for StripMarkdownFencesStep."""

    def _run(self, text: str) -> str:
        step = StripMarkdownFencesStep()
        return step.normalize(NormalizationInput(text=text)).text

    def test_removes_json_fence(self):
        """Removes ```json opening and closing ``` fence."""
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert self._run(raw) == '{"key": "value"}'

    def test_removes_plain_fence(self):
        """Removes plain ``` opening and closing fence."""
        raw = "```\n{\"key\": \"value\"}\n```"
        assert self._run(raw) == '{"key": "value"}'

    def test_no_op_when_absent(self):
        """Returns text unchanged when no fences present."""
        raw = '{"clean": true}'
        result = StripMarkdownFencesStep().normalize(NormalizationInput(text=raw))
        assert result.text == raw
        assert result.was_modified is False

    def test_was_modified_true_when_stripped(self):
        """was_modified is True when fences were removed."""
        raw = "```json\n{}\n```"
        result = StripMarkdownFencesStep().normalize(NormalizationInput(text=raw))
        assert result.was_modified is True

    def test_removes_uppercase_json_fence(self):
        """Removes ```JSON (uppercase) fence."""
        raw = '```JSON\n{"key": "value"}\n```'
        assert self._run(raw) == '{"key": "value"}'

    def test_removes_jsonc_fence(self):
        """Removes ```jsonc fence."""
        raw = '```jsonc\n{"key": "value"}\n```'
        assert self._run(raw) == '{"key": "value"}'

    def test_ignores_non_json_fence(self):
        """Does not strip fences with non-JSON language tags like ```python."""
        raw = '```python\nprint("hello")\n```'
        assert self._run(raw) == raw


class TestExtractJsonObjectStep:
    """Tests for ExtractJsonObjectStep."""

    def _run(self, text: str) -> str:
        step = ExtractJsonObjectStep()
        return step.normalize(NormalizationInput(text=text)).text

    def test_strips_preamble(self):
        """Strips text before the first brace."""
        raw = 'Here is the JSON: {"key": "value"}'
        assert self._run(raw) == '{"key": "value"}'

    def test_strips_postamble(self):
        """Strips text after the last brace."""
        raw = '{"key": "value"} Hope that helps!'
        assert self._run(raw) == '{"key": "value"}'

    def test_no_braces_unchanged(self):
        """Returns text unchanged when no braces present."""
        raw = "no json here"
        result = ExtractJsonObjectStep().normalize(NormalizationInput(text=raw))
        assert result.text == raw
        assert result.was_modified is False

    def test_nested_objects_preserved(self):
        """Nested objects are preserved (rfind gets the outermost closing brace)."""
        raw = 'preamble {"outer": {"inner": 1}} postamble'
        assert self._run(raw) == '{"outer": {"inner": 1}}'

    def test_clean_json_unchanged(self):
        """Clean JSON with no surrounding text is returned as-is."""
        raw = '{"a": 1}'
        result = ExtractJsonObjectStep().normalize(NormalizationInput(text=raw))
        assert result.text == raw
        assert result.was_modified is False

    def test_was_modified_true_when_extracted(self):
        """was_modified is True when text was sliced."""
        raw = 'preamble {"a": 1}'
        result = ExtractJsonObjectStep().normalize(NormalizationInput(text=raw))
        assert result.was_modified is True

    def test_extracts_json_array(self):
        """Extracts a JSON array from surrounding text."""
        raw = 'Here is the list: [{"a": 1}, {"b": 2}]'
        assert self._run(raw) == '[{"a": 1}, {"b": 2}]'

    def test_clean_json_array_unchanged(self):
        """Clean JSON array passes through unchanged."""
        raw = '[{"a": 1}]'
        result = ExtractJsonObjectStep().normalize(NormalizationInput(text=raw))
        assert result.text == raw
        assert result.was_modified is False

    def test_array_before_object_picks_array(self):
        """When array starts before object, extracts the array."""
        raw = 'result: [{"a": 1}]'
        assert self._run(raw) == '[{"a": 1}]'

    def test_object_before_array_picks_object(self):
        """When object starts before array, extracts the object."""
        raw = 'result: {"items": [1, 2, 3]}'
        assert self._run(raw) == '{"items": [1, 2, 3]}'

    def test_no_brackets_or_braces_unchanged(self):
        """Text with neither braces nor brackets is unchanged."""
        raw = "just plain text"
        result = ExtractJsonObjectStep().normalize(NormalizationInput(text=raw))
        assert result.text == raw
        assert result.was_modified is False


class TestGetStep:
    """Tests for the get_step registry function."""

    def test_get_strip_thinking_blocks(self):
        """get_step returns StripThinkingBlocksStep."""
        step = get_step("strip_thinking_blocks")
        assert isinstance(step, StripThinkingBlocksStep)

    def test_get_strip_markdown_fences(self):
        """get_step returns StripMarkdownFencesStep."""
        step = get_step("strip_markdown_fences")
        assert isinstance(step, StripMarkdownFencesStep)

    def test_get_extract_json_object(self):
        """get_step returns ExtractJsonObjectStep."""
        step = get_step("extract_json_object")
        assert isinstance(step, ExtractJsonObjectStep)

    def test_unknown_name_raises_key_error(self):
        """get_step raises KeyError for unknown name."""
        with pytest.raises(KeyError, match="Unknown normalization step"):
            get_step("nonexistent_step")
