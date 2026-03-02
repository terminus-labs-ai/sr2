"""Built-in normalization steps for cleaning raw LLM responses."""

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class NormalizationInput:
    """Raw LLM text to be normalized."""

    text: str
    metadata: dict | None = None


@dataclass
class NormalizationOutput:
    """Result of a normalization step."""

    text: str
    was_modified: bool


class NormalizationStep(Protocol):
    """Protocol for normalization steps."""

    def normalize(self, inp: NormalizationInput) -> NormalizationOutput: ...


class StripThinkingBlocksStep:
    """Removes <think>…</think> and <thinking>…</thinking> blocks."""

    _PATTERN = re.compile(r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

    def normalize(self, inp: NormalizationInput) -> NormalizationOutput:
        result = self._PATTERN.sub("", inp.text).strip()
        return NormalizationOutput(text=result, was_modified=result != inp.text)


class StripMarkdownFencesStep:
    """Removes ```json / ``` opening and closing fences."""

    _PATTERN = re.compile(r"^```(?:json)?\s*|```\s*$", re.MULTILINE)

    def normalize(self, inp: NormalizationInput) -> NormalizationOutput:
        result = self._PATTERN.sub("", inp.text).strip()
        return NormalizationOutput(text=result, was_modified=result != inp.text)


class ExtractJsonObjectStep:
    """Slices text from first '{' to last '}'; returns unchanged if no braces."""

    def normalize(self, inp: NormalizationInput) -> NormalizationOutput:
        start = inp.text.find("{")
        end = inp.text.rfind("}")
        if start == -1 or end == -1:
            return NormalizationOutput(text=inp.text, was_modified=False)
        result = inp.text[start : end + 1]
        return NormalizationOutput(text=result, was_modified=result != inp.text)


BUILT_IN_STEPS: dict[str, NormalizationStep] = {
    "strip_thinking_blocks": StripThinkingBlocksStep(),
    "strip_markdown_fences": StripMarkdownFencesStep(),
    "extract_json_object": ExtractJsonObjectStep(),
}


def get_step(name: str) -> NormalizationStep:
    """Get a normalization step by name. Raises KeyError if unknown."""
    if name not in BUILT_IN_STEPS:
        raise KeyError(f"Unknown normalization step: {name}")
    return BUILT_IN_STEPS[name]
