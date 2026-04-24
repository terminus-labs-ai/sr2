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
    """Removes markdown code fences (```json, ```JSON, ```jsonc, plain ```, etc.)."""

    _PATTERN = re.compile(
        r"^```(?:jsonc?|JSON)?\s*\n(.*?)^```\s*$",
        re.MULTILINE | re.DOTALL,
    )

    def normalize(self, inp: NormalizationInput) -> NormalizationOutput:
        result = self._PATTERN.sub(r"\1", inp.text).strip()
        return NormalizationOutput(text=result, was_modified=result != inp.text)


class ExtractJsonObjectStep:
    """Extracts the outermost JSON object or array from surrounding text."""

    def normalize(self, inp: NormalizationInput) -> NormalizationOutput:
        obj_start = inp.text.find("{")
        obj_end = inp.text.rfind("}")
        arr_start = inp.text.find("[")
        arr_end = inp.text.rfind("]")

        has_obj = obj_start != -1 and obj_end != -1 and obj_end > obj_start
        has_arr = arr_start != -1 and arr_end != -1 and arr_end > arr_start

        if has_obj and has_arr:
            # Pick whichever starts first (outermost)
            if arr_start < obj_start:
                start, end = arr_start, arr_end
            else:
                start, end = obj_start, obj_end
        elif has_obj:
            start, end = obj_start, obj_end
        elif has_arr:
            start, end = arr_start, arr_end
        else:
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
