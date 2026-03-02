"""ResponseNormalizer: threads raw LLM text through a chain of normalization steps."""

from sr2.normalization.steps import (
    ExtractJsonObjectStep,
    NormalizationInput,
    NormalizationStep,
    StripMarkdownFencesStep,
    StripThinkingBlocksStep,
)


class ResponseNormalizer:
    """Cleans raw LLM responses before JSON parsing."""

    def __init__(self, steps: list[NormalizationStep] | None = None):
        """Default chain: StripThinkingBlocks → StripMarkdownFences → ExtractJsonObject."""
        self._steps = [
            StripThinkingBlocksStep(),
            StripMarkdownFencesStep(),
            ExtractJsonObjectStep(),
        ] if steps is None else steps

    def normalize(self, raw: str) -> str:
        """Thread raw through each step in order, return final text."""
        inp = NormalizationInput(text=raw)
        for step in self._steps:
            out = step.normalize(inp)
            inp = NormalizationInput(text=out.text, metadata=inp.metadata)
        return inp.text
