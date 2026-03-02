"""Built-in compaction rules for replacing verbose content with compact references."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CompactionInput:
    """A piece of content eligible for compaction."""

    content: str
    content_type: str  # "tool_output", "file_content", "code_execution", etc.
    tokens: int  # Estimated token count
    metadata: dict | None = None


@dataclass
class CompactionOutput:
    """Result of compacting a piece of content."""

    content: str  # The compacted content
    tokens: int  # Token count after compaction
    was_compacted: bool  # Whether compaction actually happened
    recovery_hint: str | None = None  # How to re-fetch the original


class CompactionRule(Protocol):
    """Protocol for compaction rules."""

    def compact(self, inp: CompactionInput, config: dict) -> CompactionOutput: ...


class SchemaAndSampleRule:
    """For tool outputs: keep schema + first item + count."""

    def compact(self, inp: CompactionInput, config: dict) -> CompactionOutput:
        max_tokens = config.get("max_compacted_tokens", 80)
        lines = inp.content.strip().split("\n")
        if len(lines) <= 3:
            return CompactionOutput(
                content=inp.content,
                tokens=inp.tokens,
                was_compacted=False,
            )

        sample = "\n".join(lines[:3])
        summary = f"\u2192 {len(lines)} lines. Sample:\n{sample}..."
        est_tokens = len(summary) // 4

        hint = None
        if config.get("recovery_hint"):
            tool_name = inp.metadata.get("tool_name", "the tool") if inp.metadata else "the tool"
            hint = f"Re-fetch with {tool_name}"

        return CompactionOutput(
            content=summary,
            tokens=min(est_tokens, max_tokens),
            was_compacted=True,
            recovery_hint=hint,
        )


class ReferenceRule:
    """For file content: replace with path + metadata."""

    def compact(self, inp: CompactionInput, config: dict) -> CompactionOutput:
        meta = inp.metadata or {}
        path = meta.get("file_path", "unknown")
        line_count = meta.get("line_count", "?")
        language = meta.get("language", "")
        size = meta.get("size", "")

        parts = [f"\u2192 Saved to {path}"]
        detail_parts = []
        if line_count != "?":
            detail_parts.append(f"{line_count} lines")
        if language:
            detail_parts.append(language)
        if size:
            detail_parts.append(size)
        if detail_parts:
            parts[0] += f" ({', '.join(detail_parts)})"

        content = parts[0]
        return CompactionOutput(
            content=content,
            tokens=len(content) // 4,
            was_compacted=True,
            recovery_hint=f'read_file("{path}")',
        )


class ResultSummaryRule:
    """For code execution: keep exit code + truncated output."""

    def compact(self, inp: CompactionInput, config: dict) -> CompactionOutput:
        max_lines = config.get("max_output_lines", 3)
        meta = inp.metadata or {}
        exit_code = meta.get("exit_code", "?")

        lines = inp.content.strip().split("\n")
        truncated = lines[:max_lines]

        status = "\u2713" if str(exit_code) == "0" else "\u2717"
        summary_parts = [f"\u2192 {status} Exit {exit_code}."]
        if truncated:
            summary_parts.append(" ".join(truncated))
        if len(lines) > max_lines:
            summary_parts.append(f"... ({len(lines) - max_lines} more lines)")

        content = " ".join(summary_parts)
        return CompactionOutput(
            content=content,
            tokens=len(content) // 4,
            was_compacted=True,
            recovery_hint=meta.get("result_path"),
        )


class SupersedeRule:
    """For redundant fetches: mark as superseded."""

    def compact(self, inp: CompactionInput, config: dict) -> CompactionOutput:
        meta = inp.metadata or {}
        superseded_by = meta.get("superseded_by_turn", "later turn")
        content = f"\u2192 (superseded by {superseded_by})"
        return CompactionOutput(
            content=content,
            tokens=len(content) // 4,
            was_compacted=True,
        )


class CollapseRule:
    """For confirmations: collapse to one-liner."""

    def compact(self, inp: CompactionInput, config: dict) -> CompactionOutput:
        meta = inp.metadata or {}
        tool_name = meta.get("tool_name", "tool")
        args_summary = meta.get("args_summary", "")

        content = f"\u2192 \u2713 {tool_name}({args_summary})"
        return CompactionOutput(
            content=content,
            tokens=len(content) // 4,
            was_compacted=True,
        )


BUILT_IN_RULES: dict[str, CompactionRule] = {
    "schema_and_sample": SchemaAndSampleRule(),
    "reference": ReferenceRule(),
    "result_summary": ResultSummaryRule(),
    "supersede": SupersedeRule(),
    "collapse": CollapseRule(),
}


def get_rule(strategy: str) -> CompactionRule:
    """Get a compaction rule by strategy name. Raises KeyError if not found."""
    if strategy not in BUILT_IN_RULES:
        raise KeyError(f"Unknown compaction strategy: {strategy}")
    return BUILT_IN_RULES[strategy]
