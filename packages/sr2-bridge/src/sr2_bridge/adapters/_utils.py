"""Shared utilities for bridge adapters."""

from __future__ import annotations


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# Tool names -> content_type mapping.
# Keys are substrings matched against the tool name (case-insensitive).
_TOOL_CONTENT_TYPE_MAP = {
    # File reading tools
    "read": "file_content",
    "cat": "file_content",
    "view_file": "file_content",
    # Code execution tools
    "bash": "code_execution",
    "shell": "code_execution",
    "execute": "code_execution",
    "run": "code_execution",
    "terminal": "code_execution",
}


def _classify_tool_name(
    tool_name: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Classify a tool name into a content_type for compaction rules."""
    lower = tool_name.lower()
    # User overrides take priority
    if overrides:
        for marker, content_type in overrides.items():
            if marker.lower() in lower:
                return content_type
    for marker, content_type in _TOOL_CONTENT_TYPE_MAP.items():
        if marker in lower:
            return content_type
    # Default: any tool output we don't specifically classify
    return "tool_output"


_FILE_PATH_KEYS = ("file_path", "path", "filename", "file")


def _extract_file_path(
    tool_name: str,
    arguments: dict,
    overrides: dict[str, str] | None = None,
) -> str | None:
    """Extract file path from tool arguments if the tool reads files.

    Returns the path string if the tool is classified as file_content and
    the arguments contain a recognizable file path key, else None.
    """
    content_type = _classify_tool_name(tool_name, overrides)
    if content_type != "file_content":
        return None
    for key in _FILE_PATH_KEYS:
        val = arguments.get(key)
        if val and isinstance(val, str):
            return val
    return None
