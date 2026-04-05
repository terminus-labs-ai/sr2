"""Pydantic v2 configuration for the Claude Code bridge adapter."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaudeCodeAdapterConfig(BaseModel):
    """Configuration for the Claude Code CLI bridge adapter.

    Controls how ``claude -p`` is spawned: binary path, tool allowlist,
    permission mode, concurrency limits, and timeout.  Auth is handled via
    the ``CLAUDE_CODE_OAUTH_TOKEN`` environment variable (1-year token
    from ``claude setup-token``).

    When ``bare`` is False (default), Claude Code uses OAuth auth and loads
    its own CLAUDE.md/memory/hooks — harmless because SR2 controls the
    actual context pipeline via ``--system-prompt``.  Set ``bare: true``
    only when using ``ANTHROPIC_API_KEY`` (API billing) instead of OAuth.
    """

    path: str = Field(
        default="claude",
        description="Path to the claude CLI binary.",
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "Read",
            "Glob",
            "Grep",
            "Bash",
            "Write",
            "Edit",
            "Agent",
            "WebSearch",
            "WebFetch",
        ],
        description="Tools to pre-approve via --allowedTools.",
    )
    bare: bool = Field(
        default=False,
        description="Run with --bare flag.  Blocks OAuth auth "
        "(CLAUDE_CODE_OAUTH_TOKEN).  Only enable when using ANTHROPIC_API_KEY.",
    )
    dangerously_skip_permissions: bool = Field(
        default=True,
        description="Bypass all permission checks (--dangerously-skip-permissions). "
        "Required for unattended/container execution with no TTY.",
    )
    permission_mode: str | None = Field(
        default=None,
        description="Permission mode (default, acceptEdits, bypassPermissions). "
        "Ignored when dangerously_skip_permissions is True.",
    )
    max_turns: int | None = Field(
        default=None,
        description="Max agentic turns per Claude Code invocation.",
    )
    max_budget_usd: float | None = Field(
        default=None,
        description="Max cost in USD per Claude Code invocation.",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        description="Max concurrent Claude Code subprocesses.",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=10,
        description="Subprocess timeout in seconds.  Process is killed after this.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Working directory for Claude Code subprocess.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for Claude Code subprocess.",
    )
