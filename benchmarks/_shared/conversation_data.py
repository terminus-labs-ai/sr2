"""Conversation data generation for benchmarks.

Provides realistic multi-turn coding agent conversations using real
SR2 ConversationTurn objects.
"""

import json

from sr2.compaction.engine import ConversationTurn


SYSTEM_PROMPT = """You are EDI, a senior software engineer AI assistant. You have access to
file system tools, code execution, web search, and task management. You help the user
with software development tasks by reading code, writing code, running tests, and
debugging issues. You follow best practices: explain your reasoning, test before committing,
and ask clarifying questions when requirements are ambiguous.

## Tool Usage Guidelines
- Always read existing code before modifying it
- Run tests after every change
- Use git commits with descriptive messages
- When debugging, check logs first

## Communication Style
- Be concise but thorough
- Use code blocks for code
- Explain trade-offs when making design decisions
"""

TOOL_DEFINITIONS = """Available tools:
- read_file(path: str) -> str: Read file contents
- write_file(path: str, content: str) -> bool: Write file
- run_command(cmd: str) -> {exit_code: int, stdout: str, stderr: str}: Execute shell command
- search_code(query: str, path: str) -> list[{file: str, line: int, match: str}]: Search codebase
- create_task(title: str, description: str) -> {id: int, status: str}: Create task in tracker
- web_search(query: str) -> list[{title: str, url: str, snippet: str}]: Search the web
"""

RETRIEVED_MEMORIES = """## Recalled Context
- User prefers Python 3.12+ features (match statements, type hints)
- Project uses pytest with coverage >90% target
- Previous session: refactored the auth module last week
- User's timezone: PST, prefers concise responses
"""


def _gen_tool_output(turn_num: int) -> str:
    """Generate realistic tool outputs that grow in size."""
    outputs = {
        "file_read": lambda n: "\n".join(
            [
                f"    line {i}: {'import ' if i < 5 else 'def ' if i % 10 == 0 else '    '}"
                f"something_{i} = {i * n}"
                for i in range(1, 60 + n * 2)
            ]
        ),
        "test_run": lambda n: "\n".join(
            [
                f"tests/test_module_{n}.py::test_case_{i} {'PASSED' if i % 7 != 0 else 'FAILED'}"
                for i in range(1, 30 + n)
            ]
            + [f"\n{'=' * 60}", f"{25 + n} passed, {3 + (n % 3)} failed, {n % 2} warnings"]
        ),
        "search_results": lambda n: json.dumps(
            [
                {
                    "file": f"src/module_{i}.py",
                    "line": i * 10 + n,
                    "match": f"def process_{i}(data): # handler for case {i}",
                }
                for i in range(1, 15 + n)
            ],
            indent=2,
        ),
        "git_log": lambda n: "\n".join(
            [
                f"{'a' * 7}{i:03d}  2025-01-{10 + i:02d}  Fix: update handler for case {i + n}"
                for i in range(1, 20 + n)
            ]
        ),
        "web_search": lambda n: json.dumps(
            [
                {
                    "title": f"Result {i}: How to implement pattern {n}",
                    "url": f"https://example.com/article-{i}-{n}",
                    "snippet": "The recommended approach involves using a factory method "
                    "with dependency injection. This ensures loose coupling "
                    "between components and makes testing straightforward. "
                    "Consider also using the adapter pattern for external deps.",
                }
                for i in range(1, 8)
            ],
            indent=2,
        ),
    }
    kinds = list(outputs.keys())
    return outputs[kinds[turn_num % len(kinds)]](turn_num)


# The base conversation script — a realistic coding agent session.
_BASE_SCRIPT: list[tuple[str, str | None]] = [
    (
        "user",
        "Can you look at the auth module? I think there's a bug in the token refresh logic.",
    ),
    ("assistant", "I'll read the auth module to understand the current implementation."),
    ("tool", None),
    (
        "assistant",
        "I see the issue — the refresh token check on line 45 uses "
        "`>=` instead of `>`. The token expires exactly at the "
        "threshold. Let me check the tests too.",
    ),
    ("tool", None),
    (
        "assistant",
        "Existing tests don't cover the boundary case. I'll write a fix and add a targeted test.",
    ),
    ("user", "Yes, please fix it and add the test."),
    ("tool", None),
    ("tool", None),
    (
        "assistant",
        "Fixed. Token refresh triggers correctly at boundary now. "
        "Added test_token_refresh_at_boundary. All 47 tests pass.",
    ),
    (
        "user",
        "Great. Can you also look at the rate limiter? Users report "
        "429 errors during window boundaries.",
    ),
    ("assistant", "Let me search for the rate limiter and check error patterns."),
    ("tool", None),
    ("tool", None),
    (
        "assistant",
        "Found it. Fixed-window limiter lets 2x requests at boundary. "
        "I'll convert to sliding window using collections.deque.",
    ),
    ("user", "Makes sense. Is there a Redis dependency? I'd prefer in-memory."),
    ("tool", None),
    (
        "assistant",
        "No Redis — all in-memory with dict. I'll keep it that way. "
        "The deque approach bounds memory automatically.",
    ),
    ("tool", None),
    ("tool", None),
    (
        "assistant",
        "Sliding window rate limiter done. All tests pass. Memory "
        "usage bounded by maxlen parameter.",
    ),
    ("user", "Perfect. Search for TODO comments we should address?"),
    ("tool", None),
    (
        "assistant",
        "Found 18 TODOs. Critical: 3 in payment module (error handling), "
        "2 in notification service (retry logic). Want me to create tasks?",
    ),
    ("user", "Create tasks for the payment module TODOs — those are critical."),
    ("tool", None),
    ("tool", None),
    ("tool", None),
    (
        "assistant",
        "Created 3 high-priority tasks in tracker with descriptions "
        "of required changes for each payment module TODO.",
    ),
    ("user", "Thanks, that's all for now."),
]

_TOOLS_USED = [
    "read_file",
    "run_command",
    "write_file",
    "run_command",
    "search_code",
    "read_file",
    "search_code",
    "write_file",
    "run_command",
    "search_code",
    "create_task",
    "create_task",
    "create_task",
]


def generate_conversation(num_turns: int = 30) -> list[ConversationTurn]:
    """Generate a realistic multi-turn coding agent conversation.

    Uses real ConversationTurn from sr2.compaction.engine.
    """
    tool_idx = 0
    turns: list[ConversationTurn] = []

    for i, (role, text) in enumerate(_BASE_SCRIPT[:num_turns]):
        if role == "tool":
            tool_name = _TOOLS_USED[tool_idx % len(_TOOLS_USED)]
            tool_idx += 1
            turns.append(
                ConversationTurn(
                    turn_number=i,
                    role="tool_result",
                    content=_gen_tool_output(i),
                    content_type="tool_output",
                    metadata={"tool_name": tool_name},
                )
            )
        else:
            turns.append(
                ConversationTurn(
                    turn_number=i,
                    role=role,
                    content=text,
                )
            )

    return turns


# ── Anchor decisions injected into long conversations ──

_ANCHOR_DECISIONS: dict[int, tuple[str, str, str]] = {
    # turn_number -> (user_msg, assistant_msg, keyword)
    5: (
        "What database should we use for this project?",
        "Let's use PostgreSQL — it has strong JSON support and is production-proven.",
        "postgresql",
    ),
    15: (
        "How should we handle authentication?",
        "We'll go with JWT auth — stateless, easy to scale horizontally.",
        "jwt",
    ),
    25: (
        "What should we use for the message queue?",
        "Let's use RabbitMQ for the message queue — it handles our throughput needs.",
        "rabbitmq",
    ),
    35: (
        "What frontend framework should we pick?",
        "We should use React with TypeScript — the team has the most experience with it.",
        "react",
    ),
    45: (
        "What should we use for container orchestration?",
        "We'll deploy with Kubernetes — it gives us auto-scaling and self-healing.",
        "kubernetes",
    ),
}

# (turn_number, question, expected_keyword)
RECALL_QUESTIONS: list[tuple[int, str, str]] = [
    (5, "What database did the team decide to use?", "postgresql"),
    (15, "What authentication method was chosen?", "jwt"),
    (25, "What message queue system was selected?", "rabbitmq"),
    (35, "What frontend framework did the team pick?", "react"),
    (45, "What container orchestration tool was decided on?", "kubernetes"),
]


def generate_long_conversation(num_turns: int = 50) -> list[ConversationTurn]:
    """Generate an extended conversation with anchor decisions at known turns.

    Anchor decisions are placed at turns 5, 15, 25, 35, 45 so the coherence
    benchmark can test recall.
    """
    turns: list[ConversationTurn] = []
    tool_idx = 0

    for i in range(num_turns):
        if i in _ANCHOR_DECISIONS:
            user_msg, assistant_msg, _ = _ANCHOR_DECISIONS[i]
            turns.append(ConversationTurn(turn_number=i, role="user", content=user_msg))
            turns.append(ConversationTurn(turn_number=i, role="assistant", content=assistant_msg))
        elif i < len(_BASE_SCRIPT):
            role, text = _BASE_SCRIPT[i]
            if role == "tool":
                tool_name = _TOOLS_USED[tool_idx % len(_TOOLS_USED)]
                tool_idx += 1
                turns.append(
                    ConversationTurn(
                        turn_number=i,
                        role="tool_result",
                        content=_gen_tool_output(i),
                        content_type="tool_output",
                        metadata={"tool_name": tool_name},
                    )
                )
            else:
                turns.append(ConversationTurn(turn_number=i, role=role, content=text))
        else:
            # Generate additional turns beyond the base script
            if i % 3 == 0:
                turns.append(
                    ConversationTurn(
                        turn_number=i,
                        role="user",
                        content=f"Can you check the module_{i} implementation?",
                    )
                )
            elif i % 3 == 1:
                turns.append(
                    ConversationTurn(
                        turn_number=i,
                        role="tool_result",
                        content=_gen_tool_output(i),
                        content_type="tool_output",
                        metadata={"tool_name": _TOOLS_USED[tool_idx % len(_TOOLS_USED)]},
                    )
                )
                tool_idx += 1
            else:
                turns.append(
                    ConversationTurn(
                        turn_number=i,
                        role="assistant",
                        content=f"I've reviewed module_{i - 1}. The implementation looks solid. "
                        f"The error handling covers edge cases and tests have good coverage.",
                    )
                )

    return turns
