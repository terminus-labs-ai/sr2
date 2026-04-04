"""Sample evaluation suites for common context engineering scenarios."""

from sr2.eval.models import EvalCase


def create_coherence_suite() -> list[EvalCase]:
    """Create eval suite for context coherence (memory retention).

    Tests whether the agent maintains awareness of key facts and decisions
    across long conversations.
    """
    return [
        EvalCase(
            id="coherence_001",
            name="Long conversation with decision tracking",
            description="50-turn conversation where agent makes and recalls decisions",
            system_prompt="You are a helpful project planning assistant. Track decisions and deliverables.",
            conversation_turns=[
                ("We're building a payment system", "I'll help you plan a payment system."),
                (
                    "Should we use Stripe or custom?",
                    "Stripe is recommended for faster time-to-market.",
                ),
                ("Let's go with Stripe", "Great choice. Stripe is reliable and well-documented."),
                (
                    "What about PCI compliance?",
                    "Stripe handles PCI compliance, you don't need to worry.",
                ),
                (
                    "Can you remind me what we chose for payments?",
                    "We decided to use Stripe for PCI compliance handling.",
                ),
            ],
            expected_key_facts=[
                "Stripe",
                "payment system",
                "PCI compliance",
            ],
            expected_decisions=[
                "Stripe",
                "reliable",
            ],
            expected_tokens=2000,
            tags=["coherence", "memory"],
        ),
        EvalCase(
            id="coherence_002",
            name="Multi-topic conversation with context switching",
            description="Conversation that switches topics multiple times",
            system_prompt="You are a versatile assistant helping with multiple tasks.",
            conversation_turns=[
                ("I need to refactor my database schema", "I can help you design a better schema."),
                ("Actually, let's talk about UI first", "Sure, let's focus on the UI instead."),
                (
                    "Never mind, back to database",
                    "No problem, let's return to the database schema.",
                ),
                ("What changes did we discuss?", "We discussed refactoring your database schema."),
            ],
            expected_key_facts=[
                "database schema",
                "UI",
                "refactor",
            ],
            expected_decisions=[],
            expected_tokens=1500,
            tags=["coherence", "context_switching"],
        ),
    ]


def create_compaction_suite() -> list[EvalCase]:
    """Create eval suite for context compaction effectiveness.

    Tests whether compaction preserves important information while
    reducing token usage.
    """
    return [
        EvalCase(
            id="compaction_001",
            name="Large tool output compaction",
            description="Many tool calls should be compacted to summaries",
            system_prompt="You are a code analysis assistant. Use tools to analyze code.",
            conversation_turns=[
                (
                    "Analyze this code for performance issues",
                    "I'll analyze the code for you.",
                ),
                (
                    "Here's 500 lines of code...",
                    "I found 3 performance bottlenecks.",
                ),
            ],
            expected_key_facts=[
                "performance",
                "bottlenecks",
            ],
            expected_decisions=[],
            expected_tokens=800,  # Compacted from ~2000 raw
            tags=["compaction", "tool_output"],
        ),
        EvalCase(
            id="compaction_002",
            name="File content compaction",
            description="Large file contents should reference paths not full text",
            system_prompt="You are a code reviewer.",
            conversation_turns=[
                (
                    "Review my service file at /app/services/auth.py",
                    "I'll review the auth service.",
                ),
                (
                    "What are the main issues?",
                    "The main issue is the security context handling.",
                ),
            ],
            expected_key_facts=[
                "auth",
                "security",
            ],
            expected_decisions=[],
            expected_tokens=600,
            tags=["compaction", "file_content"],
        ),
    ]


def create_summarization_suite() -> list[EvalCase]:
    """Create eval suite for summarization quality.

    Tests whether summarization preserves critical information while
    freeing up token budget.
    """
    return [
        EvalCase(
            id="summarization_001",
            name="Complex decision summary preservation",
            description="Summarization should preserve key architecture decisions",
            system_prompt="You are an architecture advisor for a startup.",
            conversation_turns=[
                ("Should we use microservices or monolith?", "Each has tradeoffs."),
                ("We're planning 10 services", "Microservices can work at that scale."),
                ("But we're 3 engineers", "Monolith might be better for small teams."),
                ("Let's do monolith first", "Good choice for your current size."),
                ("In 6 months we'll have 10 engineers", "Plan microservice migration then."),
                (
                    "What was our decision?",
                    "You chose monolith initially, migrating to microservices when you scale.",
                ),
            ],
            expected_key_facts=[
                "monolith",
                "microservices",
                "3 engineers",
                "10 engineers",
            ],
            expected_decisions=[
                "monolith",
                "migrate",
            ],
            expected_tokens=1200,
            tags=["summarization", "decisions"],
        ),
        EvalCase(
            id="summarization_002",
            name="Discarded exploration preservation",
            description="Summarization should discard dead-end explorations but keep context",
            system_prompt="You are a problem-solving assistant.",
            conversation_turns=[
                ("How should we cache results?", "Redis is a popular option."),
                ("What about in-memory caching?", "Works for single instances."),
                ("But we're distributed", "Redis is better for distributed systems."),
                ("Let's use Redis", "Good choice for your architecture."),
                (
                    "What caching approaches did we explore?",
                    "We explored in-memory and Redis, choosing Redis.",
                ),
            ],
            expected_key_facts=[
                "Redis",
                "distributed",
            ],
            expected_decisions=[
                "Redis",
            ],
            expected_tokens=900,
            tags=["summarization", "exploration"],
        ),
    ]


def create_all_suites() -> dict[str, list[EvalCase]]:
    """Create all standard evaluation suites.

    Returns:
        Dict mapping suite name to list of eval cases
    """
    return {
        "coherence": create_coherence_suite(),
        "compaction": create_compaction_suite(),
        "summarization": create_summarization_suite(),
    }
