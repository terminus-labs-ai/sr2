"""Public API surface — what the top-level ``sr2`` package exports."""


def test_sr2_class_importable_from_package_root():
    """`from sr2 import SR2` must resolve to the orchestrator class."""
    from sr2 import SR2
    from sr2.orchestrator import SR2 as OrchestratorSR2

    assert SR2 is OrchestratorSR2


def test_sr2_listed_in_package_all():
    import sr2

    assert "SR2" in sr2.__all__


def test_core_models_still_exported():
    """Adding SR2 must not drop the existing model exports."""
    from sr2 import (  # noqa: F401
        ContentBlock,
        Message,
        TextBlock,
        TokenUsage,
        ToolDefinition,
        ToolResultBlock,
        ToolUseBlock,
    )
