"""Tests for sr2.runtime.tools — RuntimeToolExecutor."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from sr2.runtime.config import ToolConfig
from sr2.runtime.tools import RuntimeToolExecutor


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture(autouse=True)
def _cleanup_test_modules():
    """Remove test modules from sys.modules after each test."""
    before = set(sys.modules.keys())
    yield
    for key in list(sys.modules.keys()):
        if key not in before:
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Sync tool loading and execution
# ---------------------------------------------------------------------------


def test_load_sync_tool_by_name():
    """Load a module that exposes a function matching the tool name."""

    def greet(name: str) -> str:
        return f"hello {name}"

    _make_module("_test_tools_sync", greet=greet)
    config = [ToolConfig(name="greet", module="_test_tools_sync")]

    executor = RuntimeToolExecutor(config)
    assert "greet" in executor.tools


async def test_execute_sync_tool():
    """Sync tools are callable through the async execute interface."""

    def add(a: int, b: int) -> int:
        return a + b

    _make_module("_test_tools_add", add=add)
    executor = RuntimeToolExecutor([ToolConfig(name="add", module="_test_tools_add")])

    result = await executor.execute("add", {"a": 2, "b": 3})
    assert result == 5


# ---------------------------------------------------------------------------
# Async tool loading and execution
# ---------------------------------------------------------------------------


def test_load_async_tool():
    """Async functions are accepted as tool handlers."""

    async def fetch(url: str) -> str:
        return f"fetched {url}"

    _make_module("_test_tools_async", fetch=fetch)
    executor = RuntimeToolExecutor([ToolConfig(name="fetch", module="_test_tools_async")])
    assert "fetch" in executor.tools


async def test_execute_async_tool():
    """Async tools are awaited properly."""

    async def fetch(url: str) -> str:
        return f"fetched {url}"

    _make_module("_test_tools_async_exec", fetch=fetch)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="fetch", module="_test_tools_async_exec")]
    )

    result = await executor.execute("fetch", {"url": "https://example.com"})
    assert result == "fetched https://example.com"


# ---------------------------------------------------------------------------
# Fallback to run() function
# ---------------------------------------------------------------------------


def test_load_tool_via_run_fallback():
    """When no function matches the tool name, fall back to run()."""

    def run(x: int) -> int:
        return x * 2

    _make_module("_test_tools_run_fallback", run=run)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="double", module="_test_tools_run_fallback")]
    )
    assert "double" in executor.tools


async def test_execute_tool_via_run_fallback():
    def run(x: int) -> int:
        return x * 2

    _make_module("_test_tools_run_exec", run=run)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="double", module="_test_tools_run_exec")]
    )

    result = await executor.execute("double", {"x": 5})
    assert result == 10


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_module_raises_import_error():
    """ImportError when the module path doesn't exist."""
    config = [ToolConfig(name="nope", module="_test_tools_nonexistent_xyz")]
    with pytest.raises((ImportError, ModuleNotFoundError)):
        RuntimeToolExecutor(config)


def test_module_without_matching_function_raises_runtime_error():
    """RuntimeError when module has neither the named function nor run()."""
    _make_module("_test_tools_empty")
    config = [ToolConfig(name="missing", module="_test_tools_empty")]

    with pytest.raises(RuntimeError, match="has no 'missing' or 'run' function"):
        RuntimeToolExecutor(config)


async def test_execute_unknown_tool_raises_key_error():
    """KeyError lists available tools when an unknown tool is called."""

    def greet(name: str) -> str:
        return f"hi {name}"

    _make_module("_test_tools_keyerr", greet=greet)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="greet", module="_test_tools_keyerr")]
    )

    with pytest.raises(KeyError, match="Unknown tool: 'nope'"):
        await executor.execute("nope", {})

    # Verify available tools are listed in the error
    with pytest.raises(KeyError, match="greet"):
        await executor.execute("nope", {})


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


def test_get_schemas_returns_openai_format():
    """get_schemas() returns a list of OpenAI function-calling dicts."""

    def search(query: str, limit: int = 10) -> str:
        """Search for items."""
        return ""

    _make_module("_test_tools_schema", search=search)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="search", module="_test_tools_schema")]
    )

    schemas = executor.get_schemas()
    assert len(schemas) == 1

    schema = schemas[0]
    assert schema["name"] == "search"
    assert schema["description"] == "Search for items."
    assert schema["parameters"]["type"] == "object"
    assert "query" in schema["parameters"]["properties"]
    assert "limit" in schema["parameters"]["properties"]
    assert "query" in schema["parameters"]["required"]
    assert "limit" not in schema["parameters"]["required"]


def test_auto_definition_parameter_types():
    """Auto-generated definitions map Python types to JSON Schema types."""

    def typed_fn(name: str, count: int, rate: float, flag: bool) -> str:
        """A typed function."""
        return ""

    _make_module("_test_tools_types", typed_fn=typed_fn)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="typed_fn", module="_test_tools_types")]
    )

    schema = executor.get_schemas()[0]
    props = schema["parameters"]["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["rate"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"

    # All params are required (no defaults)
    assert sorted(schema["parameters"]["required"]) == [
        "count", "flag", "name", "rate"
    ]


def test_tool_definition_attribute_used_over_auto():
    """When a module has TOOL_DEFINITION, use that instead of auto-generating."""
    custom_defn = {
        "name": "custom_tool",
        "description": "A hand-crafted definition",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input"},
            },
            "required": ["input"],
        },
    }

    def custom_tool(input: str) -> str:
        return input.upper()

    _make_module(
        "_test_tools_custom_defn",
        custom_tool=custom_tool,
        TOOL_DEFINITION=custom_defn,
    )
    executor = RuntimeToolExecutor(
        [ToolConfig(name="custom_tool", module="_test_tools_custom_defn")]
    )

    schemas = executor.get_schemas()
    assert len(schemas) == 1
    assert schemas[0] is custom_defn
    assert schemas[0]["description"] == "A hand-crafted definition"


# ---------------------------------------------------------------------------
# Multiple tools
# ---------------------------------------------------------------------------


def test_load_multiple_tools():
    """Multiple tool configs are loaded and independently accessible."""

    def alpha() -> str:
        return "a"

    def beta() -> str:
        return "b"

    _make_module("_test_tools_alpha", alpha=alpha)
    _make_module("_test_tools_beta", beta=beta)

    executor = RuntimeToolExecutor([
        ToolConfig(name="alpha", module="_test_tools_alpha"),
        ToolConfig(name="beta", module="_test_tools_beta"),
    ])

    assert "alpha" in executor.tools
    assert "beta" in executor.tools
    assert len(executor.get_schemas()) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_auto_definition_no_docstring():
    """Auto-generated definition works for functions without a docstring."""

    def bare(x: str) -> str:
        return x

    _make_module("_test_tools_bare", bare=bare)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="bare", module="_test_tools_bare")]
    )

    schema = executor.get_schemas()[0]
    assert schema["description"] == ""


def test_auto_definition_skips_self_and_cls():
    """The self/cls parameters are excluded from auto-generated definitions."""

    def method(self, query: str) -> str:
        """A method-like function."""
        return query

    _make_module("_test_tools_method", method=method)
    executor = RuntimeToolExecutor(
        [ToolConfig(name="method", module="_test_tools_method")]
    )

    schema = executor.get_schemas()[0]
    assert "self" not in schema["parameters"]["properties"]
    assert "query" in schema["parameters"]["properties"]
