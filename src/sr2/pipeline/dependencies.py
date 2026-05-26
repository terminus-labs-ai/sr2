"""Pipeline dependency container."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from sr2.protocols.llm import LLMCallable


@dataclasses.dataclass(frozen=True)
class Dependencies:
    """Immutable container for runtime dependencies injected into pipeline components."""

    llm: dict[str, LLMCallable] | None = None
    extras: Mapping[str, Any] = dataclasses.field(default_factory=dict)
