"""Memory schema definitions for structured memory records."""

from datetime import UTC, datetime
from typing import Literal
import uuid

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single structured memory record."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    key: str = Field(description="Hierarchical key, e.g. 'user.identity.employer'")
    value: str = Field(description="The memory value")
    memory_type: Literal["identity", "semi_stable", "dynamic", "ephemeral"] = "semi_stable"
    stability_score: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence_source: Literal[
        "explicit_statement", "direct_answer", "contextual_mention", "inferred", "offhand"
    ] = "contextual_mention"
    dimensions: dict[str, str] = Field(default_factory=dict)
    scope: str = "private"
    scope_ref: str | None = None
    source: str | None = None
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    conflicts_with: str | None = Field(default=None, description="ID of conflicting memory")
    archived: bool = False
    raw_text: str | None = Field(default=None, description="Original text this was extracted from")

    def touch(self) -> None:
        """Update last_accessed and increment access_count."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1


class MemorySearchResult(BaseModel):
    """A memory with a relevance score from retrieval."""

    memory: Memory
    relevance_score: float = Field(ge=0.0, le=1.0)
    match_type: Literal["semantic", "keyword", "key_match", "dimensional"] = "semantic"


class ExtractionResult(BaseModel):
    """Output from the memory extraction model."""

    memories: list[Memory]
    source: str | None = None


STABILITY_DEFAULTS: dict[str, float] = {
    "identity": 1.0,
    "semi_stable": 0.7,
    "dynamic": 0.3,
    "ephemeral": 0.0,
}

CONFIDENCE_SCORES: dict[str, float] = {
    "explicit_statement": 1.0,
    "direct_answer": 0.9,
    "contextual_mention": 0.7,
    "inferred": 0.5,
    "offhand": 0.3,
}
