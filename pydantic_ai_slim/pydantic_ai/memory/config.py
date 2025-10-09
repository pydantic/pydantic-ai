"""Configuration classes for memory system."""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = (
    'MemoryConfig',
    'RetrievalStrategy',
    'MemoryScope',
)


class RetrievalStrategy(str, Enum):
    """Strategy for retrieving memories."""

    SEMANTIC_SEARCH = 'semantic_search'
    """Use semantic similarity search to find relevant memories."""

    RECENCY = 'recency'
    """Retrieve most recent memories."""

    HYBRID = 'hybrid'
    """Combine semantic search with recency."""


class MemoryScope(str, Enum):
    """Scope for memory storage and retrieval."""

    USER = 'user'
    """Memories scoped to a specific user."""

    AGENT = 'agent'
    """Memories scoped to a specific agent."""

    RUN = 'run'
    """Memories scoped to a specific run/session."""

    GLOBAL = 'global'
    """Global memories not scoped to any identifier."""


@dataclass
class MemoryConfig:
    """Configuration for memory behavior in agents.

    Attributes:
        auto_store: Automatically store conversations as memories after each run.
        auto_retrieve: Automatically retrieve relevant memories before each model request.
        retrieval_strategy: Strategy to use for retrieving memories.
        top_k: Maximum number of memories to retrieve.
        min_relevance_score: Minimum relevance score (0.0-1.0) for retrieved memories.
        store_after_turns: Store memories after this many conversation turns.
        memory_summary_in_system: Include memory summary in system prompt.
        scope: Default scope for memory operations.
        metadata: Additional metadata to include with all memory operations.
    """

    auto_store: bool = True
    auto_retrieve: bool = True
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC_SEARCH
    top_k: int = 5
    min_relevance_score: float = 0.0
    store_after_turns: int = 1
    memory_summary_in_system: bool = True
    scope: MemoryScope = MemoryScope.USER
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.top_k < 1:
            raise ValueError('top_k must be at least 1')
        if not 0.0 <= self.min_relevance_score <= 1.0:
            raise ValueError('min_relevance_score must be between 0.0 and 1.0')
        if self.store_after_turns < 1:
            raise ValueError('store_after_turns must be at least 1')
