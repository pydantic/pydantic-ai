"""Tests for memory system."""

from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai.memory import (
    BaseMemoryProvider,
    MemoryConfig,
    MemoryContext,
    MemoryProvider,
    MemoryScope,
    RetrievalStrategy,
    RetrievedMemory,
    StoredMemory,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart


def test_retrieved_memory_basic():
    """Test RetrievedMemory creation with basic params."""
    memory = RetrievedMemory(
        id='mem_123',
        memory='User likes Python',
    )

    assert memory.id == 'mem_123'
    assert memory.memory == 'User likes Python'
    assert memory.score == 1.0
    assert memory.metadata == {}
    assert memory.created_at is None


def test_retrieved_memory_full():
    """Test RetrievedMemory creation with all params."""
    memory = RetrievedMemory(
        id='mem_123',
        memory='User likes Python',
        score=0.95,
        metadata={'topic': 'preferences'},
        created_at='2024-01-01T00:00:00Z',
    )

    assert memory.id == 'mem_123'
    assert memory.memory == 'User likes Python'
    assert memory.score == 0.95
    assert memory.metadata == {'topic': 'preferences'}
    assert memory.created_at == '2024-01-01T00:00:00Z'


def test_retrieved_memory_repr():
    """Test RetrievedMemory __repr__ method."""
    memory = RetrievedMemory(
        id='mem_123',
        memory='User likes Python',
        score=0.95,
    )
    repr_str = repr(memory)
    assert repr_str == "RetrievedMemory(id='mem_123', memory='User likes Python', score=0.95)"


def test_stored_memory_basic():
    """Test StoredMemory creation with basic params."""
    memory = StoredMemory(
        id='mem_456',
        memory='User prefers dark mode',
    )

    assert memory.id == 'mem_456'
    assert memory.memory == 'User prefers dark mode'
    assert memory.event == 'ADD'
    assert memory.metadata == {}


def test_stored_memory_full():
    """Test StoredMemory creation with all params."""
    memory = StoredMemory(
        id='mem_456',
        memory='User prefers dark mode',
        event='UPDATE',
        metadata={'importance': 'high'},
    )

    assert memory.id == 'mem_456'
    assert memory.memory == 'User prefers dark mode'
    assert memory.event == 'UPDATE'
    assert memory.metadata == {'importance': 'high'}


def test_stored_memory_repr():
    """Test StoredMemory __repr__ method."""
    memory = StoredMemory(
        id='mem_456',
        memory='User prefers dark mode',
        event='ADD',
    )
    repr_str = repr(memory)
    assert repr_str == "StoredMemory(id='mem_456', memory='User prefers dark mode', event='ADD')"


def test_memory_config_defaults():
    """Test MemoryConfig default values."""
    config = MemoryConfig()

    assert config.auto_store is True
    assert config.auto_retrieve is True
    assert config.retrieval_strategy == RetrievalStrategy.SEMANTIC_SEARCH
    assert config.top_k == 5
    assert config.min_relevance_score == 0.0
    assert config.store_after_turns == 1
    assert config.memory_summary_in_system is True
    assert config.scope == MemoryScope.USER
    assert config.metadata == {}


def test_memory_config_custom():
    """Test MemoryConfig with custom values."""
    config = MemoryConfig(
        auto_store=False,
        auto_retrieve=False,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        top_k=10,
        min_relevance_score=0.8,
        store_after_turns=3,
        memory_summary_in_system=False,
        scope=MemoryScope.AGENT,
        metadata={'custom': 'value'},
    )

    assert config.auto_store is False
    assert config.auto_retrieve is False
    assert config.retrieval_strategy == RetrievalStrategy.HYBRID
    assert config.top_k == 10
    assert config.min_relevance_score == 0.8
    assert config.store_after_turns == 3
    assert config.memory_summary_in_system is False
    assert config.scope == MemoryScope.AGENT
    assert config.metadata == {'custom': 'value'}


def test_memory_config_validation_top_k():
    """Test MemoryConfig validation for top_k."""
    with pytest.raises(ValueError, match='top_k must be at least 1'):
        MemoryConfig(top_k=0)


def test_memory_config_validation_relevance_low():
    """Test MemoryConfig validation for min_relevance_score (too low)."""
    with pytest.raises(ValueError, match='min_relevance_score must be between 0.0 and 1.0'):
        MemoryConfig(min_relevance_score=-0.1)


def test_memory_config_validation_relevance_high():
    """Test MemoryConfig validation for min_relevance_score (too high)."""
    with pytest.raises(ValueError, match='min_relevance_score must be between 0.0 and 1.0'):
        MemoryConfig(min_relevance_score=1.1)


def test_memory_config_validation_store_after_turns():
    """Test MemoryConfig validation for store_after_turns."""
    with pytest.raises(ValueError, match='store_after_turns must be at least 1'):
        MemoryConfig(store_after_turns=0)


def test_retrieval_strategy_enum():
    """Test RetrievalStrategy enum values."""
    assert RetrievalStrategy.SEMANTIC_SEARCH == 'semantic_search'
    assert RetrievalStrategy.RECENCY == 'recency'
    assert RetrievalStrategy.HYBRID == 'hybrid'


def test_memory_scope_enum():
    """Test MemoryScope enum values."""
    assert MemoryScope.USER == 'user'
    assert MemoryScope.AGENT == 'agent'
    assert MemoryScope.RUN == 'run'
    assert MemoryScope.GLOBAL == 'global'


class MockMemoryProvider(BaseMemoryProvider):
    """Mock memory provider for testing."""

    def __init__(self) -> None:
        self.stored_memories: list[tuple[list[ModelMessage], dict[str, Any]]] = []
        self.deleted_ids: list[str] = []
        self.mock_memories = [
            RetrievedMemory(
                id='mem_1',
                memory='Test memory 1',
                score=0.9,
            ),
            RetrievedMemory(
                id='mem_2',
                memory='Test memory 2',
                score=0.8,
            ),
        ]

    async def retrieve_memories(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        return self.mock_memories

    async def store_memories(
        self,
        messages: list[ModelMessage],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[StoredMemory]:
        self.stored_memories.append(
            (messages, {'user_id': user_id, 'agent_id': agent_id, 'run_id': run_id, 'metadata': metadata})
        )
        return [
            StoredMemory(
                id='mem_new',
                memory='Stored memory',
                event='ADD',
            )
        ]

    async def get_all_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedMemory]:
        return self.mock_memories

    async def delete_memory(self, memory_id: str) -> bool:
        self.deleted_ids.append(memory_id)
        return True


async def test_memory_context_init():
    """Test MemoryContext initialization."""
    provider = MockMemoryProvider()
    context = MemoryContext(provider)

    assert context.provider is provider
    assert context.retrieved_memories == []
    assert context.stored_memories == []


async def test_memory_context_search():
    """Test MemoryContext search functionality."""
    provider = MockMemoryProvider()
    context = MemoryContext(provider)

    # Test search with minimal params
    memories = await context.search('test query')
    assert len(memories) == 2
    assert memories[0].memory == 'Test memory 1'
    assert len(context.retrieved_memories) == 2

    # Test search with all params
    memories2 = await context.search(
        'another query',
        user_id='user_123',
        agent_id='agent_456',
        run_id='run_789',
        top_k=10,
        metadata={'key': 'value'},
    )
    assert len(memories2) == 2
    assert len(context.retrieved_memories) == 4  # Accumulated


async def test_memory_context_add():
    """Test MemoryContext add functionality."""
    provider = MockMemoryProvider()
    context = MemoryContext(provider)

    # Create test messages
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[TextPart(content='Hi there')]),
    ]

    # Test add with minimal params
    stored = await context.add(messages=messages)
    assert len(stored) == 1
    assert stored[0].memory == 'Stored memory'
    assert len(context.stored_memories) == 1

    # Test add with all params
    stored2 = await context.add(
        messages=messages,
        user_id='user_123',
        agent_id='agent_456',
        run_id='run_789',
        metadata={'importance': 'high'},
    )
    assert len(stored2) == 1
    assert len(context.stored_memories) == 2


async def test_memory_context_get_all():
    """Test MemoryContext get_all functionality."""
    provider = MockMemoryProvider()
    context = MemoryContext(provider)

    # Test get_all with minimal params
    all_memories = await context.get_all()
    assert len(all_memories) == 2

    # Test get_all with all params
    all_memories2 = await context.get_all(
        user_id='user_123',
        agent_id='agent_456',
        run_id='run_789',
        limit=10,
    )
    assert len(all_memories2) == 2


async def test_memory_context_delete():
    """Test MemoryContext delete functionality."""
    provider = MockMemoryProvider()
    context = MemoryContext(provider)

    result = await context.delete('mem_1')
    assert result is True
    assert provider.deleted_ids == ['mem_1']


async def test_memory_provider_protocol():
    """Test that MockMemoryProvider implements MemoryProvider protocol."""
    provider = MockMemoryProvider()

    # Verify it's recognized as a MemoryProvider
    assert isinstance(provider, MemoryProvider)

