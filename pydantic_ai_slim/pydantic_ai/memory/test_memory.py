"""Simple tests for memory system."""

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


def test_retrieved_memory():
    """Test RetrievedMemory creation."""
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


def test_stored_memory():
    """Test StoredMemory creation."""
    memory = StoredMemory(
        id='mem_456',
        memory='User prefers dark mode',
        event='ADD',
        metadata={'importance': 'high'},
    )

    assert memory.id == 'mem_456'
    assert memory.memory == 'User prefers dark mode'
    assert memory.event == 'ADD'
    assert memory.metadata == {'importance': 'high'}


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


def test_memory_config_validation():
    """Test MemoryConfig validation."""
    # Invalid top_k
    with pytest.raises(ValueError, match='top_k must be at least 1'):
        MemoryConfig(top_k=0)

    # Invalid min_relevance_score (too low)
    with pytest.raises(ValueError, match='min_relevance_score must be between 0.0 and 1.0'):
        MemoryConfig(min_relevance_score=-0.1)

    # Invalid min_relevance_score (too high)
    with pytest.raises(ValueError, match='min_relevance_score must be between 0.0 and 1.0'):
        MemoryConfig(min_relevance_score=1.1)

    # Invalid store_after_turns
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

    def __init__(self):
        self.stored_memories: list[tuple] = []
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

    async def retrieve_memories(self, query, **kwargs):
        return self.mock_memories

    async def store_memories(self, messages, **kwargs):
        self.stored_memories.append((messages, kwargs))
        return [
            StoredMemory(
                id='mem_new',
                memory='Stored memory',
                event='ADD',
            )
        ]

    async def get_all_memories(self, **kwargs):
        return self.mock_memories

    async def delete_memory(self, memory_id):
        return True


@pytest.mark.asyncio
async def test_memory_context():
    """Test MemoryContext functionality."""
    provider = MockMemoryProvider()
    context = MemoryContext(provider)

    # Test search
    memories = await context.search('test query', user_id='user_123')

    assert len(memories) == 2
    assert memories[0].memory == 'Test memory 1'
    assert len(context.retrieved_memories) == 2

    # Test add
    stored = await context.add(
        messages=[],
        user_id='user_123',
    )

    assert len(stored) == 1
    assert stored[0].memory == 'Stored memory'
    assert len(context.stored_memories) == 1

    # Test get_all
    all_memories = await context.get_all(user_id='user_123')
    assert len(all_memories) == 2

    # Test delete
    result = await context.delete('mem_1')
    assert result is True


@pytest.mark.asyncio
async def test_memory_provider_protocol():
    """Test that MockMemoryProvider implements MemoryProvider protocol."""
    provider = MockMemoryProvider()

    # Verify it's recognized as a MemoryProvider
    assert isinstance(provider, MemoryProvider)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
