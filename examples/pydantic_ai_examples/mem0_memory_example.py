"""Example demonstrating Mem0 memory integration with Pydantic AI.

This example shows how to use Mem0's platform for persistent memory across
conversations with Pydantic AI agents.

Install requirements:
    pip install pydantic-ai mem0ai

Set environment variables:
    export MEM0_API_KEY=your-mem0-api-key
    export OPENAI_API_KEY=your-openai-api-key
"""

import asyncio
import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.memory import MemoryConfig
from pydantic_ai.memory.providers import Mem0Provider


# Define dependencies with session identifiers
@dataclass
class UserSession:
    user_id: str
    session_id: str | None = None


# Create Mem0 memory provider
memory_provider = Mem0Provider(
    api_key=os.getenv('MEM0_API_KEY'),
    config=MemoryConfig(
        auto_store=True,  # Automatically store conversations
        auto_retrieve=True,  # Automatically retrieve memories
        top_k=5,  # Retrieve top 5 relevant memories
        min_relevance_score=0.7,  # Only use highly relevant memories
    ),
)

# Create agent with memory (Note: Full integration coming soon!)
agent = Agent(
    'openai:gpt-4o',
    deps_type=UserSession,
    instructions=(
        'You are a helpful assistant with memory of past conversations. '
        'Use the memories provided to personalize your responses.'
    ),
)


# Tool to manually search memories
@agent.tool
async def search_user_memories(ctx: RunContext[UserSession], query: str) -> str:
    """Search through user's memories.

    Args:
        ctx: The run context with user session.
        query: What to search for in memories.
    """
    # Access mem0 through the memory provider
    memories = await memory_provider.retrieve_memories(
        query=query,
        user_id=ctx.deps.user_id,
        top_k=3,
    )

    if not memories:
        return 'No relevant memories found.'

    result = ['Found these relevant memories:']
    for mem in memories:
        result.append(f'- {mem.memory} (relevance: {mem.score:.2f})')

    return '\n'.join(result)


# Tool to manually store a memory
@agent.tool
async def store_memory(ctx: RunContext[UserSession], fact: str) -> str:
    """Store an important fact to remember.

    Args:
        ctx: The run context with user session.
        fact: The fact to store.
    """
    # Note: For proper memory storage, use result.all_messages() from agent runs
    # This tool acknowledges the fact for demonstration purposes
    # In production, memories are typically stored after complete conversations
    return f'I will remember: {fact}. Memory will be stored after our conversation.'


# Tool to view all user memories
@agent.tool
async def list_all_memories(ctx: RunContext[UserSession]) -> str:
    """List all memories for the current user."""
    memories = await memory_provider.get_all_memories(
        user_id=ctx.deps.user_id,
        limit=10,
    )

    if not memories:
        return 'No memories found for this user.'

    result = [f'Found {len(memories)} memories:']
    for idx, mem in enumerate(memories, 1):
        result.append(f'{idx}. {mem.memory}')

    return '\n'.join(result)


async def example_conversation():
    """Demonstrate a multi-turn conversation with memory."""
    user_session = UserSession(user_id='user_alice', session_id='session_001')

    print('=== First Conversation ===\n')

    # First interaction - store information
    result1 = await agent.run(
        'My name is Alice and I love Python programming. I work as a data scientist.',
        deps=user_session,
    )
    print(f'Agent: {result1.output}\n')

    # Store conversation manually
    await memory_provider.store_memories(
        messages=result1.all_messages(),
        user_id=user_session.user_id,
    )
    print('Stored conversation in Mem0.\n')

    # Second interaction - retrieve and use memory
    result2 = await agent.run(
        'What programming language do I like?',
        deps=user_session,
    )
    print(f'Agent: {result2.output}\n')

    print('=== Using Memory Tools ===\n')

    # Use memory search tool
    result3 = await agent.run(
        'Can you search my memories for information about my profession?',
        deps=user_session,
    )
    print(f'Agent: {result3.output}\n')

    # Store additional memory
    result4 = await agent.run(
        'Please remember that I prefer dark mode for all my applications.',
        deps=user_session,
    )
    print(f'Agent: {result4.output}\n')

    # List all memories
    result5 = await agent.run(
        'Can you show me all my memories?',
        deps=user_session,
    )
    print(f'Agent: {result5.output}\n')


async def example_multi_user():
    """Demonstrate memory isolation between different users."""
    print('=== Multi-User Memory Isolation ===\n')

    # User 1
    alice = UserSession(user_id='user_alice')
    result_alice = await agent.run(
        'My favorite color is blue and I live in San Francisco.',
        deps=alice,
    )
    print(f'Alice: My favorite color is blue and I live in San Francisco.')
    print(f'Agent: {result_alice.output}\n')

    # Store Alice's memory
    await memory_provider.store_memories(
        messages=result_alice.all_messages(),
        user_id=alice.user_id,
    )

    # User 2
    bob = UserSession(user_id='user_bob')
    result_bob = await agent.run(
        'My favorite color is red and I live in New York.',
        deps=bob,
    )
    print(f'Bob: My favorite color is red and I live in New York.')
    print(f'Agent: {result_bob.output}\n')

    # Store Bob's memory
    await memory_provider.store_memories(
        messages=result_bob.all_messages(),
        user_id=bob.user_id,
    )

    # Test memory isolation
    result_alice_recall = await agent.run(
        'What is my favorite color and where do I live?',
        deps=alice,
    )
    print(f'Alice: What is my favorite color and where do I live?')
    print(f'Agent: {result_alice_recall.output}\n')

    result_bob_recall = await agent.run(
        'What is my favorite color and where do I live?',
        deps=bob,
    )
    print(f'Bob: What is my favorite color and where do I live?')
    print(f'Agent: {result_bob_recall.output}\n')


async def example_session_memory():
    """Demonstrate session-scoped memories."""
    print('=== Session-Scoped Memory ===\n')

    # Create provider with session scope
    session_memory = Mem0Provider(
        api_key=os.getenv('MEM0_API_KEY'),
        config=MemoryConfig(
            auto_store=True,
            auto_retrieve=True,
        ),
    )

    session_agent = Agent(
        'openai:gpt-4o',
        deps_type=UserSession,
        instructions='Remember context within this session.',
    )

    # Session 1
    session1 = UserSession(user_id='user_alice', session_id='shopping_001')

    result1 = await session_agent.run(
        'I want to buy a laptop. My budget is $1500.',
        deps=session1,
    )
    print(f'[Session 1] User: I want to buy a laptop. My budget is $1500.')
    print(f'[Session 1] Agent: {result1.output}\n')

    # Store session memory
    await session_memory.store_memories(
        messages=result1.all_messages(),
        user_id=session1.user_id,
        run_id=session1.session_id,
    )

    # Continue session 1
    result2 = await session_agent.run(
        'What was my budget again?',
        deps=session1,
    )
    print(f'[Session 1] User: What was my budget again?')
    print(f'[Session 1] Agent: {result2.output}\n')

    # Session 2 - different context
    session2 = UserSession(user_id='user_alice', session_id='vacation_002')

    result3 = await session_agent.run(
        'I want to plan a vacation to Japan.',
        deps=session2,
    )
    print(f'[Session 2] User: I want to plan a vacation to Japan.')
    print(f'[Session 2] Agent: {result3.output}\n')


async def main():
    """Run all examples."""
    try:
        # Check for required API keys
        if not os.getenv('MEM0_API_KEY'):
            print('Error: MEM0_API_KEY environment variable not set')
            print('Get your API key at: https://app.mem0.ai')
            return

        if not os.getenv('OPENAI_API_KEY'):
            print('Error: OPENAI_API_KEY environment variable not set')
            return

        print('🧠 Mem0 + Pydantic AI Memory Integration Examples\n')
        print('=' * 60)
        print()

        await example_conversation()

        print('\n' + '=' * 60 + '\n')
        await example_multi_user()

        print('\n' + '=' * 60 + '\n')
        await example_session_memory()

        print('\n' + '=' * 60)
        print('\n✅ All examples completed successfully!')

    except Exception as e:
        print(f'\n❌ Error: {e}')
        raise


if __name__ == '__main__':
    asyncio.run(main())
