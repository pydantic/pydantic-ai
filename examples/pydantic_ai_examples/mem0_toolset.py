"""Example demonstrating Mem0Toolset for memory capabilities.

This example shows how to use Mem0Toolset to add memory capabilities
to your Pydantic AI agents using a simple toolset approach.

Install requirements:
    pip install pydantic-ai mem0ai

Set environment variables:
    export MEM0_API_KEY=your-mem0-api-key
    export OPENAI_API_KEY=your-openai-api-key
"""

# pyright: reportArgumentType=false, reportAssignmentType=false, reportOptionalMemberAccess=false

import asyncio
import os

from pydantic_ai import Agent, Mem0Toolset

# Initialize agent and toolset as None - will be created in main()
agent = None
mem0_toolset = None


def create_agent_with_memory():
    """Create agent with Mem0 toolset. Requires MEM0_API_KEY to be set."""
    # Create Mem0 toolset
    # The toolset provides two tools: _search_memory_impl and _save_memory_impl
    toolset = Mem0Toolset(
        api_key=os.getenv('MEM0_API_KEY'),
        limit=5,  # Return top 5 memories by default
    )

    # Create agent with Mem0 toolset
    # The agent can now automatically use memory tools
    return Agent(
        'openai:gpt-4o',
        toolsets=[toolset],
        instructions=(
            'You are a helpful assistant with memory capabilities. '
            'You can save important information to memory and search through memories. '
            'Use these tools to remember user preferences and provide personalized assistance.'
        ),
    )


async def example_basic_memory():
    """Demonstrate basic memory save and search."""
    print('=== Basic Memory Example ===\n')

    # The agent can save information to memory
    result1 = await agent.run(
        'My name is Alice and I love Python programming. Please remember this.',
        deps='user_alice',
    )
    print(f'Agent: {result1.output}\n')

    # Later, the agent can search and recall memories
    result2 = await agent.run(
        'What do you know about my programming preferences?',
        deps='user_alice',
    )
    print(f'Agent: {result2.output}\n')


async def example_multi_user():
    """Demonstrate memory isolation between users."""
    print('=== Multi-User Memory Isolation ===\n')

    # User Alice
    result_alice = await agent.run(
        'Please remember that my favorite color is blue.',
        deps='user_alice',
    )
    print(f"Alice's Agent: {result_alice.output}\n")

    # User Bob
    result_bob = await agent.run(
        'Please remember that my favorite color is red.',
        deps='user_bob',
    )
    print(f"Bob's Agent: {result_bob.output}\n")

    # Check Alice's memory
    result_alice_recall = await agent.run(
        'What is my favorite color?',
        deps='user_alice',
    )
    print('Alice asks: What is my favorite color?')
    print(f'Agent: {result_alice_recall.output}\n')

    # Check Bob's memory
    result_bob_recall = await agent.run(
        'What is my favorite color?',
        deps='user_bob',
    )
    print('Bob asks: What is my favorite color?')
    print(f'Agent: {result_bob_recall.output}\n')


async def example_with_dataclass():
    """Demonstrate using a dataclass for deps with user_id."""
    from dataclasses import dataclass

    @dataclass
    class UserSession:
        user_id: str
        session_id: str

    print('=== Using Dataclass Deps ===\n')

    session = UserSession(user_id='user_charlie', session_id='session_123')

    result = await agent.run(
        'I work as a data scientist and prefer TypeScript for web development.',
        deps=session,
    )
    print(f'Agent: {result.output}\n')

    result2 = await agent.run(
        'What do you know about my profession and programming preferences?',
        deps=session,
    )
    print(f'Agent: {result2.output}\n')


async def example_personalized_assistant():
    """Demonstrate a personalized assistant with memory."""
    print('=== Personalized Assistant ===\n')

    user_id = 'user_diana'

    # First conversation - learning preferences
    result1 = await agent.run(
        'I prefer concise responses and always want code examples in Python.',
        deps=user_id,
    )
    print('User: I prefer concise responses and always want code examples in Python.')
    print(f'Agent: {result1.output}\n')

    # Later conversation - agent recalls preferences
    result2 = await agent.run(
        'Can you explain how to read a CSV file?',
        deps=user_id,
    )
    print('User: Can you explain how to read a CSV file?')
    print(f'Agent: {result2.output}\n')


async def main():
    """Run all examples."""
    global agent

    if not os.getenv('MEM0_API_KEY'):
        print('Error: MEM0_API_KEY environment variable not set')
        print('Get your API key at: https://app.mem0.ai')
        return

    if not os.getenv('OPENAI_API_KEY'):
        print('Error: OPENAI_API_KEY environment variable not set')
        return

    # Create agent with memory capabilities
    agent = create_agent_with_memory()

    print('🧠 Mem0Toolset Examples for Pydantic AI\n')
    print('=' * 60)
    print()

    await example_basic_memory()

    print('\n' + '=' * 60 + '\n')
    await example_multi_user()

    print('\n' + '=' * 60 + '\n')
    await example_with_dataclass()

    print('\n' + '=' * 60 + '\n')
    await example_personalized_assistant()

    print('\n' + '=' * 60)
    print('\n✅ All examples completed successfully!')


if __name__ == '__main__':
    asyncio.run(main())
