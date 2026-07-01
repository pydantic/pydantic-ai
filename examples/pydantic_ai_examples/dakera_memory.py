"""Dakera memory integration example for Pydantic AI.

Demonstrates persistent cross-session memory for an AI assistant using Dakera
as the memory backend. The agent automatically recalls relevant memories before
each response and stores new information after each run.

Prerequisites
-------------
1. Start a local Dakera server:

    docker run -p 3300:3300 -e DAKERA_API_KEY=demo \\
        ghcr.io/dakera-ai/dakera:latest

2. Install dependencies:

    pip install pydantic-ai httpx
    pip install pydantic-ai-dakera  # or: pip install -e ./pydantic_ai_dakera

Run
---
    python -m pydantic_ai_examples.dakera_memory

Session 1 — store a preference:

    python -m pydantic_ai_examples.dakera_memory store "I am a Python developer"

Session 2 — recall cross-session:

    python -m pydantic_ai_examples.dakera_memory ask "What do you know about me?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic_ai import Agent

# Install: pip install pydantic-ai-dakera
# or: sys.path.insert(0, '<path-to-pydantic-ai-src>')
try:
    from pydantic_ai_dakera import DakeraMemory
except ImportError as e:
    raise ImportError(
        'Install pydantic-ai-dakera: pip install pydantic-ai-dakera\n'
        'Or start Dakera: docker run -p 3300:3300 ghcr.io/dakera-ai/dakera:latest'
    ) from e


# ---------------------------------------------------------------------------
# Build the agent with Dakera persistent memory
# ---------------------------------------------------------------------------

DAKERA_URL = os.environ.get('DAKERA_URL', 'http://localhost:3300')
DAKERA_KEY = os.environ.get('DAKERA_API_KEY', 'demo')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'openai:gpt-4o-mini')

memory = DakeraMemory(
    base_url=DAKERA_URL,
    api_key=DAKERA_KEY,
    agent_id='personal-assistant',   # Namespace all memories for this agent role
    top_k=5,                          # Retrieve the 5 most relevant memories
    store_importance=0.75,            # Auto-stored memories get this importance
    inject_instruction_prefix='Context from previous conversations:',
)

agent = Agent(
    OPENAI_MODEL,
    instructions=(
        'You are a helpful personal assistant with persistent memory. '
        'When you receive context from previous conversations at the top of '
        'your instructions, use it to personalise your responses. '
        'You can explicitly remember things using the dakera_remember tool, '
        'and remove outdated information with dakera_forget.'
    ),
    capabilities=[memory],
)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------


async def demo_store(fact: str) -> None:
    """Store a fact and verify it was remembered."""
    print(f'\n--- Storing: {fact!r} ---')
    result = await agent.run(f'Please remember this about me: {fact}')
    print(f'Agent: {result.output}')


async def demo_ask(question: str) -> None:
    """Ask the agent something that should be answerable from memory."""
    print(f'\n--- Asking: {question!r} ---')
    result = await agent.run(question)
    print(f'Agent: {result.output}')


async def demo_multi_turn() -> None:
    """Multi-turn conversation demonstrating cross-run memory persistence."""
    print('\n=== Multi-turn demo ===')

    # Turn 1 — establish a preference
    r1 = await agent.run(
        'My name is Alex and I prefer concise technical answers. '
        'I am a senior Python engineer.'
    )
    print(f'Turn 1 → {r1.output}')

    # Turn 2 — new run, should recall the name and preference
    r2 = await agent.run('What is my name and how should you answer my questions?')
    print(f'Turn 2 → {r2.output}')

    # Turn 3 — domain knowledge stored via tool
    r3 = await agent.run(
        'I just switched to Rust for performance-critical work. '
        'Please remember this language preference.',
        message_history=r2.all_messages(),  # Continue the same conversation
    )
    print(f'Turn 3 → {r3.output}')

    print('\n--- After all turns, memories stored in Dakera are available in future sessions. ---')


async def demo_search_and_forget() -> None:
    """Demonstrate explicit memory management tools."""
    print('\n=== Explicit memory management ===')

    # Store something via the agent tool
    r1 = await agent.run(
        "Remember with dakera_remember: My preferred IDE is Neovim with LSP."
    )
    print(f'Store result: {r1.output}')

    # Ask to forget it
    r2 = await agent.run(
        "I've switched to VS Code now. Use dakera_forget to remove the Neovim preference, "
        "then use dakera_remember to save: My preferred IDE is VS Code with Pylance."
    )
    print(f'Update result: {r2.output}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(command: str = 'demo', arg: str = '') -> None:
    if command == 'store':
        await demo_store(arg or 'I am a Python developer who loves clean code.')
    elif command == 'ask':
        await demo_ask(arg or 'What programming language do I work with?')
    elif command == 'multi-turn':
        await demo_multi_turn()
    elif command == 'memory-tools':
        await demo_search_and_forget()
    else:
        # Default: run all demos in sequence
        print('Dakera Memory Integration Demo')
        print('=' * 40)
        print(f'Dakera URL: {DAKERA_URL}')
        print(f'Model: {OPENAI_MODEL}')
        print()
        await demo_store('I am a Python developer who values type safety.')
        await demo_ask('What programming language do I use?')
        await demo_multi_turn()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    extra = sys.argv[2] if len(sys.argv) > 2 else ''
    asyncio.run(main(cmd, extra))
