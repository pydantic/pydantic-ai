r"""Persistent memory agent using Dakera — self-hosted, decay-weighted vector memory.

Dakera is a self-hosted REST server that gives pydantic-ai agents persistent memory
across process restarts. Each memory is stored as an embedding; recall uses
decay-weighted scoring so stale context fades naturally.

Start the Dakera server:

    docker run -d \\
        --name dakera \\
        -p 3300:3300 \\
        -e DAKERA_API_KEY=demo \\
        -v dakera_data:/data \\
        ghcr.io/dakera-ai/dakera:latest

Store a fact:

    uv run -m pydantic_ai_examples.dakera_memory_agent store \\
        "My name is Alice and I prefer concise technical answers."

Recall with a semantic query:

    uv run -m pydantic_ai_examples.dakera_memory_agent recall \\
        "What do you know about my preferences?"

Chat with the agent (memory persists across runs):

    uv run -m pydantic_ai_examples.dakera_memory_agent chat \\
        "What are my communication preferences?"
"""

from __future__ import annotations as _annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx

from pydantic_ai import Agent, RunContext

DAKERA_URL = os.getenv('DAKERA_BASE_URL', 'http://localhost:3300')
DAKERA_KEY = os.getenv('DAKERA_API_KEY', 'demo')
DAKERA_NS = os.getenv('DAKERA_NAMESPACE', 'pydantic-ai-agent')


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class MemoryDeps:
    """Runtime dependencies injected into all agent tools."""

    client: httpx.AsyncClient
    namespace: str = DAKERA_NS


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

agent = Agent(
    os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'),
    deps_type=MemoryDeps,
    system_prompt=(
        'You are a helpful assistant with persistent memory. '
        'Before answering questions that may require past context, '
        'use `recall_memory` to search for relevant information. '
        'When the user shares important facts about themselves or their preferences, '
        'use `store_memory` to save those facts for future sessions. '
        'Be concise and accurate.'
    ),
)


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


@agent.tool
async def store_memory(ctx: RunContext[MemoryDeps], content: str) -> str:
    """Store a fact or observation in persistent memory.

    Args:
        ctx: The call context with Dakera client.
        content: The text to store (a fact, preference, or observation).

    Returns:
        Confirmation that the memory was stored.
    """
    resp = await ctx.deps.client.post(
        '/v1/memories',
        json={'content': content, 'namespace': ctx.deps.namespace},
    )
    resp.raise_for_status()
    return f'Stored: {content!r}'


@agent.tool
async def recall_memory(
    ctx: RunContext[MemoryDeps], query: str, limit: int = 5
) -> list[str]:
    """Recall relevant memories by semantic similarity.

    Results are ranked by a composite score: semantic similarity + recency +
    importance (which decays over time without access). More recently accessed
    memories rank higher.

    Args:
        ctx: The call context with Dakera client.
        query: Natural-language query describing what you want to recall.
        limit: Maximum number of memories to return (default 5).

    Returns:
        List of memory contents ranked by relevance.
    """
    resp = await ctx.deps.client.post(
        '/v1/memories/search',
        json={'query': query, 'namespace': ctx.deps.namespace, 'limit': limit},
    )
    resp.raise_for_status()
    data = resp.json()
    memories: list[dict[str, Any]] = data.get('memories', [])
    if not memories:
        return ['No relevant memories found.']
    return [m['content'] for m in memories]


@agent.tool
async def forget_old_memories(
    ctx: RunContext[MemoryDeps], older_than_days: int = 90
) -> str:
    """Forget memories older than N days.

    Use this to prune stale context that is no longer relevant.

    Args:
        ctx: The call context with Dakera client.
        older_than_days: Delete memories older than this many days (default 90).

    Returns:
        Confirmation of how many memories were deleted.
    """
    resp = await ctx.deps.client.request(
        'DELETE',
        '/v1/memories',
        json={'namespace': ctx.deps.namespace, 'older_than_days': older_than_days},
    )
    resp.raise_for_status()
    data = resp.json()
    deleted = data.get('deleted', 'some')
    return f'Forgot {deleted} memories older than {older_than_days} days.'


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


async def _run(prompt: str) -> str:
    async with httpx.AsyncClient(
        base_url=DAKERA_URL,
        headers={'Authorization': f'Bearer {DAKERA_KEY}'},
        timeout=15.0,
    ) as client:
        deps = MemoryDeps(client=client)
        result = await agent.run(prompt, deps=deps)
        return result.output


async def _store_direct(content: str) -> None:
    async with httpx.AsyncClient(
        base_url=DAKERA_URL,
        headers={'Authorization': f'Bearer {DAKERA_KEY}'},
        timeout=15.0,
    ) as client:
        resp = await client.post(
            '/v1/memories',
            json={'content': content, 'namespace': DAKERA_NS},
        )
        resp.raise_for_status()
        print(f'Stored: {content!r}')


async def _recall_direct(query: str) -> None:
    async with httpx.AsyncClient(
        base_url=DAKERA_URL,
        headers={'Authorization': f'Bearer {DAKERA_KEY}'},
        timeout=15.0,
    ) as client:
        resp = await client.post(
            '/v1/memories/search',
            json={'query': query, 'namespace': DAKERA_NS, 'limit': 5},
        )
        resp.raise_for_status()
        memories = resp.json().get('memories', [])
        if not memories:
            print('No memories found.')
        for i, m in enumerate(memories, 1):
            score = m.get('score', 0.0)
            print(f'[{i}] ({score:.2f}) {m["content"]}')


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == 'store':
        content = ' '.join(args)
        if not content:
            print('Usage: store <text>')
            sys.exit(1)
        asyncio.run(_store_direct(content))

    elif command == 'recall':
        query = ' '.join(args)
        if not query:
            print('Usage: recall <query>')
            sys.exit(1)
        asyncio.run(_recall_direct(query))

    elif command == 'chat':
        prompt = ' '.join(args)
        if not prompt:
            print('Usage: chat <prompt>')
            sys.exit(1)
        response = asyncio.run(_run(prompt))
        print(response)

    else:
        print(f'Unknown command: {command!r}')
        print('Commands: store, recall, chat')
        sys.exit(1)


if __name__ == '__main__':
    main()
