"""Example of Pydantic AI with persistent workspace memory using sayou.

Demonstrates how to give an agent persistent file storage, search, and conversation
history that survives across sessions. The agent can write notes, search past findings,
and pick up conversations where it left off.

Run with:

    uv run -m pydantic_ai_examples.persistent_memory
"""

from __future__ import annotations as _annotations

import asyncio
import os

import logfire

from pydantic_ai import Agent

from sayou.workspace import Workspace
from sayou_pydantic_ai import SayouMessageHistory, SayouToolset

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


async def main():
    # Workspace uses SQLite by default (~/.sayou/sayou.db) — zero config needed
    async with Workspace(slug='pydantic-ai-example') as ws:
        # SayouToolset gives the agent 7 workspace tools:
        # write, read, list, search, grep, glob, kv
        toolset = SayouToolset(workspace=ws)

        agent = Agent(
            os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'),
            toolsets=[toolset],
            instructions=(
                'You have access to a persistent workspace. '
                'Use it to save notes, research findings, and any information '
                'the user asks you to remember.'
            ),
        )

        # --- Session 1: Save some information ---
        print('=== Session 1: Saving notes ===')
        result = await agent.run(
            'Save a meeting note: We decided to use PostgreSQL for the new '
            'service, with Redis for caching. Target launch is Q2.'
        )
        print(f'Agent: {result.output}\n')

        # --- Session 2: Recall information ---
        # In a real app, this would be a separate process/session.
        # The workspace persists, so the agent can find previous notes.
        print('=== Session 2: Recalling information ===')
        result = await agent.run(
            'What decisions have we made? Search the workspace for meeting notes.'
        )
        print(f'Agent: {result.output}\n')

        # --- Conversation history persistence ---
        # SayouMessageHistory stores conversation messages in the workspace KV store
        print('=== Session 3: Persistent conversation ===')
        history = SayouMessageHistory(ws, conversation_id='project-planning')

        # Load any previous messages (empty on first run)
        messages = await history.load()
        print(f'Loaded {len(messages)} previous messages')

        result = await agent.run(
            'Let\'s plan the API endpoints. We need auth, users, and billing.',
            message_history=messages,
        )
        print(f'Agent: {result.output}\n')

        # Save conversation for next time
        await history.save(result.all_messages())
        print(f'Saved {len(result.all_messages())} messages for next session')

        # Continue the conversation (simulating a new session)
        messages = await history.load()
        result = await agent.run(
            'What endpoints did we plan? Add error handling requirements.',
            message_history=messages,
        )
        print(f'Agent: {result.output}\n')
        await history.save(result.all_messages())

        # Show what's in the workspace
        print('=== Workspace contents ===')
        listing = await ws.list('/', recursive=True)
        for f in listing.get('files', []):
            print(f'  {f["path"]}')


if __name__ == '__main__':
    asyncio.run(main())
