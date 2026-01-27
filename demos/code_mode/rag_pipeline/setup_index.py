"""RAG Pipeline - One-time Pinecone index setup.

Creates the 'pydantic-kb' index with integrated inference (multilingual-e5-large).
Optionally pre-seeds with a few FAQs.

Usage:
    source .env && uv run python demos/code_mode/rag_pipeline/setup_index.py
"""

from __future__ import annotations

import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

INDEX_NAME = 'pydantic-kb'
NAMESPACE = 'default'

# Pre-seed FAQs about Pydantic AI
SEED_RECORDS = [
    {
        '_id': 'faq-streaming',
        'content': 'To create a streaming agent in Pydantic AI, use the run_stream() method instead of run(). '
        'This returns an async iterator that yields chunks as they arrive. '
        'Example: async for chunk in agent.run_stream("prompt"): print(chunk)',
    },
    {
        '_id': 'faq-output-types',
        'content': 'Pydantic AI supports several output types: TextOutput for plain text, '
        'ToolOutput for structured data via tool calls, NativeOutput for provider-specific structured output, '
        'and PromptedOutput for prompt-based structured extraction.',
    },
    {
        '_id': 'faq-tools',
        'content': 'To add custom tools to an agent, use the @agent.tool decorator on a function. '
        'The function can be sync or async and receives a RunContext as its first argument. '
        'Example: @agent.tool async def my_tool(ctx: RunContext[MyDeps], query: str) -> str: ...',
    },
    {
        '_id': 'faq-run-vs-stream',
        'content': 'The difference between run() and run_stream() is that run() waits for the complete response '
        'before returning, while run_stream() yields chunks incrementally as they arrive from the model. '
        'Use run_stream() for better user experience with long responses.',
    },
    {
        '_id': 'faq-retries',
        'content': 'To handle retries in Pydantic AI, configure the retries parameter when creating an Agent. '
        'You can also use ModelRetry exception to trigger retries with custom messages. '
        'Example: agent = Agent(model, retries=3)',
    },
]


async def setup_index():
    """Create Pinecone index and optionally seed with FAQs."""
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        print('ERROR: PINECONE_API_KEY environment variable is required')
        return

    print('=' * 60)
    print('RAG Pipeline - Pinecone Index Setup')
    print('=' * 60)
    print()

    pinecone = MCPServerStdio(
        'npx',
        args=['-y', '@pinecone-database/mcp'],
        env={'PINECONE_API_KEY': api_key, **os.environ},
    )

    # Use a simple agent to interact with Pinecone MCP
    agent: Agent[None, str] = Agent(
        'gateway/anthropic:claude-sonnet-4-5',
        toolsets=[pinecone],
        system_prompt='You are a setup assistant. Execute the requested operations using the available tools.',
    )

    async with pinecone:
        # Step 1: Check if index exists
        print(f'Checking if index "{INDEX_NAME}" exists...')
        check_result = await agent.run(
            f'List all Pinecone indexes and tell me if "{INDEX_NAME}" exists. Just answer yes or no.'
        )
        print(f'  Result: {check_result.output}')
        print()

        index_exists = 'yes' in check_result.output.lower()

        if not index_exists:
            # Step 2: Create index with integrated inference
            print(f'Creating index "{INDEX_NAME}" with multilingual-e5-large embedding...')
            create_result = await agent.run(
                f'Create a Pinecone index named "{INDEX_NAME}" with integrated inference using the '
                f'multilingual-e5-large embedding model. Set dimension to 1024 and use cosine metric.'
            )
            print(f'  Result: {create_result.output}')
            print()
        else:
            print(f'Index "{INDEX_NAME}" already exists, skipping creation.')
            print()

        # Step 3: Seed with FAQs
        print(f'Upserting {len(SEED_RECORDS)} FAQ records to namespace "{NAMESPACE}"...')
        for record in SEED_RECORDS:
            await agent.run(
                f'Upsert a record to index "{INDEX_NAME}" namespace "{NAMESPACE}" with id "{record["_id"]}" '
                f'and content: {record["content"]}'
            )
            print(f'  - {record["_id"]}: done')

        print()
        print('=' * 60)
        print('Setup complete!')
        print()
        print('You can now run the RAG pipeline demo:')
        print('  source .env && uv run python demos/code_mode/rag_pipeline/web.py')
        print('=' * 60)


if __name__ == '__main__':
    asyncio.run(setup_index())
