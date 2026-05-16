"""Pydantic AI + emem MCP geospatial agent example.

Connects a Pydantic AI agent to the emem MCP server over Streamable HTTP
and asks a place-based geospatial verification question.

Install:
    pip/uv-add "pydantic-ai[mcp]"

Usage:
    export OPENAI_API_KEY="sk-..."
    python/uv-run -m pydantic_ai_examples.emem_geospatial_agent

The agent will check whether Helsinki Airport, Finland (60.3172, 24.9633)
appears to be low-lying or flood-prone, citing signed receipts.
"""

import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

EMEM_MCP_URL = os.getenv('EMEM_MCP_URL', 'https://emem.dev/mcp')


async def main() -> None:
    toolset = MCPToolset(EMEM_MCP_URL)

    agent = Agent(
        'openai:gpt-5.2',
        toolsets=[toolset],
        system_prompt=(
            'You are a geospatial verification agent. '
            'Use emem tools for place-based evidence. '
            'When emem returns signed facts or receipts, cite them in the answer.'
        ),
    )

    async with agent:
        result = await agent.run(
            'Using emem, check whether Helsinki Airport, Finland '
            '(60.3172, 24.9633) appears to be low-lying or flood-prone. '
            'Use verifiable evidence and cite signed facts or receipts '
            'when available.'
        )

    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
