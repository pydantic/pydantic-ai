"""Inspect Pinecone MCP tool schemas to determine available parameters."""

from __future__ import annotations

import asyncio
import json

from .demo import create_pinecone_mcp


async def main() -> None:
    pinecone = create_pinecone_mcp()
    async with pinecone:
        tools = await pinecone.list_tools()
        for tool in tools:
            if 'rerank' in tool.name.lower():
                print(f'Tool: {tool.name}')
                print(f'Description: {tool.description}')
                print(f'Fields: {list(tool.model_fields.keys())}')
                print(f'Schema:\n{json.dumps(tool.inputSchema, indent=2)}')
                print()


if __name__ == '__main__':
    asyncio.run(main())
