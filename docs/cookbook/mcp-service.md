---
title: Connect an agent to an MCP service
description: Give an agent tools from a remote MCP server without writing provider-specific tool wrappers.
---

# Connect an agent to an MCP service

Use the `MCP` capability when a service already exposes its operations through Model Context Protocol.

```bash
pip/uv-add "pydantic-ai[mcp]"
```

```python {test="skip"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions='Use the issue tracker tools to answer questions. Never modify an issue unless asked.',
    capabilities=[MCP('https://issues.example.com/mcp')],
)


async def main() -> None:
    result = await agent.run('Summarize the open release-blocking issues.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

Authenticate the MCP transport in application configuration, restrict the server-side tool permissions, and add approval around consequential tools. Treat tool descriptions and returned content from an external server as untrusted input.
