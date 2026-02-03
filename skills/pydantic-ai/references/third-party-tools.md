# Third-Party Tools Reference

Source: `pydantic_ai_slim/pydantic_ai/ext/`

## Overview

PydanticAI integrates with external tool libraries: LangChain and ACI.dev. These tools are not validated by PydanticAI — argument validation is handled by the third-party libraries.

## LangChain Tools

### Single Tool

```python {title="langchain_tool.py" test="skip"}
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import tool_from_langchain

search = DuckDuckGoSearchRun()
search_tool = tool_from_langchain(search)

agent = Agent(
    'google-gla:gemini-2.5-flash',
    tools=[search_tool],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

Requires: `langchain-community` and tool-specific packages (e.g., `ddgs` for DuckDuckGo).

### Multiple Tools / Toolkit

```python {title="langchain_toolkit.py" test="skip"}
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset

toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

## ACI.dev Tools

### Single Tool

```python {title="aci_tool.py" test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import tool_from_aci

tavily_search = tool_from_aci(
    'TAVILY__SEARCH',
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent(
    'google-gla:gemini-2.5-flash',
    tools=[tavily_search],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
```

Requires: `aci-sdk` package and `ACI_API_KEY` environment variable.

### Multiple Tools

```python {title="aci_toolset.py" test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset

toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

## MCP Tools

See [mcp.md](mcp.md) for Model Context Protocol server integration.

## Integration Patterns

### Database Integration

Use dependency injection to share database connections across tools:

```python {test="skip"}
from dataclasses import dataclass

import asyncpg

from pydantic_ai import Agent, RunContext


@dataclass
class DBDeps:
    pool: asyncpg.Pool


agent = Agent('openai:gpt-5', deps_type=DBDeps)


@agent.tool
async def query_users(ctx: RunContext[DBDeps], name_filter: str) -> str:
    """Search for users by name."""
    async with ctx.deps.pool.acquire() as conn:
        rows = await conn.fetch(
            'SELECT id, name, email FROM users WHERE name ILIKE $1 LIMIT 10',
            f'%{name_filter}%',
        )
        return str([dict(r) for r in rows])


@agent.tool
async def get_user_orders(ctx: RunContext[DBDeps], user_id: int) -> str:
    """Get orders for a user."""
    async with ctx.deps.pool.acquire() as conn:
        rows = await conn.fetch(
            'SELECT * FROM orders WHERE user_id = $1 ORDER BY created_at DESC LIMIT 20',
            user_id,
        )
        return str([dict(r) for r in rows])


async def main():
    pool = await asyncpg.create_pool('postgresql://localhost/mydb')
    try:
        result = await agent.run('Find orders for user John', deps=DBDeps(pool=pool))
        print(result.output)
    finally:
        await pool.close()
```

### HTTP Client Integration

Share an `httpx` client for efficient connection pooling:

```python {test="skip"}
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass
class APIDeps:
    client: httpx.AsyncClient
    api_key: str
    base_url: str = 'https://api.example.com'


agent = Agent('openai:gpt-5', deps_type=APIDeps)


@agent.tool
async def fetch_resource(ctx: RunContext[APIDeps], resource_type: str, resource_id: str) -> str:
    """Fetch a resource from the API."""
    url = f'{ctx.deps.base_url}/{resource_type}/{resource_id}'
    response = await ctx.deps.client.get(
        url,
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )

    if response.status_code == 404:
        raise ModelRetry(f'Resource {resource_type}/{resource_id} not found')

    response.raise_for_status()
    return response.text


@agent.tool
async def create_resource(ctx: RunContext[APIDeps], resource_type: str, data: str) -> str:
    """Create a new resource."""
    url = f'{ctx.deps.base_url}/{resource_type}'
    response = await ctx.deps.client.post(
        url,
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
        json={'data': data},
    )
    response.raise_for_status()
    return f'Created: {response.json()}'


async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        deps = APIDeps(client=client, api_key='secret')
        result = await agent.run('Fetch user 123', deps=deps)
        print(result.output)
```

### Queue/Messaging Integration

Tools that publish to message queues:

```python {test="skip"}
from dataclasses import dataclass

import redis.asyncio as redis

from pydantic_ai import Agent, RunContext


@dataclass
class QueueDeps:
    redis_client: redis.Redis
    queue_prefix: str = 'tasks'


agent = Agent('openai:gpt-5', deps_type=QueueDeps)


@agent.tool
async def enqueue_task(ctx: RunContext[QueueDeps], task_type: str, payload: str) -> str:
    """Enqueue a task for background processing."""
    queue_name = f'{ctx.deps.queue_prefix}:{task_type}'
    await ctx.deps.redis_client.lpush(queue_name, payload)
    queue_length = await ctx.deps.redis_client.llen(queue_name)
    return f'Task enqueued. Queue length: {queue_length}'


@agent.tool
async def get_queue_status(ctx: RunContext[QueueDeps], task_type: str) -> str:
    """Get the current queue length for a task type."""
    queue_name = f'{ctx.deps.queue_prefix}:{task_type}'
    length = await ctx.deps.redis_client.llen(queue_name)
    return f'Queue {task_type}: {length} pending tasks'


async def main():
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    try:
        result = await agent.run('Enqueue an email task', deps=QueueDeps(redis_client=client))
        print(result.output)
    finally:
        await client.close()
```

### Authentication Patterns

Securely manage API keys and OAuth tokens:

```python {test="skip"}
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai import Agent, RunContext


@dataclass
class AuthDeps:
    api_key: str  # Static API key
    oauth_token: str | None = None  # Optional OAuth token
    token_expiry: datetime | None = None


agent = Agent('openai:gpt-5', deps_type=AuthDeps)


@agent.tool
async def authenticated_request(ctx: RunContext[AuthDeps], endpoint: str) -> str:
    """Make an authenticated API request."""
    # Choose auth method based on what's available
    if ctx.deps.oauth_token and ctx.deps.token_expiry:
        if datetime.now() < ctx.deps.token_expiry:
            auth_header = f'Bearer {ctx.deps.oauth_token}'
        else:
            # Token expired, fall back to API key
            auth_header = f'ApiKey {ctx.deps.api_key}'
    else:
        auth_header = f'ApiKey {ctx.deps.api_key}'

    # In real code: make the request with auth_header
    return f'Authenticated request to {endpoint}'
```

### Environment-Based Configuration

```python {test="skip"}
import os
from dataclasses import dataclass, field

from pydantic_ai import Agent


@dataclass
class Deps:
    api_key: str = field(default_factory=lambda: os.environ['API_KEY'])
    environment: str = field(default_factory=lambda: os.environ.get('ENV', 'development'))
    base_url: str = field(init=False)

    def __post_init__(self):
        self.base_url = {
            'development': 'http://localhost:8000',
            'staging': 'https://staging.example.com',
            'production': 'https://api.example.com',
        }.get(self.environment, 'http://localhost:8000')


agent = Agent('openai:gpt-5', deps_type=Deps)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `tool_from_langchain` | `pydantic_ai.ext.langchain.tool_from_langchain` | Convert single LangChain tool |
| `LangChainToolset` | `pydantic_ai.ext.langchain.LangChainToolset` | Toolset from LangChain tools |
| `tool_from_aci` | `pydantic_ai.ext.aci.tool_from_aci` | Convert single ACI.dev tool |
| `ACIToolset` | `pydantic_ai.ext.aci.ACIToolset` | Toolset from ACI.dev tools |

## Installation

```bash
# LangChain tools
pip install langchain-community

# ACI.dev tools
pip install aci-sdk
```

## Important Notes

- PydanticAI does **not** validate arguments for third-party tools
- Models provide arguments based on tool schemas
- Third-party tools handle their own error cases
- Check tool-specific documentation for required packages

## See Also

- [tools.md](tools.md) — Native PydanticAI tools
- [toolsets.md](toolsets.md) — Toolset patterns
- [mcp.md](mcp.md) — MCP server integration
- [observability.md](observability.md) — Logfire debugging
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools/) — LangChain tool library
- [ACI.dev Tools](https://www.aci.dev/tools) — ACI.dev tool library
