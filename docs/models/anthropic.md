# Anthropic

## Install

To use `AnthropicModel` models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `anthropic` optional group:

```bash
pip/uv-add "pydantic-ai-slim[anthropic]"
```

## Configuration

To use [Anthropic](https://anthropic.com) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.

`AnthropicModelName` contains a list of available Anthropic models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

You can then use `AnthropicModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('anthropic:claude-sonnet-4-5')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-sonnet-4-5')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'claude-sonnet-4-5', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the `AnthropicProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
    'claude-sonnet-4-5',
    provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Prompt Caching

Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to reduce costs by caching parts of your prompts. Pydantic AI provides four ways to use prompt caching:

1. **Cache User Messages with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker in your user messages to cache everything before it
2. **Cache System Instructions**: Set [`AnthropicModelSettings.anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly
3. **Cache Tool Definitions**: Set [`AnthropicModelSettings.anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly
4. **Cache All (Convenience)**: Set [`AnthropicModelSettings.anthropic_cache_all`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_all] to `True` to automatically cache both system instructions and the last user message

You can combine multiple strategies for maximum savings:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

# Option 1: Use anthropic_cache_all for convenience (caches system + last message)
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_all=True,  # Caches both system prompt and last message
    ),
)

# Option 2: Fine-grained control with individual settings
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        # Use True for default 5m TTL, or specify '5m' / '1h' directly
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions='1h',  # Longer cache for tool definitions
    ),
)

@agent.tool
def search_docs(ctx: RunContext, query: str) -> str:
    """Search documentation."""
    return f'Results for {query}'

async def main():
    # First call - writes to cache
    result1 = await agent.run([
        'Long context from documentation...',
        CachePoint(),
        'First question'
    ])

    # Subsequent calls - read from cache (90% cost reduction)
    result2 = await agent.run([
        'Long context from documentation...',  # Same content
        CachePoint(),
        'Second question'
    ])
    print(f'First: {result1.output}')
    print(f'Second: {result2.output}')
```

Access cache usage statistics via `result.usage()`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True  # Default 5m TTL
    ),
)

async def main():
    result = await agent.run('Your question')
    usage = result.usage()
    print(f'Cache write tokens: {usage.cache_write_tokens}')
    print(f'Cache read tokens: {usage.cache_read_tokens}')
```

### Cache Point Limits

Anthropic enforces a maximum of 4 cache points per request. Pydantic AI automatically manages this limit:

- **`anthropic_cache_all`**: Uses 2 cache points (system instructions + last message)
- **`anthropic_cache_instructions`**: Uses 1 cache point
- **`anthropic_cache_tool_definitions`**: Uses 1 cache point
- **`CachePoint` markers**: Use remaining available cache points

When the total exceeds 4 cache points, Pydantic AI automatically removes cache points from **older messages** (keeping the most recent ones), ensuring your requests always comply with Anthropic's limits without errors.

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_all=True,  # Uses 2 cache points
    ),
)

async def main():
    # Even with multiple CachePoint markers, only 2 more will be kept
    # (4 total limit - 2 from cache_all = 2 available)
    result = await agent.run([
        'Context 1', CachePoint(),  # Will be kept
        'Context 2', CachePoint(),  # Will be kept
        'Context 3', CachePoint(),  # Automatically removed (oldest)
        'Question'
    ])
    print(result.output)
```
