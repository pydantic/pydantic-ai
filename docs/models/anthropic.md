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

Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to reduce costs by caching parts of your prompts. Pydantic AI provides three ways to use prompt caching:

1. **Cache User Messages with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker in your user messages to cache everything before it
2. **Cache System Instructions**: Enable the [`AnthropicModelSettings.anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions] [model setting](../agents.md#model-run-settings) to cache your system prompt
3. **Cache Tool Definitions**: Enable the [`AnthropicModelSettings.anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions] [model setting](../agents.md#model-run-settings) to cache your tool definitions

You can combine all three strategies for maximum savings:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions=True,
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
        anthropic_cache_instructions=True
    ),
)

async def main():
    result = await agent.run('Your question')
    usage = result.usage()
    print(f'Cache write tokens: {usage.cache_write_tokens}')
    print(f'Cache read tokens: {usage.cache_read_tokens}')
```

## Structured outputs & strict tool calls

Anthropic offers [Structured Outputs](https://docs.claude.com/en/docs/build-with-claude/structured-outputs), which force the model to emit JSON matching a supplied schema and to validate tool inputs before they reach your code. Pydantic AI enables this automatically:

- When you use [`NativeOutput`][pydantic_ai.output.NativeOutput] (or set `output_mode='native'`), the agent sends Anthropic the JSON schema as an `output_format` payload and adds the required `structured-outputs-2025-11-13` beta header. The model's response will be validated just like on other providers, and `result.output` will contain your typed object.
- Tool definitions that are strict-compatible are sent with `strict: true`, which enables Anthropic's parameter validation. You can opt out for a specific tool by passing `strict=False` to `Tool(...)` or an agent decorator.

```python {test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.anthropic import AnthropicModel


class Contact(BaseModel):
    name: str
    email: str


model = AnthropicModel('claude-sonnet-4-5')
agent = Agent(model, output_type=NativeOutput(Contact))
result = agent.run_sync('Extract the contact info for Ada Lovelace.')
print(result.output)
#> Contact(name='Ada Lovelace', email='ada@example.com')
```

!!! note
    Pydantic AI automatically sets the beta headers, so you do not need to modify `model_settings.extra_headers`. If you also need to send custom `extra_body` fields, avoid supplying your own `output_format` as it will be generated for you.
