# OpenRouter

## Install

To use `OpenRouterModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openrouter` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openrouter]"
```

## Configuration

To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

You can set the `OPENROUTER_API_KEY` environment variable and use [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('openrouter:anthropic/claude-sonnet-4-5')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenRouterModel(
    'anthropic/claude-sonnet-4-5',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
agent = Agent(model)
...
```

## App Attribution

OpenRouter has an [app attribution](https://openrouter.ai/docs/app-attribution) feature to track your application in their public ranking and analytics.

You can pass in an `app_url` and `app_title` when initializing the provider to enable app attribution.

```python
from pydantic_ai.providers.openrouter import OpenRouterProvider

provider=OpenRouterProvider(
    api_key='your-openrouter-api-key',
    app_url='https://your-app.com',
    app_title='Your App',
),
...
```

## Model Settings

You can customize model behavior using [`OpenRouterModelSettings`][pydantic_ai.models.openrouter.OpenRouterModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

settings = OpenRouterModelSettings(
    openrouter_reasoning={
        'effort': 'high',
    },
    openrouter_usage={
        'include': True,
    }
)
model = OpenRouterModel('openai/gpt-5.2')
agent = Agent(model, model_settings=settings)
...
```

## Prompt Caching

OpenRouter supports [prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching) for downstream providers that implement it (currently Anthropic and Gemini). Pydantic AI provides four ways to use prompt caching through OpenRouter:

1. **Cache System Instructions**: Set [`OpenRouterModelSettings.openrouter_cache_instructions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_instructions] to `True` or specify `'5m'` / `'1h'` directly
2. **Cache User Messages**: Set [`OpenRouterModelSettings.openrouter_cache_messages`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_messages] to `True` to automatically cache the last message in the conversation
3. **Cache Tool Definitions**: Set [`OpenRouterModelSettings.openrouter_cache_tool_definitions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_tool_definitions] to `True` or specify `'5m'` / `'1h'` directly
4. **Fine-Grained Control with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker in user messages to cache everything before it

!!! note "Provider Differences"
    - **Anthropic** models support prefix-based caching for both system instructions and user messages. TTL values (`'5m'`, `'1h'`) are passed through to the provider.
    - **Gemini** models only reliably cache system-level content. For Gemini, prefer `openrouter_cache_instructions` over `openrouter_cache_messages`. TTL values are ignored by Gemini.
    - **Minimum token thresholds** apply: Anthropic requires ~2048 tokens, Gemini requires ~1024 tokens for content to be eligible for caching.

### Caching via Model Settings

Use [`OpenRouterModelSettings`][pydantic_ai.models.openrouter.OpenRouterModelSettings] to enable caching for system instructions, user messages, and tool definitions:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

model = OpenRouterModel('anthropic/claude-sonnet-4-6')
agent = Agent(
    model,
    instructions='You are a specialized assistant with deep domain knowledge...',
    model_settings=OpenRouterModelSettings(
        openrouter_cache_instructions=True,  # Cache system instructions (broadly supported)
        openrouter_cache_messages=True,  # Cache the last message (best with Anthropic)
        openrouter_cache_tool_definitions=True,  # Cache tool definitions (Anthropic only)
    ),
)


@agent.tool
def search_docs(ctx: RunContext, query: str) -> str:
    """Search documentation."""
    return f'Results for {query}'
...
```

Each setting accepts `True` (defaults to `'5m'` TTL) or an explicit `'5m'` / `'1h'` TTL value. After running, check `result.usage().cache_write_tokens` and `result.usage().cache_read_tokens` to see caching in action. On subsequent calls with `message_history=result.all_messages()`, cached tokens are reused.

### Fine-Grained Control with CachePoint

Use [`CachePoint`][pydantic_ai.messages.CachePoint] markers to control exactly where cache boundaries are placed:

```python
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel('anthropic/claude-sonnet-4-6')
agent = Agent(model)

prompt = [
    'Long reference document or context to cache...',
    CachePoint(),  # Cache everything before this point
    'Now answer my question about the context above',
]
...
```

Pass the prompt list to `agent.run_sync(prompt)`. Everything before the `CachePoint()` marker is cached. You can place multiple markers for fine-grained control over cache boundaries.

### Provider Routing for Cache Locality

OpenRouter load-balances requests across providers by default. Since cache state is provider-specific, consecutive requests may be routed to different providers, resulting in cache misses. Use [`openrouter_provider`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_provider] to pin requests to a specific downstream provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

model = OpenRouterModel('anthropic/claude-sonnet-4-6')
agent = Agent(
    model,
    model_settings=OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_provider={
            'order': ['anthropic'],
            'allow_fallbacks': False,
        },
    ),
)
...
```

## Web Search

OpenRouter supports web search via its [plugins](https://openrouter.ai/docs/guides/features/plugins/web-search). You can enable it using the [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool].

### Web Search Parameters

You can customize the web search behavior using the `search_context_size` parameter on [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool]:

```python
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models.openrouter import OpenRouterModel

tool = WebSearchTool(search_context_size='high')
model = OpenRouterModel('openai/gpt-4.1')
agent = Agent(
    model,
    builtin_tools=[tool]
)
result = agent.run_sync('What is the latest news in AI?')
```
