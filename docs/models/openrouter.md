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

agent = Agent('openrouter:anthropic/claude-sonnet-4.6')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenRouterModel(
    'anthropic/claude-sonnet-4.6',
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

### Eager Input Streaming

For Anthropic models via OpenRouter, you can enable eager input streaming to reduce latency for tool calls with large inputs.
Set [`anthropic_eager_input_streaming`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_eager_input_streaming] in [`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel('anthropic/claude-sonnet-4-5')
settings = AnthropicModelSettings(anthropic_eager_input_streaming=True)
agent = Agent(model, model_settings=settings)
...
```

## Prompt Caching

OpenRouter supports [prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching) for downstream providers that implement it. Pydantic AI's OpenRouter cache settings control explicit `cache_control` breakpoints for Anthropic and Gemini models:

1. **Cache System Instructions**: Set [`OpenRouterModelSettings.openrouter_cache_instructions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_instructions] to `True` or specify `'5m'` / `'1h'` directly
2. **Cache the Last Message**: Set [`OpenRouterModelSettings.openrouter_cache_messages`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_messages] to `True` to automatically cache the last message in the conversation
3. **Cache Tool Definitions**: Set [`OpenRouterModelSettings.openrouter_cache_tool_definitions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_tool_definitions] to `True` or specify `'5m'` / `'1h'` directly
4. **Fine-Grained Control with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker in user messages to cache everything before it

!!! note "Provider Differences"
    - **Anthropic** models support prefix-based caching for both system instructions and message content. TTL values (`'5m'`, `'1h'`) are passed through to the provider.
    - **Gemini** models support caching for system instructions and normal message content, but [OpenRouter uses only the last breakpoint across normal message content for Gemini caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching#how-gemini-prompt-caching-works-on-openrouter).
      Use `openrouter_cache_messages` or [`CachePoint`][pydantic_ai.messages.CachePoint] when that final message boundary is intentional; use `openrouter_cache_instructions` only for fully static system context. TTL values are ignored by Gemini.
      Cached Gemini `systemInstruction` content is immutable, so put dynamic prompt segments in a later user message instead of after cached system instructions.
    - **OpenAI GPT-5.6** models use OpenAI's `prompt_cache_options` and `prompt_cache_breakpoint` protocol, not `cache_control`. OpenRouter exposes this protocol only through the Responses API.
    - **Minimum token thresholds** apply; see OpenRouter's [minimum token requirements](https://openrouter.ai/docs/guides/best-practices/prompt-caching#minimum-token-requirements) for current provider-specific values.

### OpenAI GPT-5.6 explicit caching

[`OpenRouterModel`][pydantic_ai.models.openrouter.OpenRouterModel] uses the Chat Completions API, where OpenRouter currently supports only automatic caching for OpenAI models. For explicit GPT-5.6 breakpoints, combine [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] with [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider]:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenAIResponsesModel(
    'openai/gpt-5.6-sol',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
settings = OpenAIResponsesModelSettings(
    openai_prompt_cache_key='product-docs-v1',
    openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
    # OpenRouter also offers Azure routes for GPT-5.6, where explicit caching is not documented.
    extra_body={'provider': {'only': ['openai']}},
)
agent = Agent(model, model_settings=settings)

result = agent.run_sync([
    'Long-lived reference material...',
    CachePoint(),
    'Answer using the reference material.',
])
```

The OpenRouter Responses API uses the same request-wide TTL and usage fields as OpenAI. Restricting the downstream provider to `openai` avoids routing explicit-cache requests to endpoints where these fields are not documented. OpenRouter currently documents explicit breakpoints only on `input_text` blocks, so a `CachePoint` after an image or file raises [`UserError`][pydantic_ai.exceptions.UserError] instead of sending an unsupported request shape.

### Caching via Model Settings

Use [`OpenRouterModelSettings`][pydantic_ai.models.openrouter.OpenRouterModelSettings] to enable explicit caching for system instructions, the last conversation message, and tool definitions:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

model = OpenRouterModel('anthropic/claude-sonnet-4.6')
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

Each setting accepts `True` or an explicit `'5m'` / `'1h'` TTL value. `True` sends Anthropic's default `'5m'` TTL for Anthropic models; Gemini ignores TTL values and manages cache lifetime itself. Check `result.usage.cache_write_tokens` on initial writes and `result.usage.cache_read_tokens` on reuse, including subsequent calls with `message_history=result.all_messages()`.

OpenRouter uses [provider sticky routing](https://openrouter.ai/docs/guides/best-practices/prompt-caching#provider-sticky-routing) after prompt-cached requests to improve cache locality. For cache-sensitive workflows that need stricter provider control or disabled fallbacks, also set [`openrouter_provider`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_provider], for example with `{'order': ['anthropic'], 'allow_fallbacks': False}`.

### Fine-Grained Control with CachePoint

Use [`CachePoint`][pydantic_ai.messages.CachePoint] markers to control exactly where cache boundaries are placed:

```python
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel('anthropic/claude-sonnet-4.6')
agent = Agent(model)

prompt = [
    'Long reference document or context to cache...',
    CachePoint(),  # Cache everything before this point
    'Now answer my question about the context above',
]
...
```

Pass the prompt list to `agent.run_sync(prompt)`. Everything before the `CachePoint()` marker is cached. You can place multiple markers for fine-grained control over cache boundaries.

!!! warning "Anthropic cache-breakpoint ordering"
    Anthropic processes cache breakpoints in a fixed order — tool definitions, then system instructions, then messages — and rejects a `'1h'` breakpoint that appears *after* a `'5m'` one in that sequence. When mixing TTLs across `CachePoint` markers or the cache settings on an Anthropic model, place the longer (`'1h'`) breakpoints before the shorter (`'5m'`) ones. Anthropic also allows at most four explicit breakpoints per request; excess breakpoints are dropped (oldest first) before the request is sent.

## Web Search

OpenRouter supports web search via its [plugins](https://openrouter.ai/docs/guides/features/plugins/web-search). You can enable it using the [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool].

### Web Search Parameters

You can customize the web search behavior using the `search_context_size` parameter on [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool]:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.native_tools import WebSearchTool

tool = WebSearchTool(search_context_size='high')
model = OpenRouterModel('openai/gpt-4.1')
agent = Agent(
    model,
    capabilities=[NativeTool(tool)],
)
result = agent.run_sync('What is the latest news in AI?')
```
