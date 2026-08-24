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

You can pass in an `app_url` and `app_title` when initializing the provider to enable app attribution. Both fall back to the `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE` environment variables when omitted.

!!! note
    The environment fallbacks only apply to clients the provider builds itself. If you pass your own
    `openai_client`, it is reused as-is, so set the `HTTP-Referer` and `X-Title` headers on that client
    directly.

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

## Forced tool choice

Pydantic AI treats a forced [`tool_choice`][pydantic_ai.settings.ModelSettings.tool_choice] as incompatible with [thinking](../capabilities/thinking.md) on every `anthropic/` model routed through OpenRouter. Pydantic AI is more conservative than [the direct Anthropic API](anthropic.md#forced-tool-choice), where adaptive thinking accepts forcing — the OpenRouter route hasn't been verified, and it fails quietly rather than loudly: where Anthropic rejects an incompatible combination outright, OpenRouter silently drops the `reasoning` field from the request instead, so the response comes back with no thinking at all. See [#7283](https://github.com/pydantic/pydantic-ai/issues/7283). With thinking enabled on an `anthropic/` model:

- An explicit `tool_choice='required'` (or a list of tool names) raises a [`UserError`][pydantic_ai.exceptions.UserError]; disable thinking or use `tool_choice='auto'`.
- A `required` choice that Pydantic AI resolved on your behalf (e.g. from an [output tool](../output.md#tool-output)) falls back softly to `'auto'`, so thinking is preserved. If the resolved choice named a single tool, the available tool list is filtered to that tool while `tool_choice` remains `'auto'`. The model may therefore answer with text instead of calling it; when an output tool is required, Pydantic AI retries with a prompt to call a tool.

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
    - **OpenAI GPT-5.6** models use OpenAI's `prompt_cache_options` and `prompt_cache_breakpoint` protocol, not `cache_control`. See [OpenAI GPT-5.6 explicit caching](#openai-gpt-56-explicit-caching) below.
    - **Minimum token thresholds** apply; see OpenRouter's [minimum token requirements](https://openrouter.ai/docs/guides/best-practices/prompt-caching#minimum-token-requirements) for current provider-specific values.

### OpenAI GPT-5.6 explicit caching

[`OpenRouterModel`][pydantic_ai.models.openrouter.OpenRouterModel] does not currently translate [`CachePoint`][pydantic_ai.messages.CachePoint] into OpenAI's breakpoint protocol (OpenAI models on OpenRouter still get automatic caching). For explicit GPT-5.6 breakpoints, combine [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] (or [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]) with [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider]:

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

The OpenRouter Responses API uses the same request-wide TTL and usage fields as OpenAI. Restricting the downstream provider to `openai` avoids routing explicit-cache requests to endpoints where these fields are not documented. OpenRouter currently documents explicit breakpoints only on text blocks, so place `CachePoint` markers after text content.

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

OpenRouter supports web search through its [Beta server tool](https://openrouter.ai/docs/guides/features/server-tools/web-search). Enable it with [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool]. The model decides whether to search and may make zero or multiple searches for a request.

Before Pydantic AI v2.30.0, [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] enabled OpenRouter's `web` plugin, which searched on every request and billed a flat fee for each one, whether or not the question needed the web. If you want that always-on grounding, OpenRouter's plugin is deprecated but still reachable by passing it yourself:

```python {title="web_search_openrouter_plugin.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

model = OpenRouterModel('openai/gpt-5.2')
settings = OpenRouterModelSettings(extra_body={'plugins': [{'id': 'web'}]})
agent = Agent(model, model_settings=settings)
result = agent.run_sync('What is the latest news in AI?')
```

### Web Search Parameters

You can configure search context, approximate user location, domain filters, and a limit on searches with [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool]:

```python {title="web_search_openrouter.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.native_tools import WebSearchTool

tool = WebSearchTool(
    search_context_size='high',
    user_location={'city': 'London', 'country': 'GB'},
    allowed_domains=['pydantic.dev'],
    max_uses=1,
)
model = OpenRouterModel('openai/gpt-4.1')
agent = Agent(
    model,
    capabilities=[NativeTool(tool)],
)
result = agent.run_sync('What is the latest news in AI?')
```

Pydantic AI surfaces the per-request web-search count under [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] `['server_tool_use']['web_search_requests']`.

### Search Sources

When OpenRouter runs the search itself rather than delegating to the downstream provider's own search, it attaches the sources it used to the message as `url_citation` annotations. Pydantic AI surfaces them verbatim under [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] `['annotations']`, each carrying the result's `url`, `title` and the excerpt that was given to the model:

```python {title="web_search_openrouter_sources.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.openrouter import OpenRouterModel

agent = Agent(OpenRouterModel('deepseek/deepseek-chat'), capabilities=[WebSearch()])
result = agent.run_sync('What is the latest news in AI?')

annotations = (result.response.provider_details or {}).get('annotations', [])
for annotation in annotations:
    if annotation['type'] == 'url_citation':
        print(annotation['url_citation']['url'])
```

!!! note "Only non-native search reports its sources"
    Models whose downstream provider runs the search natively — OpenAI and Anthropic among them — return no annotations at all, so `provider_details` has no `annotations` entry for those. The normal OpenRouter provider details remain available. Which engine OpenRouter picks is not currently configurable from Pydantic AI.

!!! note "Engine-specific parameters"
    A recorded request verifies only that OpenRouter accepts these parameter names. The per-engine effects below come from OpenRouter's [Beta server-tool documentation](https://openrouter.ai/docs/guides/features/server-tools/web-search), not from responses recorded in this project: native provider search ignores `search_context_size`; `user_location` works only with native search; and domain-filter support varies (native OpenAI ignores `excluded_domains`). The server tool can make zero or several searches when it is available to the model. `max_uses` caps a request when OpenRouter uses a non-native search engine or Anthropic's native search; other native providers, including the OpenAI model in this example, ignore it. OpenRouter does not support [`WebSearchTool.external_web_access`][pydantic_ai.native_tools.WebSearchTool.external_web_access].
