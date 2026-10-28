---
description: "Use Anthropic Claude models with Pydantic AI, with prompt caching, task budgets and compaction, directly or via Bedrock, Vertex AI or Microsoft Foundry."
---

# Anthropic

## Install

To use `AnthropicModel`, install either `pydantic-ai` or `pydantic-ai-slim` with the `anthropic` optional group:

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

agent = Agent('{{ anthropic:medium }}')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('{{ anthropic:medium.name }}')
agent = Agent(model)
...
```

!!! note "Claude Opus 4.7 / 4.8 / 5 / 5.5 migration"
    Anthropic's [Claude Opus migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide) recommends removing `temperature`, `top_p`, and `top_k` from Opus 4.7, 4.8, 5, and 5.5 requests. Pydantic AI drops those keys automatically for `claude-opus-4-7`, `claude-opus-4-8`, `claude-opus-5`, `claude-opus-5-5`, `claude-sonnet-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-fable-5` and `claude-mythos-5`, including `extra_body` overrides.

    The same guide also recommends re-evaluating `max_tokens` and any token-count assumptions when migrating from Opus 4.6, since Opus 4.7 introduced updated tokenization (carried into 4.8). If you rely on `count_tokens()` or `count_tokens_before_request`, verify your thresholds against the new model.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    '{{ anthropic:medium.name }}', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the `AnthropicProvider` with a custom `httpx2.AsyncClient`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
    '{{ anthropic:medium.name }}',
    provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

A legacy `httpx.AsyncClient` is not accepted: `anthropic` 1.0 is built on `httpx2` and rejects one at
client construction.

The `AsyncAnthropic` client that the provider builds also retries failed requests on its own — `max_retries=2` by default, so a request can reach the network up to three times before your code sees an error. Pass `max_retries=0` when you construct the client yourself (for example as `anthropic_client=`) to keep the retry policy in your transport alone, or set [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) to change the count without building the client, including for an `AsyncAnthropicBedrock` or `AsyncAnthropicVertex` client. See [Provider SDK retries](../retries.md#provider-sdk-retries) for when this layer fires.

## Model settings

You can customize model behavior using [`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('{{ anthropic:medium.name }}')
settings = AnthropicModelSettings(
    anthropic_effort='low',
    service_tier='auto',
)
agent = Agent(model, model_settings=settings)
...
```

Anthropic requires [`max_tokens`][pydantic_ai.settings.ModelSettings.max_tokens], which thinking counts toward. When you don't set it, Pydantic AI sends the model's maximum output, like 64,000 on Claude Sonnet 4.5 or 128,000 on Claude Opus 5 and Claude Haiku 5.5, and streams the request behind the scenes, since a response that long can take more than 10 minutes. A [`timeout`][pydantic_ai.settings.ModelSettings.timeout] then limits the wait between streamed chunks rather than the whole response. A model whose maximum isn't known gets 16384, and models older than Claude Sonnet 4.5 get 4096, since they reject a request whose input plus `max_tokens` exceeds the context window. Models are recognized by name, so set `max_tokens` yourself if you reach one through a Bedrock ARN or a custom deployment name.

### Service tier

Anthropic supports controlling the [service tier](https://platform.claude.com/docs/en/api/service-tiers) to manage latency and throughput.
You can use the unified [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] field or the provider-specific [`anthropic_service_tier`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_service_tier] field. `anthropic_service_tier` takes precedence over the unified field when both are set, and accepts Anthropic's native values (`'auto'` or `'standard_only'`).

The unified field maps as follows for Anthropic:

- `'auto'`: passed through as `'auto'` (Anthropic's native value — uses priority capacity when available).
- `'default'`: maps to `'standard_only'` (forces the standard tier, opting out of priority capacity).
- `'flex'` and `'priority'` are not part of Anthropic's tier model and are silently ignored.

## Cloud Platform Integrations

You can use Anthropic models through cloud platforms by passing a custom client to [`AnthropicProvider`][pydantic_ai.providers.anthropic.AnthropicProvider].

### AWS Bedrock

To use Claude models via [AWS Bedrock](https://aws.amazon.com/bedrock/claude/), follow the [Anthropic documentation](https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock) on how to set up a Bedrock client and then pass it to `AnthropicProvider`. Both the newer `AsyncAnthropicBedrockMantle` client (recommended by Anthropic, using the Messages API) and the legacy `AsyncAnthropicBedrock` client (using the `InvokeModel` API with ARN-versioned model IDs) are supported:

```python {test="skip" typecheck="skip - anthropic's __all__ omits this client, so pyright reports it as not exported"}
from anthropic import AsyncAnthropicBedrockMantle

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

bedrock_client = AsyncAnthropicBedrockMantle()  # Uses AWS credentials from environment
provider = AnthropicProvider(anthropic_client=bedrock_client)
model = AnthropicModel('anthropic.claude-haiku-4-5', provider=provider)
agent = Agent(model)
...
```

!!! note "Bedrock vs BedrockConverseModel"
    This approach uses Anthropic's SDK with AWS Bedrock credentials. For an alternative using AWS SDK (boto3) directly, see [`BedrockConverseModel`](bedrock.md).

!!! note "Tool search on the legacy `AsyncAnthropicBedrock` client"
    On the `AsyncAnthropicBedrock` client, [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] with no `strategy` uses the `'regex'` [tool search](../tools-advanced.md#tool-search) variant instead of `'bm25'`. Pass `ToolSearch(strategy='bm25')` to use BM25, which the legacy `InvokeModel` API also accepts. Claude Sonnet 4.5 needs the `tool-search-tool-2025-10-19` beta for either variant on this client: add it with `anthropic_betas=['tool-search-tool-2025-10-19']` in [`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings]. A `bm25` search recorded in message history replays as `regex` on this client unless the run uses `ToolSearch(strategy='bm25')`.

!!! note "Token counting on the legacy `AsyncAnthropicBedrock` client"
    The Anthropic SDK blocks its high-level token-counting method on Bedrock, so `count_tokens()` (and `count_tokens_before_request`) instead call Bedrock's own `/model/{model}/count-tokens` endpoint. That endpoint only accepts **base** foundation-model IDs (e.g. `anthropic.claude-sonnet-4-20250514-v1:0`); cross-region inference profile IDs (`us.`/`eu.`/`global.` prefixes) and end-of-life model versions are rejected by Bedrock.

!!! note "Structured output on the legacy `AsyncAnthropicBedrock` client"
    Bedrock withholds structured output from some Claude models that have it on the direct Anthropic API, such as Claude Fable 5. On the `AsyncAnthropicBedrock` client, those models don't send `strict` tool definitions, a bare structured `output_type` doesn't use [Native Output](../output.md#native-output), and an explicit [`NativeOutput`][pydantic_ai.output.NativeOutput] raises a [`UserError`][pydantic_ai.exceptions.UserError], matching [`BedrockConverseModel`](bedrock.md#thinking-and-structured-output).

### Google Cloud

To use Claude models via [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude), follow the [Anthropic documentation](https://docs.anthropic.com/en/api/claude-on-vertex-ai) on how to set up an `AsyncAnthropicVertex` client and then pass it to `AnthropicProvider`:

```python {test="skip" typecheck="skip - anthropic's __all__ omits this client, so pyright reports it as not exported"}
from anthropic import AsyncAnthropicVertex

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

vertex_client = AsyncAnthropicVertex(region='us-east5', project_id='your-project-id')
provider = AnthropicProvider(anthropic_client=vertex_client)
model = AnthropicModel('{{ anthropic:medium.name }}', provider=provider)
agent = Agent(model)
...
```

### Microsoft Foundry

To use Claude models via [Microsoft Foundry](https://ai.azure.com/), follow the [Anthropic documentation](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry) on how to set up an `AsyncAnthropicFoundry` client and then pass it to `AnthropicProvider`:

```python {test="skip"}
from anthropic import AsyncAnthropicFoundry

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

foundry_client = AsyncAnthropicFoundry(
    api_key='your-foundry-api-key',  # Or set ANTHROPIC_FOUNDRY_API_KEY
    resource='your-resource-name',
)
provider = AnthropicProvider(anthropic_client=foundry_client)
model = AnthropicModel('{{ anthropic:medium.name }}', provider=provider)
agent = Agent(model)
...
```

See [Anthropic's Microsoft Foundry documentation](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry) for setup instructions including Entra ID authentication.

### History across platforms

Every platform verifies Claude's thinking signatures and redacted thinking, so they're sent back whichever client produced them, including [`BedrockConverseModel`](bedrock.md#claude-thinking-from-other-platforms) with a Claude model and a subclass of `AnthropicProvider` that only changes its `name`: all have the `anthropic` [canonical system](../message-history.md#sending-history-to-another-provider). Search, fetch and advisor results go along. Code execution results, and the container and file IDs they refer to, stay with the provider that ran them.

Anthropic Files API file IDs and code execution container IDs aren't sent through the `AsyncAnthropicBedrock`, `AsyncAnthropicVertex` and `AsyncAnthropicFoundry` clients, where [`anthropic_accepts_file_and_container_ids`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_accepts_file_and_container_ids] is `False`: an Anthropic [`UploadedFile`][pydantic_ai.messages.UploadedFile] raises a [`UserError`][pydantic_ai.exceptions.UserError], a [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool]'s files aren't uploaded, and no container from the history is reused.

## Task Budgets (Beta)

Anthropic's [task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets) let you give Claude an advisory token budget for a full agentic loop — including thinking, tool calls, tool results, and output — so the model can pace itself and finish gracefully as the budget is consumed. Configure them with [`AnthropicModelSettings.anthropic_task_budget`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_task_budget], which takes an [`AnthropicTaskBudget`][pydantic_ai.models.anthropic.AnthropicTaskBudget] payload and maps to `output_config.task_budget`.

Pydantic AI automatically enables Anthropic's required `task-budgets-2026-03-13` beta when this setting is present. See Anthropic's [task budget feature support](https://platform.claude.com/docs/en/build-with-claude/task-budgets#feature-support) for the current supported-model list. Task budgets are not currently documented as supported on Bedrock or Vertex. For Microsoft Foundry, check Microsoft's [Claude model capability list](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/claude-models#available-claude-models).

```python {title="anthropic_task_budget.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-opus-4-8')
settings = AnthropicModelSettings(
    anthropic_task_budget={'type': 'tokens', 'total': 20_000},
)
agent = Agent(model, model_settings=settings)
...
```

Task budgets compose with [`anthropic_effort`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_effort]: effort tunes per-step reasoning depth, while task budgets cap total work across the loop. Both fields end up under the same `output_config` object.

!!! note
    Task budgets are advisory, not a hard cap; pair them with [`max_tokens`][pydantic_ai.settings.ModelSettings.max_tokens] for an enforced ceiling.

### Carrying budgets across compaction

If you use [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] for server-side compaction, you can skip this section: the server tracks the countdown itself, so leave `remaining` unset and let `total` self-regulate.

The `remaining` field on `task_budget` is for *client-side* compaction patterns where you summarize earlier turns yourself between requests, so the server has no memory of how much budget was spent before the rewrite. Pydantic AI does not track `remaining` for you — accumulate token usage across requests yourself (e.g. from [`RunUsage`][pydantic_ai.usage.RunUsage] on each run) and pass the updated value on the next request so the countdown continues from where you left off rather than resetting to `total`. Setting `remaining` also invalidates any prompt-cache prefix that contains the budget, so if you want to preserve caching, set `total` once and let the server self-regulate against the running countdown.

!!! warning
    `task_budget.remaining` is mutually exclusive with [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction]: Anthropic rejects requests that combine the two because server-side compaction tracks the budget itself. Pydantic AI raises a [`UserError`][pydantic_ai.exceptions.UserError] before sending the request when this combination is configured. Choose one: `remaining` for client-side budget tracking, or [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] for server-side compaction.

## Prompt Caching

Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to reduce costs by caching parts of your prompts. Pydantic AI supports the unified `cache` setting as well as Anthropic's automatic caching, per-block message caching, and explicit cache breakpoints:

### Unified `cache` setting

The provider-agnostic way to enable prompt caching is the unified [`ModelSettings.cache`][pydantic_ai.settings.ModelSettings.cache] setting (or the [`Caching`][pydantic_ai.capabilities.Caching] capability), which works the same across every supporting provider:

```python {title="anthropic_unified_cache.py"}
from pydantic_ai import Agent

agent = Agent(
    '{{ anthropic:large }}',
    instructions='You are a helpful assistant.',
    model_settings={'cache': True},
)
```

On Anthropic, `cache=True` (or a retention like `cache='1h'`) places cache breakpoints at the end of the tool definitions and the static instructions, like `anthropic_cache_tool_definitions` and `anthropic_cache_instructions` below, so a new conversation reads back the prefix it shares with earlier ones. It caches the conversation with automatic caching, like `anthropic_cache`, or, on the legacy `AsyncAnthropicBedrock` client with models that don't support automatic caching, with a breakpoint at the end of the conversation (see [Automatic Caching](#automatic-caching)). Cache writes cost 1.25x the input price for the 5-minute cache and 2x for the 1-hour cache, and cache reads 0.1x; see [Caching](../capabilities/caching.md) for the trade-off, Pydantic AI's prefix-stability guarantees and how to monitor cache efficiency. The provider-specific `anthropic_cache*` settings below take precedence when any is set, and offer finer control.

### Automatic Caching

The Anthropic-specific way to use automatic caching is [`AnthropicModelSettings.anthropic_cache`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache]. This uses Anthropic's [automatic caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching), passing a top-level `cache_control` parameter so the server automatically applies a cache breakpoint to the last cacheable block in each request:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

handbook = '...'  # a long document, above the model's minimum cacheable length

agent = Agent(
    '{{ anthropic:large }}',
    instructions=f'Answer questions about this employee handbook:\n\n{handbook}',
    model_settings=AnthropicModelSettings(
        anthropic_cache=True,
    ),
)

result1 = agent.run_sync('How many vacation days do new employees get?')

result2 = agent.run_sync('And after five years?', message_history=result1.all_messages())
print(f'Cache write: {result1.response.usage.cache_write_tokens}')
print(f'Cache read: {result2.response.usage.cache_read_tokens}')
print(f'Cache hit ratio: {result2.response.usage.cache_hit_ratio}')
```

This is ideal for multi-turn conversations where the cache breakpoint should move forward as the conversation grows. You can also specify a custom TTL with `anthropic_cache='1h'`.

The `AsyncAnthropicBedrockMantle`, `AsyncAnthropicVertex` and `AsyncAnthropicFoundry` clients support automatic caching. The legacy `AsyncAnthropicBedrock` client uses it on the models Anthropic [documents](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy#api-model-ids) as served from the Messages API infrastructure. On other models, `anthropic_cache` falls back to per-block caching on the last user message, which gives multi-turn conversations the same benefit.

Whether a model uses automatic caching comes from its [profile](overview.md#models-and-providers)'s `supports_auto_cache`, for both `anthropic_cache` and the unified `cache` setting. If Pydantic AI doesn't yet know that a model or platform supports it, set the flag yourself:

```python {test="skip" typecheck="skip - anthropic's __all__ omits this client, so pyright reports it as not exported"}
from anthropic import AsyncAnthropicBedrock

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'us.anthropic.claude-sonnet-4-6',
    provider=AnthropicProvider(anthropic_client=AsyncAnthropicBedrock()),
    profile=AnthropicModelProfile(supports_auto_cache=True),
)
```

Setting it to `False` makes `anthropic_cache` use the per-block fallback, which can help with Anthropic-compatible gateways that don't take the top-level parameter.

### Per-block Message Caching

As an alternative to `anthropic_cache`, [`AnthropicModelSettings.anthropic_cache_messages`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_messages] adds per-block `cache_control` to the last content block of the final message instead of using Anthropic's top-level automatic caching parameter. Use this with Anthropic-compatible gateways and proxies (such as MiniMax, OpenRouter, or LiteLLM) that accept the Anthropic message format but don't support top-level automatic caching:

```python {test="skip"}
from anthropic import AsyncAnthropic

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

client = AsyncAnthropic(
    api_key='your-api-key',
    base_url='https://your-anthropic-compatible-gateway.example.com',
)

model = AnthropicModel(
    '{{ anthropic:medium.name }}',
    provider=AnthropicProvider(anthropic_client=client),
)
agent = Agent(
    model,
    model_settings=AnthropicModelSettings(
        anthropic_cache_messages=True,
    ),
)

result = agent.run_sync('What is the capital of France?')
print(result.output)
```

You can also specify a custom TTL with `anthropic_cache_messages='1h'`. `anthropic_cache_messages` cannot be combined with `anthropic_cache`.

### Explicit Cache Breakpoints

In addition to automatic caching, Pydantic AI provides several ways to place cache breakpoints on specific content:

1. **Cache User Messages with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker in your user messages to cache everything before it
2. **Cache the Final Message Block**: Set [`AnthropicModelSettings.anthropic_cache_messages`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_messages] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly
3. **Cache System Instructions**: Set [`AnthropicModelSettings.anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly
4. **Cache Tool Definitions**: Set [`AnthropicModelSettings.anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly

Anthropic rejects a request where a one-hour breakpoint comes after a five-minute one (it processes tools, then system instructions, then messages), so when a one-hour breakpoint, including the automatic caching one at the end of the conversation, follows five-minute ones, those earlier breakpoints are raised to one hour.

#### Example: Comprehensive Caching Strategy

Combine automatic caching with explicit breakpoints for maximum savings. Automatic caching handles the conversation, while explicit breakpoints pin system instructions and tool definitions:

```python {test="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache=True,                   # Server auto-caches last block
        anthropic_cache_instructions=True,      # Explicitly cache system instructions
        anthropic_cache_tool_definitions='1h',  # Explicitly cache tool definitions with 1h TTL
    ),
)

@agent.tool
def search_docs(ctx: RunContext, query: str) -> str:
    """Search documentation."""
    return f'Results for {query}'


result = agent.run_sync('Search for Python best practices')
print(result.output)
```

### Smart Instruction Caching

When you use `anthropic_cache_instructions` with both static and dynamic [instructions](../agent.md#instructions), Pydantic AI automatically places the cache boundary at the optimal point. Static instructions (from `Agent(instructions=...)`) are sorted before dynamic instructions (from `@agent.instructions` functions or [toolsets](../toolsets.md)), and the cache point is placed after the last static instruction part. Dynamic instructions with [`on_change='append'`](../agent.md#appending-instruction-changes) sort between the two, since their changes are appended to the history rather than rewriting the prefix, and get a second cache point after them. The first one is what a new conversation reads even when its append-mode instructions differ, such as per-user memory, and the second also covers append-mode instructions that are the same in every conversation, such as the catalogs Pydantic AI computes.

This means your stable, static instructions are cached efficiently, while dynamic instructions (which may change between requests) remain outside the cache boundary, so a change doesn't invalidate the cached static instructions. It does invalidate the cached conversation after them, so for instructions that change mid-conversation, use [`on_change='append'`](../agent.md#appending-instruction-changes), which appends each change to the history instead; [keep untrusted text out of instructions](../agent.md#untrusted-instruction-text).

```python {test="skip"}
from datetime import date

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    deps_type=str,
    instructions='You are a helpful customer service agent. Follow company policy.',  # (1)!
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,  # (2)!
    ),
)


@agent.instructions
def dynamic_context(ctx: RunContext[str]) -> str:  # (3)!
    return f"Customer name: {ctx.deps}. Today's date: {date.today()}."


result = agent.run_sync('What is your return policy?', deps='Alice')
print(result.output)
```

1. Static instructions are cached across requests.
2. Enables smart cache placement at the static/dynamic boundary.
3. Dynamic instructions change per-request and are not cached.

### Fine-Grained Control with CachePoint

Use manual `CachePoint` markers to control cache locations precisely:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Instructions...',
)

# Manually control cache points for specific content blocks
result = agent.run_sync([
    'Long context from documentation...',
    CachePoint(),  # Cache everything up to this point
    'First question'
])
print(result.output)
```

### Accessing Cache Usage Statistics

Access cache usage statistics via `result.usage`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache=True,
    ),
)

result = agent.run_sync('Your question')
usage = result.usage
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

`cache_write_tokens` counts all cache writes. When some of them used a one-hour TTL, which Anthropic bills at a higher rate than five-minute writes, `cache_write_1h_tokens` counts those, and the cost is calculated at the one-hour rate for them. Anthropic's own counts are also in `usage.details`: `'ephemeral_1h_input_tokens'`, and `'compaction_ephemeral_1h_input_tokens'` for writes made during [message compaction](#message-compaction).

### Cache Diagnostics

When a request misses the cache unexpectedly, Anthropic's [cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics) can tell you why. Set [`AnthropicModelSettings.anthropic_cache_diagnostics`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_diagnostics] to `True`, and each request names the most recent Anthropic response in the message history as the one to compare against. When the prompt prefix diverged, the response's [`provider_details`][pydantic_ai.messages.ModelResponse.provider_details] has a `'cache_diagnostics'` key with Anthropic's result, such as `{'cache_miss_reason': {'type': 'tools_changed', 'cache_missed_input_tokens': 3713}}`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Instructions...',
    model_settings=AnthropicModelSettings(anthropic_cache=True, anthropic_cache_diagnostics=True),
)

result1 = agent.run_sync('Your question')
result2 = agent.run_sync('Follow-up question', message_history=result1.all_messages())
diagnostics = (result2.response.provider_details or {}).get('cache_diagnostics')
if diagnostics and (reason := diagnostics['cache_miss_reason']):
    print(f"Cache miss: {reason['type']}, {reason.get('cache_missed_input_tokens')} tokens")
```

The `type` is one of `model_changed`, `system_changed`, `tools_changed`, `messages_changed`, `previous_message_not_found` or `unavailable`. The key is absent when Anthropic found no divergence, and holds `{'cache_miss_reason': None}` when the comparison hadn't finished yet.

Diagnostics are free and don't affect caching, but they're off by default: Anthropic keeps a short-lived fingerprint of each request that asks for them (hashes and token counts, not prompt content), and rejects them for HIPAA-enabled organizations. They're only available on the Claude API, so the setting is ignored on Bedrock, Vertex AI and Microsoft Foundry.

### Cache Point Limits

Anthropic enforces a maximum of 4 cache points per request. Pydantic AI automatically manages this limit to ensure your requests always comply without errors.

#### How Cache Points Are Allocated

Cache points can come from several sources:

1. **Automatic caching**: Via `anthropic_cache` (the server applies 1 cache point to the last cacheable block)
2. **Final message block**: Via `anthropic_cache_messages` setting (adds cache point to last message content block)
3. **System Prompt**: Via `anthropic_cache_instructions` setting (adds cache point to the last static system prompt block, and a second one after any `on_change='append'` instructions)
4. **Tool Definitions**: Via `anthropic_cache_tool_definitions` setting (adds cache point to last tool definition)
5. **Messages**: Via `CachePoint` markers (adds cache points to message content)

Each setting uses **at most 1 cache point**, except `anthropic_cache_instructions` with append-mode instructions, but you can combine them — except `anthropic_cache` and `anthropic_cache_messages`, which are mutually exclusive. If the total exceeds 4, Pydantic AI automatically trims the cache point after the append-mode instructions first (when another one ends the static instructions or system prompt), then excess cache points from older messages.

#### Example: Combining Automatic and Explicit Caching

Define an agent with automatic caching plus explicit breakpoints:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache=True,                   # 1 cache point (server-applied)
        anthropic_cache_instructions=True,      # 1 cache point
        anthropic_cache_tool_definitions=True,  # 1 cache point
    ),
)

@agent.tool_plain
def my_tool() -> str:
    return 'result'


# 3 of 4 slots used (1 automatic + 1 instructions + 1 tools)
# Room for 1 more explicit CachePoint marker
result = agent.run_sync([
    'Context', CachePoint(),  # 4th cache point - OK
    'Question'
])
print(result.output)
usage = result.usage
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

#### Automatic Cache Point Limiting

When explicit cache points from all sources (settings + `CachePoint` markers) exceed the available budget, Pydantic AI automatically removes excess cache points from **older message content** (keeping the most recent ones).

Define an agent with 2 explicit cache points from settings:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,      # 1 cache point
        anthropic_cache_tool_definitions=True,  # 1 cache point
    ),
)

@agent.tool_plain
def search() -> str:
    return 'data'

# Already using 2 cache points (instructions + tools)
# Can add 2 more CachePoint markers (4 total limit)
result = agent.run_sync([
    'Context 1', CachePoint(),  # Oldest - will be removed
    'Context 2', CachePoint(),  # Will be kept (3rd point)
    'Context 3', CachePoint(),  # Will be kept (4th point)
    'Question'
])
# Final cache points: instructions + tools + Context 2 + Context 3 = 4
print(result.output)
usage = result.usage
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

**Key Points**:
- Tool cache points and the cache point after the static instructions are **always preserved**
- When static instructions or a system prompt come before them, the cache point after `on_change='append'` instructions is dropped before any message cache point: within a conversation the message cache points already cover those instructions
- `anthropic_cache` counts as 1 cache point, just like `anthropic_cache_tool_definitions` (and `anthropic_cache_instructions` without append-mode instructions)
- Excess `CachePoint` markers in messages are removed from oldest to newest when the limit is exceeded
- This ensures critical caching (instructions/tools) is maintained while still benefiting from message-level caching

## Carried-over thinking

A [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] from history that can't ride Anthropic's native reasoning channel — most often because its signature didn't survive your storage round trip — is sent as text in a `user` message of its own ahead of the assistant turn it came from, wrapped in an `<assistant_thinking>` tag. That is why a request can carry a `user` turn you never wrote. See [the thinking page](../capabilities/thinking.md#carried-over-thinking) for why the assistant turn is the wrong place for it, and for the opt-out.

## Token counting

[`count_tokens()`][pydantic_ai.models.Model.count_tokens] and [`count_tokens_before_request`][pydantic_ai.usage.UsageLimits.count_tokens_before_request] use Anthropic's [token counting endpoint](https://platform.claude.com/docs/en/build-with-claude/token-counting). On the `AsyncAnthropic` client, the count includes the definitions of the web search, web fetch, code execution, tool search and advisor tools, so it matches the request's input tokens. The Bedrock, Vertex AI and Foundry clients leave those tools out of the count, as their count endpoints haven't been verified to accept them, and no client counts MCP servers, which the endpoint rejects.

[`anthropic_count_tokens_accepts_server_tools`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_count_tokens_accepts_server_tools] in the model's `profile` controls this. Set it to `False` if your `AsyncAnthropic` client's `base_url` points at a proxy whose count endpoint rejects server tools, or to `True` on a platform client whose count endpoint accepts them. With it off, a request whose function tools are all deferred can't be counted.

## Batch API

`AnthropicModel` supports [batch processing](../batch.md) through the Claude API's
[Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing), which charge
[50% of standard API prices](https://platform.claude.com/docs/en/build-with-claude/batch-processing#pricing).
Batching and prompt-caching discounts apply together. One batch can hold up to 100,000 requests or 256 MB, and its
requests can use different models and settings. Beta flags apply to a whole batch, so requests that need different
ones, for example through `anthropic_betas`, go into separate batches. A request's `custom_id` must be 1 to 64
letters, digits, `_` or `-`.

Message Batches aren't available through Bedrock, Vertex AI or Microsoft Foundry, nor yet through the
[Pydantic AI Gateway](../gateway.md): use `AnthropicProvider` with a Claude API key instead. They need an API key or
auth token on the client. They don't support streaming, `anthropic_speed` or `max_tokens=0` cache pre-warming.
Configure transport timeouts and custom headers on the Anthropic client: per-request `timeout` and
`extra_headers` are unsupported. Retrieving a batch needs the endpoint and beta flags that submitted it, but not the same API key: any key the workspace accepts works.

## Mid-conversation system messages

Adding an instruction to the agent's `system_prompt` partway through a long session rewrites the front of the prompt, which invalidates every cached prefix behind it. Anthropic avoids that by accepting a system message *inside* the conversation, at the instruction's own position in the history rather than ahead of it, so everything cached up to that point stays cached.

Any [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] outside the first [`ModelRequest`][pydantic_ai.messages.ModelRequest] is a mid-conversation instruction — whether it came from a stored `message_history` or from [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] during a run. There's nothing extra to turn on:

```python {title="mid_conversation_system_prompt.py"}
from pydantic_ai import Agent, RunContext, SystemPromptPart

agent = Agent('anthropic:claude-opus-4-8', system_prompt='You are a code reviewer.')


@agent.tool
def require_type_annotations(ctx: RunContext) -> str:
    ctx.enqueue(SystemPromptPart(content='Every suggestion must include explicit type annotations.'))
    return 'rule added'
```

Keeping the instruction in place leaves the prefix ahead of it reusable, but it doesn't enable caching on its own — that still comes from [`anthropic_cache`](#automatic-caching), `anthropic_cache_messages`, or an explicit [`CachePoint`][pydantic_ai.messages.CachePoint]. A `CachePoint` at the end of an enqueued batch caches everything before it in that batch, the instruction included. One with more content after it caches up to where you put it and leaves the instruction outside: the instruction is sent after the content it accompanies, so it can't be inside a boundary that content is outside of.

Support varies by model and by transport — the [Microsoft Foundry](#microsoft-foundry) integration doesn't serve the role, and some Claude models accept the entry without acting on it. Anthropic's [mid-conversation system messages docs](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages) have the current list. Pydantic AI picks the rendering that works for the model and transport you're using, falling back to a `<system>`-tagged user message at the same position, so the instruction applies where you put it either way.

The difference between the two shows up on instructions a model *should* be wary of taking from its user: given the native entry, Claude will lift a restriction its top-level prompt set, and given the identical text in a `<system>` tag it refuses. For an instruction with nothing to distrust, such as a change of format, both work.

See [mid-conversation system prompts](../message-history.md#mid-conversation-system-prompts) for how these behave across providers, how to phrase one, and why untrusted content doesn't belong in one.

!!! note "Placement"
    Anthropic requires a system message to sit between a user turn and the model's reply, so Pydantic AI nudges the position when a history doesn't already satisfy that: an instruction arriving with no user content alongside it gets a minimal `.` user message to follow, and one that would land ahead of another user turn moves to just before the reply it governs. Neither changes which turn the instruction applies to — only where it sits on the wire.

### Refusals on repeated prompts

Claude's `reasoning_extraction` classifier can refuse a conversation that repeats the same user prompt while the model's earlier reasoning is in context, and a refusal is more likely right after an instruction changes, whether the instructions are rewritten or [appended](../agent.md#appending-instruction-changes). Conversations with varied prompts and tool loops were unaffected in testing. The refusal propagates as a [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError]. If you hit it, configure a [fallback model](overview.md#fallback-model) or strip thinking from history with [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory].

### Turn-scoped system messages

A [request-scoped](../message-history.md#request-scoped-system-prompts) `SystemPromptPart` (`scope='request'`) is sent as a [turn-scoped system message](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages#turn-scoped-system-messages): a `{'role': 'system'}` entry with `clear_at: 'next_user_message'`, which the API stops rendering once a later user turn exists, including one that only carries tool results. Pydantic AI sends every one of them with every request, current or not, and adds the `mid-conversation-system-clear-at-2026-08-21` beta header for you.

That matters most on models with [preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking), such as Claude Opus 5.5, Sonnet 5.5 and Fable 5.1: they tie each thinking block to the exact conversation it was produced in, so deleting an earlier reminder from a later request invalidates every thinking block after it, while a cleared one left in place doesn't. A cleared entry costs no input tokens and is never part of a cache key, so `anthropic_cache_messages` and `anthropic_cache` put their breakpoints on the content before it.

Support follows mid-conversation system messages, on the same models and transports. Elsewhere, including on models that don't take the `system` role, a request-scoped prompt is sent as `<system>`-tagged user text with its own request only, and the cache breakpoint goes before it. To opt out of the beta on a model that has it, for example if your account doesn't have access, turn the profile flag off; request-scoped prompts are then sent only while they're current:

```python {title="request_scoped_opt_out.py"}
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.profiles.anthropic import AnthropicModelProfile

model = AnthropicModel(
    '{{ anthropic:large.name }}',
    profile=AnthropicModelProfile(supports_request_scoped_system_prompts=False),
)
```

## Fast mode

Fast mode provides higher output tokens per second and is currently supported on **Claude Opus 4.6**, **Claude Opus 4.7**, **Claude Opus 4.8**, **Claude Opus 5**, and **Claude Opus 5.5**. It is a research preview. Set [`anthropic_speed`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_speed] to `'fast'` to enable it; Pydantic AI automatically adds the required `fast-mode-2026-02-01` beta. On unsupported models, `anthropic_speed='fast'` is ignored with a `UserWarning`. For pricing, rate limits, and the latest list of supported models, see the [Anthropic fast mode docs](https://platform.claude.com/docs/en/build-with-claude/fast-mode).

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-opus-4-8',
    model_settings=AnthropicModelSettings(anthropic_speed='fast'),
)
...
```

!!! note "Prompt cache interaction"
    Switching between `'fast'` and `'standard'` invalidates the prompt cache. Requests at different speeds do not share cached prefixes, so pick one speed per cache-sensitive conversation.

!!! note "Bedrock, Vertex, and Foundry"
    Fast mode is only available on the direct Anthropic API. Bedrock, Vertex, and Foundry clients do not support the `speed` parameter, so `anthropic_speed='fast'` is ignored with a `UserWarning` on those clients.

## Forced tool choice

Most Anthropic models let you force a tool call via [`tool_choice='required'`][pydantic_ai.settings.ModelSettings.tool_choice] (or a list of tool names), except while [extended thinking](../capabilities/thinking.md#anthropic) is enabled. [Adaptive thinking](../capabilities/thinking.md#adaptive-thinking-effort) accepts forcing, but Claude answers a forced tool choice without thinking. Anthropic documents **Claude Fable 5.1**, **Claude Mythos 5.1**, **Claude Opus 5.5**, and **Claude Sonnet 5.5** as rejecting a forced tool choice unconditionally, even without thinking, and Pydantic AI marks those with [`supports_forced_tool_choice=False`][pydantic_ai.profiles.ModelProfile.supports_forced_tool_choice].

On a model that doesn't support forcing, or while extended thinking is enabled:

- An explicit `tool_choice='required'` (or a list of tool names) raises a [`UserError`][pydantic_ai.exceptions.UserError]; use `tool_choice='auto'` instead.
- A `required` choice that Pydantic AI resolved on your behalf (e.g. from an [output tool](../output.md#tool-output)) falls back softly to `'auto'`. If the resolved choice named a single tool, the available tool list is filtered to that tool while `tool_choice` remains `'auto'`, which invalidates Anthropic's prompt cache since the cached prefix includes the tool array. The model may therefore answer with text instead of calling it; when an output tool is required, Pydantic AI retries with a prompt to call a tool.

While adaptive thinking is on, an explicit forcing `tool_choice` is still sent, and that request comes back without thinking. A forced choice that Pydantic AI resolved on your behalf falls back to `'auto'` instead, as above, so the model keeps thinking.

[Tool Output](../output.md#tool-output) forces a call to the output tool on every request that can't end with text, so a bare structured `output_type` uses [Native Output](../output.md#native-output), where the model supports it, whenever the request thinks: with a thinking setting, or with none on a model that [thinks by default](../capabilities/thinking.md#adaptive-thinking-effort), like Claude Opus 5 and later, Claude Sonnet 5, Claude Haiku 5.5, and the Claude Fable and Mythos models. On a model without JSON schema support, extended thinking falls back to [Prompted Output](../output.md#prompted-output), and adaptive thinking keeps Tool Output with the output tool offered under `tool_choice='auto'`. To keep Tool Output, turn thinking off with `thinking=False` where the model allows it. An explicit `ToolOutput(...)` keeps Tool Output, with the output tool offered under `tool_choice='auto'`.

The output mode is chosen on the run's first request and kept for the rest of the run (except behind a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], where each model chooses on every request), so a thinking setting that changes from step to step (through a [`model_settings`][pydantic_ai.agent.Agent.__init__] callable or a capability) doesn't add or remove the output tool between requests and [invalidate the prompt cache](../capabilities/caching.md#what-invalidates-a-cache). A run that starts with Tool Output keeps it once thinking turns on, with the output tool offered under `tool_choice='auto'`.

## Schema complexity limits {#schema-complexity-limits}

Anthropic compiles every strict schema in a request into a grammar that constrains generation, and enforces [schema complexity limits](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#schema-complexity-limits) on it. On Anthropic, these schemas are strict:

- The output schema of [Native Output](../output.md#native-output), which is always strict: `NativeOutput(..., strict=False)` raises a [`UserError`][pydantic_ai.exceptions.UserError]. This includes a bare structured `output_type` on a request that thinks, which uses Native Output as described [above](#forced-tool-choice).
- Function tools and output tools with [`strict=True`](../tools-advanced.md#strict-mode), such as `ToolOutput(MyModel, strict=True)`. Tools are not strict unless you set `strict=True`, so default Tool Output is unaffected.

[Prompted Output](../output.md#prompted-output) sends no schema for Anthropic to compile, so it is never affected.

Anthropic counts its explicit limits on the number of strict tools, optional parameters, and parameters with union types across all strict schemas in the request combined, so strict function tools and a Native Output schema add up. In a Pydantic model, a field with a default is optional, and a field typed `X | None` or another union has a union type. Anthropic also rejects a recursive schema in strict mode, along with the other [unsupported features](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations).

Pydantic AI checks the explicit limits and recursion before it sends a request:

- A bare structured `output_type` whose Native Output schema would be over a limit uses [Tool Output](../output.md#tool-output) instead. Its output tool isn't strict, so it doesn't count toward the limits, and on a request that thinks it's offered under `tool_choice='auto'` as described [above](#forced-tool-choice), so the model may answer with text and get a retry prompt. The resolved output mode is recorded in the `model_request_parameters` attribute of the request's [span](../logfire.md).
- Strict schemas you asked for explicitly, with `NativeOutput(...)` or `strict=True` on a tool or `ToolOutput`, raise a [`ModelRequestRejectedError`][pydantic_ai.exceptions.ModelRequestRejectedError] that names the limit when they're over it, without waiting for a 400 from Anthropic. It's the same error category as that 400, so a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] still falls back to its next model, and `except ModelAPIError` still catches it, though it isn't a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] as no request was sent.

The check applies to the Anthropic API, and to Claude on Amazon Bedrock through [`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel] or an `AnthropicModel` with a Bedrock client. On Google Cloud and Microsoft Foundry, where the limits haven't been verified, Pydantic AI sends these requests for Anthropic to judge, unless you set [`anthropic_enforces_strict_schema_limits=True`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_enforces_strict_schema_limits] in the model's `profile`.

A schema within the explicit limits can still exceed Anthropic's internal limits on grammar size, for example with deeply nested models or many optional or union-typed fields. The request then fails with a 400 saying that the compiled grammar is too large or that the schema is too complex, and that 400 can arrive immediately or only after the request has waited for up to Anthropic's 180-second compilation timeout. Pydantic AI raises that 400 as a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] (and a [`ModelRequestRejectedError`][pydantic_ai.exceptions.ModelRequestRejectedError]) whose [`hint`][pydantic_ai.exceptions.ModelHTTPError.hint] points here, without retrying it. If you set a [`timeout`][pydantic_ai.settings.ModelSettings.timeout] shorter than the compilation, the request times out before the error arrives, and the Anthropic SDK retries a timed-out request, twice by default, so a run can wait several times the timeout before it fails.

To stay within the limits, simplify the schema following [Anthropic's tips](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#tips-for-reducing-schema-complexity), such as making fields required and flattening nested models, and set `strict=True` only on the tools that need it. For a large output schema, wrap the output type in [`ToolOutput`][pydantic_ai.output.ToolOutput] without `strict=True` or in [`PromptedOutput`][pydantic_ai.output.PromptedOutput]: neither is compiled, so the model isn't constrained to the schema, and Pydantic AI validates its output and asks it to try again when validation fails.

## Thinking block binding

**Claude Fable 5.1**, **Claude Opus 5.5**, **Claude Sonnet 5.5**, and **Claude Haiku 5.5** bind each thinking block to the conversation prefix that produced it. Anthropic [documents a 400](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting) (`The block is bound to a different conversation`) when replaying history after `system`, `tools`, or earlier `messages` change. Two ordinary Pydantic AI features change the prefix:

- a [dynamic instructions](../agent.md#instructions) function whose text differs between runs, unless it uses [`on_change='append'`](../agent.md#appending-instruction-changes), and
- a [filtered toolset](../toolsets.md#filtering-tools) that advertises a new tool mid-conversation, unless the tool uses [deferred loading](../toolsets.md#deferred-loading).

Both are the same instability that [costs you a provider's prompt cache](../capabilities/caching.md#what-invalidates-a-cache): a request prefix that changes between turns. The thinking block turns it into a 400 you can see; the cache turns it into a bill you can't — every request after the change re-sends the whole conversation at uncached rates, silently. Where the prefix can be held stable, that is worth more than handling the rejection.

Anthropic enforces the check for accounts created on or after 31 August 2026. For an older account it records the mismatch but returns the error only if the request sets `thinking.block_binding.prefix_mismatch_behavior`.

**Pydantic AI sets nothing by default**, so an older account keeps replaying its reasoning untouched. Where the check is enforced, the rejected request is retried once with `prefix_mismatch_behavior='drop_block'`: the stale block is dropped, the run continues, and a [`AnthropicStaleThinkingBlockWarning`][pydantic_ai.models.anthropic.AnthropicStaleThinkingBlockWarning] explains what happened. **The model no longer sees that turn's reasoning** — the trade is one turn's thinking against a failed run. Models marked [`anthropic_binds_thinking_blocks=True`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_binds_thinking_blocks] are the only ones that retry. A request whose thinking type isn't `adaptive`, such as `anthropic_thinking={'type': 'between_tools'}`, is not retried, because Anthropic accepts `block_binding` only alongside adaptive thinking: keep that conversation's prefix stable instead.

Anthropic applies the drop to one request only. Its response records the transformation, so Pydantic AI uses that response history to keep sending `drop_block` for the rest of the affected conversation. This avoids another rejected request on every later turn without storing mutable state on the model or changing an unrelated conversation. The same behavior applies to [`count_tokens()`][pydantic_ai.models.Model.count_tokens]: token counting retries the first matching rejection once, and later counts carrying the recovery history send the drop immediately. A recovery seen only while counting applies only to later counts, because counting omits MCP servers, and on Bedrock, Vertex AI and Foundry server-side tools too (see [Token counting](#token-counting)), and can therefore have a different prefix from inference. An explicit `block_binding` still wins for both counting and inference.

The automatic behavior is available through the direct Anthropic API, Pydantic AI Gateway and other `AsyncAnthropic`-compatible proxies, Claude Platform on AWS, both Amazon Bedrock clients (the Bedrock Messages API's `AsyncAnthropicBedrockMantle` and the legacy `AsyncAnthropicBedrock`), and Google Vertex AI. It is not enabled for Microsoft Foundry, where this beta is not currently documented. Explicit settings remain pass-through on every transport.

Anthropic reports every drop, and Pydantic AI surfaces it two ways. Under [instrumentation](../logfire.md) the Pydantic AI model request span carries an `anthropic.input_transformations` event, so a drop is visible in the trace as it happens; an unrelated ambient application span is never modified. On the response it is always recorded in `provider_details`:

```python {title="dropped_thinking_blocks.py"}
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse

agent = Agent('anthropic:claude-fable-5-1')
result = agent.run_sync('What is the capital of France?')

for message in result.new_messages():
    if isinstance(message, ModelResponse) and message.provider_details:
        for transformation in message.provider_details.get('input_transformations', []):
            print(transformation['path'], transformation['reason'])
```

To skip the rejected request — and the warning — ask for the drop up front. Pydantic AI sends an explicit `block_binding` as given, with the beta the field requires, and never retries:

```python {title="drop_stale_thinking_blocks.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

settings: AnthropicModelSettings = {
    'anthropic_thinking': {
        'type': 'adaptive',
        'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
    }
}
agent = Agent('anthropic:claude-fable-5-1', model_settings=settings)
...
```

To fail loudly instead of losing the reasoning, set `'error'` in the same place. To keep the retry but stop hearing about it, filter the warning:

```python {title="silence_stale_thinking_block_warning.py"}
import warnings

from pydantic_ai.models.anthropic import AnthropicStaleThinkingBlockWarning

warnings.simplefilter('ignore', AnthropicStaleThinkingBlockWarning)
```

There is no third behavior: `prefix_mismatch_behavior` is either `'error'` or `'drop_block'`. Passing `None` is how you ask for Anthropic's account default explicitly, which keeps the block only where the check isn't enforced.

Models that don't bind thinking blocks are unaffected *by default*: with no explicit `block_binding`, their requests carry neither the field nor the binding beta, and a 400 from them is never retried. Setting `block_binding` yourself puts both on the wire for any model — the beta follows the field, not the profile flag — and such a request is never retried either. Anthropic documents **Claude Mythos 5.1** as not running this check, so it is not marked either.

## Message Compaction

Anthropic supports [automatic context compaction](https://docs.anthropic.com/en/docs/build-with-claude/compaction) to manage long conversations. When input tokens exceed a configured threshold, the API automatically generates a summary that replaces older messages while preserving context.

After compaction, subsequent requests send only the compacted window, from the latest compaction block onward, which reduces request size — the API ignores earlier content either way. The standing system prompt is unaffected: it's sent as the separate `system` parameter, which compaction doesn't replace.

The easiest way to enable compaction is with the [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] capability:

```python {title="anthropic_compaction.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[AnthropicCompaction(token_threshold=100_000)],
)
```

The capability accepts:

- **`token_threshold`** (default: 150,000, minimum: 50,000): Compaction triggers when input tokens exceed this value.
- **`instructions`**: Custom instructions for how the summary should be generated.
- **`pause_after_compaction`**: When `True`, the response stops after the compaction block with `stop_reason='compaction'`, allowing explicit handling before continuing.

Alternatively, you can configure compaction directly via model settings using [`anthropic_context_management`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_context_management]:

```python {title="anthropic_compaction_settings.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent('{{ anthropic:medium }}')
result = agent.run_sync(
    'Hello!',
    model_settings=AnthropicModelSettings(
        anthropic_context_management={
            'edits': [{'type': 'compact_20260112', 'trigger': {'type': 'input_tokens', 'value': 100_000}}]
        }
    ),
)
```

!!! note
    Compaction blocks returned by Anthropic contain readable text summaries. They are automatically round-tripped in subsequent requests when included in the message history.

### Compacting on demand

To compact on demand from an agent, add the [`NativeCompaction`](../capabilities/compaction.md#compacting-on-demand) capability and [request compaction](../capabilities/compaction.md#requesting-compaction), from the run or with `compact=` on the run method. It uses the API below for the model that serves the request, with the conversation's own settings and tools, and sends a focus as the instructions.

To decide yourself when to compact without either, call [`compact_messages`][pydantic_ai.models.anthropic.AnthropicModel.compact_messages] on the model. It sends the conversation with Anthropic's [on-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand) parameter, which summarizes it instead of replying, and returns a [`ModelResponse`][pydantic_ai.messages.ModelResponse] holding one [`CompactionPart`][pydantic_ai.messages.CompactionPart]: a readable summary, with its signature in `provider_details`. Add it to the message history, and later requests send it in place of the messages before it:

```python {title="anthropic_compact_messages.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('{{ anthropic:medium.name }}')
agent = Agent(model)


async def main():
    result = await agent.run('Our API uses PostgreSQL 17 behind pgbouncer. Note that down.')
    messages = result.all_messages()

    compacted = await model.compact_messages(
        ModelRequestContext(
            model=model,
            messages=messages,
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        ),
        instructions='Keep every database decision.',
    )

    result = await agent.run('Which database do we use?', message_history=[*messages, compacted])
    print(result.output)
```

`instructions` is an optional focus for the summary. Anthropic replaces its whole summarization prompt with the instructions it's given, so Pydantic AI sends the focus inside a full summarization prompt of its own.

The compaction request is the conversation's own request with the `compaction` parameter added, so it reads the conversation's prompt cache, as long as the context carries the same settings and request parameters (tools, output type) as the conversation's requests. The [`ModelRequestContext`][pydantic_ai.models.ModelRequestContext] a [`before_model_request`](../hooks.md) hook receives does; the one built in the example above only matches an agent without tools, structured output or settings. Anthropic rejects some parameters on it, which are left out: `stop_sequences`, the structured output format, the `remaining` [task budget](#task-budgets-beta) and [`anthropic_context_management`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_context_management]. A forcing `tool_choice` is relaxed to `auto`, and the calls of a trailing response that has no tool results yet are dropped. Images and documents in the summarized messages don't survive compaction.

The usage of the returned response is the compaction's, which Anthropic reports in `usage.iterations` rather than in the top-level counts. The context it leaves behind, as [`ctx.context_window_used`][pydantic_ai.tools.RunContext.context_window_used] reads it, is the summary alone. If Anthropic returns no summary, for example because `max_tokens` cut it off, [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] is raised rather than an empty part returned.

Requests that carry the block send the `compact-2026-09-04` beta it needs, and don't get the default threshold compaction edit, which Anthropic rejects alongside it. For the same reason, don't combine on-demand compaction with [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] or a `compact_20260112` edit of your own in one conversation: Anthropic rejects every request that carries both the block and the edit. The block works on every model that supports on-demand compaction, not only the one that wrote it: Claude Fable 5 and 5.1, Mythos 5, 5.1 and Preview, Opus 4.6 and later, Sonnet 4.6 and later, and Haiku 5.5. `compact_messages` raises a [`UserError`][pydantic_ai.exceptions.UserError] for other models, according to the [`supports_compaction_on_demand`][pydantic_ai.profiles.ModelProfile.supports_compaction_on_demand] profile flag.

On-demand compaction works the same way on every client `AnthropicModel` takes, as long as the platform serves the model:

| Platform | Client | Support |
| --- | --- | --- |
| Claude API | `AsyncAnthropic` | Documented and verified |
| [Claude Platform on AWS](https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws) | `AsyncAnthropicAWS` | Documented, not verified by Pydantic AI |
| [Amazon Bedrock](#aws-bedrock) Messages API | `AsyncAnthropicBedrockMantle` | Not documented, verified |
| [Amazon Bedrock](#aws-bedrock) InvokeModel (legacy) | `AsyncAnthropicBedrock` | Not documented, verified |
| [Google Cloud Vertex AI](#google-cloud) | `AsyncAnthropicVertex` | Documented, not verified by Pydantic AI |
| [Microsoft Foundry](#microsoft-foundry) | `AsyncAnthropicFoundry` | Documented, not verified by Pydantic AI |

"Documented" means Anthropic's [on-demand compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand) list the platform; "verified" means a compaction request and a request sending the block back both succeed there. Bedrock serves fewer models than the Claude API: its Messages API doesn't serve Claude Haiku 5.5 or Sonnet 4.6, for example.

## Server-side tool loops

With a server-side tool such as [web search](../native-tools.md#web-search-tool), Anthropic can sample the model several times within one request, once per tool round. The response's usage sums every pass, which is what Anthropic bills, but each pass re-reads the conversation so far, so the sum can be several times the context the model was given. A response in which [compaction](#message-compaction) ran likewise sums the compaction pass and the pass after it, which saw only the compacted context.

Anthropic reports each pass only under a beta that returns `usage.iterations`. With it, [`ctx.context_window_used`][pydantic_ai.tools.RunContext.context_window_used], [`per_request_input_tokens_limit`][pydantic_ai.usage.UsageLimits.per_request_input_tokens_limit], and [harness compaction](../harness/compaction.md) measure the context by the last pass's input tokens, cached tokens included. Without it, they read the summed usage.

[`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] and [`anthropic_context_management`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_context_management] send the `compact-2026-01-12` beta, and the betas that the [`AdvisorTool`][pydantic_ai.native_tools.AdvisorTool] and [`anthropic_task_budget`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_task_budget] send return `usage.iterations` too. Otherwise, send the compaction beta yourself:

```python {title="anthropic_server_tool_loop_context.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[WebSearch()],
    model_settings=AnthropicModelSettings(anthropic_betas=['compact-2026-01-12']),
)
```

The response's usage, and so its cost, stays the sum of the passes either way.

## Code Execution Tool Version

By default, Pydantic AI chooses a compatible Anthropic code execution tool version for the selected model. You can override this with [`AnthropicModelSettings.anthropic_code_execution_tool_version`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_code_execution_tool_version] when you need a specific supported Anthropic tool version:

```py {title="anthropic_code_execution_tool_version.py"}
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[NativeTool(CodeExecutionTool())],
    model_settings=AnthropicModelSettings(anthropic_code_execution_tool_version='20260120'),
)
```

Pydantic AI raises a [`UserError`][pydantic_ai.exceptions.UserError] if you explicitly select a tool version that the model does not support.

### Code Execution Containers

When you continue a run from message history, Pydantic AI automatically reuses the Anthropic code execution container recorded in that history. Anthropic containers expire after 30 days, and a request that refers to an expired container returns an error.

If a request uploads files through [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] and Anthropic returns HTTP 500 for a history-derived container, Pydantic AI retries once without the rejected container ID so Anthropic can create a fresh container and receive the uploads. Other errors are raised without this retry. The fresh container does not contain state or files from the expired container.

Set [`anthropic_container`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_container] explicitly when container continuity is required. An explicitly configured container is never replaced automatically; Anthropic's original error is raised instead.
