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

agent = Agent('anthropic:claude-sonnet-4-6')
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

!!! note "Claude Opus 4.7 migration"
    Anthropic's [Claude Opus 4.7 migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide) recommends removing `temperature`, `top_p`, and `top_k` from Opus 4.7 requests. Pydantic AI drops those keys automatically for `claude-opus-4-7`, including `extra_body` overrides.

    The same guide also recommends re-evaluating `max_tokens` and any token-count assumptions when migrating from Opus 4.6, since Opus 4.7 uses updated tokenization. If you rely on `count_tokens()` or `count_tokens_before_request`, verify your thresholds against the new model.

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

## Model settings

You can customize model behavior using [`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-sonnet-4-5')
settings = AnthropicModelSettings(
    temperature=0.2,
    service_tier='auto',
)
agent = Agent(model, model_settings=settings)
...
```

### Service tier

Anthropic supports controlling the [service tier](https://docs.anthropic.com/en/docs/build-with-claude/latency-and-throughput) to manage latency and throughput.
You can use the unified [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] field or the provider-specific [`anthropic_service_tier`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_service_tier] field. `anthropic_service_tier` takes precedence over the unified field when both are set, and accepts Anthropic's native values (`'auto'` or `'standard_only'`).

The unified field maps as follows for Anthropic:

- `'auto'`: passed through as `'auto'` (Anthropic's native value — uses priority capacity when available).
- `'default'`: maps to `'standard_only'` (forces the standard tier, opting out of priority capacity).
- `'flex'` and `'priority'` are not part of Anthropic's tier model and are silently ignored.

## Cloud Platform Integrations

You can use Anthropic models through cloud platforms by passing a custom client to [`AnthropicProvider`][pydantic_ai.providers.anthropic.AnthropicProvider].

### AWS Bedrock

To use Claude models via [AWS Bedrock](https://aws.amazon.com/bedrock/claude/), follow the [Anthropic documentation](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock) on how to set up an `AsyncAnthropicBedrock` client and then pass it to `AnthropicProvider`:

```python {test="skip"}
from anthropic import AsyncAnthropicBedrock

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

bedrock_client = AsyncAnthropicBedrock()  # Uses AWS credentials from environment
provider = AnthropicProvider(anthropic_client=bedrock_client)
model = AnthropicModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider)
agent = Agent(model)
...
```

!!! note "Bedrock vs BedrockConverseModel"
    This approach uses Anthropic's SDK with AWS Bedrock credentials. For an alternative using AWS SDK (boto3) directly, see [`BedrockConverseModel`](bedrock.md).

### Google Vertex AI

To use Claude models via [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude), follow the [Anthropic documentation](https://docs.anthropic.com/en/api/claude-on-vertex-ai) on how to set up an `AsyncAnthropicVertex` client and then pass it to `AnthropicProvider`:

```python {test="skip"}
from anthropic import AsyncAnthropicVertex

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

vertex_client = AsyncAnthropicVertex(region='us-east5', project_id='your-project-id')
provider = AnthropicProvider(anthropic_client=vertex_client)
model = AnthropicModel('claude-sonnet-4-5', provider=provider)
agent = Agent(model)
...
```

!!! note "Vertex vs GoogleModel"
    This approach uses Anthropic's SDK with Vertex AI credentials. For an alternative using Google's Vertex AI SDK directly, see [`GoogleModel`](google.md).

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
model = AnthropicModel('claude-sonnet-4-5', provider=provider)
agent = Agent(model)
...
```

See [Anthropic's Microsoft Foundry documentation](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry) for setup instructions including Entra ID authentication.

## Task Budgets (Beta)

Anthropic's [task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets) let you give Claude an advisory token budget for a full agentic loop — including thinking, tool calls, tool results, and output — so the model can pace itself and finish gracefully as the budget is consumed. Configure them with [`AnthropicModelSettings.anthropic_task_budget`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_task_budget], which takes an [`AnthropicTaskBudget`][pydantic_ai.models.anthropic.AnthropicTaskBudget] payload and maps to `output_config.task_budget`.

Pydantic AI automatically enables Anthropic's required `task-budgets-2026-03-13` beta when this setting is present. Support is currently limited to native Anthropic `claude-opus-4-7` requests, not Bedrock, Vertex, or Microsoft Foundry Anthropic model IDs.

```python {title="anthropic_task_budget.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-opus-4-7')
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

Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to reduce costs by caching parts of your prompts. Pydantic AI supports automatic caching, per-block message caching, and explicit cache breakpoints:

### Automatic Caching

The simplest way to enable prompt caching is with [`AnthropicModelSettings.anthropic_cache`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache]. This uses Anthropic's [automatic caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching), passing a top-level `cache_control` parameter so the server automatically applies a cache breakpoint to the last cacheable block in each request:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='You are a helpful assistant.',
    model_settings=AnthropicModelSettings(
        anthropic_cache=True,
    ),
)

result1 = agent.run_sync('What is the capital of France?')

result2 = agent.run_sync(
    'What is the capital of Germany?', message_history=result1.all_messages()
)
print(f'Cache write: {result1.usage().cache_write_tokens}')
print(f'Cache read: {result2.usage().cache_read_tokens}')
```

This is ideal for multi-turn conversations where the cache breakpoint should move forward as the conversation grows. You can also specify a custom TTL with `anthropic_cache='1h'`.

!!! note "Bedrock and Vertex"
    Bedrock and Vertex [do not yet support automatic caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching). On these platforms, `anthropic_cache` falls back to per-block caching on the last user message, providing the same benefit for multi-turn conversations.

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
    'claude-sonnet-4-6',
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

#### Example: Comprehensive Caching Strategy

Combine automatic caching with explicit breakpoints for maximum savings. Automatic caching handles the conversation, while explicit breakpoints pin system instructions and tool definitions:

```python {test="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-6',
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

When you use `anthropic_cache_instructions` with both static and dynamic [instructions](../agent.md#instructions), Pydantic AI automatically places the cache boundary at the optimal point. Static instructions (from `Agent(instructions=...)`) are sorted before dynamic instructions (from `@agent.instructions` functions or [toolsets](../toolsets.md)), and the cache point is placed after the last static instruction block.

This means your stable, static instructions are cached efficiently, while dynamic instructions (which may change between requests) remain outside the cache boundary and don't cause cache invalidation.

```python {test="skip"}
from datetime import date

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-6',
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
    'anthropic:claude-sonnet-4-6',
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

Access cache usage statistics via `result.usage()`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache=True,
    ),
)

result = agent.run_sync('Your question')
usage = result.usage()
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

### Cache Point Limits

Anthropic enforces a maximum of 4 cache points per request. Pydantic AI automatically manages this limit to ensure your requests always comply without errors.

#### How Cache Points Are Allocated

Cache points can come from several sources:

1. **Automatic caching**: Via `anthropic_cache` (the server applies 1 cache point to the last cacheable block)
2. **Final message block**: Via `anthropic_cache_messages` setting (adds cache point to last message content block)
3. **System Prompt**: Via `anthropic_cache_instructions` setting (adds cache point to last system prompt block)
4. **Tool Definitions**: Via `anthropic_cache_tool_definitions` setting (adds cache point to last tool definition)
5. **Messages**: Via `CachePoint` markers (adds cache points to message content)

Each setting uses **at most 1 cache point**, but you can combine them — except `anthropic_cache` and `anthropic_cache_messages`, which are mutually exclusive. If the total exceeds 4, Pydantic AI automatically trims excess cache points from older messages.

#### Example: Combining Automatic and Explicit Caching

Define an agent with automatic caching plus explicit breakpoints:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-6',
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
usage = result.usage()
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
    'anthropic:claude-sonnet-4-6',
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
usage = result.usage()
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

**Key Points**:
- System and tool cache points are **always preserved**
- `anthropic_cache` counts as 1 cache point, just like `anthropic_cache_instructions` and `anthropic_cache_tool_definitions`
- Excess `CachePoint` markers in messages are removed from oldest to newest when the limit is exceeded
- This ensures critical caching (instructions/tools) is maintained while still benefiting from message-level caching

## Fast mode

Fast mode provides higher output tokens per second and is currently supported on **Claude Opus 4.6** (`anthropic:claude-opus-4-6`) only. It is a research preview. Set [`anthropic_speed`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_speed] to `'fast'` to enable it; Pydantic AI automatically adds the required `fast-mode-2026-02-01` beta. On unsupported models, `anthropic_speed='fast'` is ignored with a `UserWarning`. For pricing, rate limits, and the latest list of supported models, see the [Anthropic fast mode docs](https://platform.claude.com/docs/en/build-with-claude/fast-mode).

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-opus-4-6',
    model_settings=AnthropicModelSettings(anthropic_speed='fast'),
)
...
```

!!! note "Prompt cache interaction"
    Switching between `'fast'` and `'standard'` invalidates the prompt cache. Requests at different speeds do not share cached prefixes, so pick one speed per cache-sensitive conversation.

!!! note "Bedrock, Vertex, and Foundry"
    Fast mode is only available on the direct Anthropic API. Bedrock, Vertex, and Foundry clients do not support the `speed` parameter, so `anthropic_speed='fast'` is ignored with a `UserWarning` on those clients.

## Message Compaction

Anthropic supports [automatic context compaction](https://docs.anthropic.com/en/docs/build-with-claude/compaction) to manage long conversations. When input tokens exceed a configured threshold, the API automatically generates a summary that replaces older messages while preserving context.

The easiest way to enable compaction is with the [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] capability:

```python {title="anthropic_compaction.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicCompaction

agent = Agent(
    'anthropic:claude-sonnet-4-6',
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

agent = Agent('anthropic:claude-sonnet-4-6')
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

## Code Execution Tool Version

By default, Pydantic AI chooses a compatible Anthropic code execution tool version for the selected model. You can override this with [`AnthropicModelSettings.anthropic_code_execution_tool_version`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_code_execution_tool_version] when you need a specific supported Anthropic tool version:

```py {title="anthropic_code_execution_tool_version.py"}
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    builtin_tools=[CodeExecutionTool()],
    model_settings=AnthropicModelSettings(anthropic_code_execution_tool_version='20260120'),
)
```

Pydantic AI raises a [`UserError`][pydantic_ai.exceptions.UserError] if you explicitly select a tool version that the model does not support.
