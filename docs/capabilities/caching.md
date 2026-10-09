---
description: "Prompt caching in Pydantic AI: how each provider caches, what caching costs, how to enable it portably with the Caching capability or the unified cache setting, what invalidates a cache, and how to monitor cache efficiency."
---

# Caching

Prompt caching lets a provider reuse the work it did on a prompt prefix it has seen recently (tool definitions, instructions, and the conversation so far) instead of processing it again, which makes long and multi-turn requests cheaper and faster. Because [agent runs](../agent.md#running-agents) resend the whole conversation at every step, caching is often the difference between paying for the history once and paying for it on every turn.

A cache hit requires the serialized request to be an exact prefix of an earlier request, in the provider's cache order: tool definitions, then the system prompt, then messages. A change early in the request silently causes everything after it to be charged again, so caching pays off only while that prefix stays stable.

Some providers cache prompts implicitly, without any configuration: OpenAI, Gemini, DeepSeek and xAI, for example. Others cache nothing unless the request says what to cache: Anthropic (including on Amazon Bedrock, Google Vertex AI and Microsoft Foundry), Amazon Bedrock's Claude and Nova models, and OpenRouter's Anthropic models. OpenAI's GPT-5.6 and later, and Gemini 2.5 and later on OpenRouter, sit in between: they cache implicitly, and also take explicit breakpoints, such as one at the end of the instructions so separate conversations share them.

To enable caching the same way on every provider, use the [`Caching`][pydantic_ai.capabilities.Caching] capability, described under [Unified caching settings](#unified-caching-settings).

## How providers cache

Each section below summarizes a provider's native caching behavior and its provider-specific settings. The linked provider pages document those settings in detail. For what the unified setting sends to each provider, see [Provider translation](#provider-translation).

### Anthropic

Anthropic caches nothing unless the request marks what to cache, with a 5-minute default TTL and a 1-hour opt-in. A request carries at most four cache breakpoints.

[`anthropic_cache`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache] uses Anthropic's [automatic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching), which places the breakpoint on the last cacheable block and moves it forward as the conversation grows. The Anthropic API and Microsoft Foundry support automatic caching; the Bedrock and Vertex AI SDK clients don't, so there `anthropic_cache` falls back to a breakpoint on the last user message. [`anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions], [`anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions] and [`anthropic_cache_messages`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_messages] place per-block breakpoints, and a [`CachePoint`][pydantic_ai.messages.CachePoint] in a user message marks one explicitly.

See [Anthropic prompt caching](../models/anthropic.md#prompt-caching) for details.

### Bedrock

The Bedrock Converse API caches explicitly on Claude and Nova models, at a 5-minute default TTL, with a 1-hour TTL on the Claude models AWS grants it to. Nova doesn't cache tool definitions. Content below the model's minimum token threshold isn't cached, a request carries at most four cache points, and a cache point only finds the previous request's cache entry within about 20 content blocks.

[`bedrock_cache_instructions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_instructions], [`bedrock_cache_tool_definitions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_tool_definitions] and [`bedrock_cache_messages`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_messages] place cache points, and a [`CachePoint`][pydantic_ai.messages.CachePoint] marks one explicitly.

See [Bedrock prompt caching](../models/bedrock.md#prompt-caching) for details.

### OpenAI

OpenAI caches prompts implicitly once they pass a minimum length (1,024 tokens on GPT-5.6 and later).

On models before GPT-5.6, how long a cached prefix lives depends on the retention policy, which [`openai_prompt_cache_retention`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_retention] overrides per request.

On GPT-5.6 and later, a cached prefix stays eligible for reuse for 30 minutes after its most recent write or read. [`openai_prompt_cache_options`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_options] sets the TTL and the mode: in the default implicit mode OpenAI also places a breakpoint of its own, in explicit mode only the request's breakpoints are cached. A [`CachePoint`][pydantic_ai.messages.CachePoint] adds a breakpoint after a user content block, and [`openai_cache_instructions`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_cache_instructions] adds one after the static instructions. OpenAI writes at most four breakpoints per request.

On any model, [`openai_prompt_cache_key`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_key] groups requests that share a prefix to improve hit rates.

See [OpenAI prompt caching](../models/openai.md#prompt-caching) for details.

### OpenRouter

OpenRouter passes explicit breakpoints through to Anthropic and Gemini models: [`openrouter_cache_instructions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_instructions], [`openrouter_cache_tool_definitions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_tool_definitions], [`openrouter_cache_messages`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_messages] and [`CachePoint`][pydantic_ai.messages.CachePoint]. Anthropic routes honor the requested TTL. Gemini routes ignore it, use only the last breakpoint in the conversation, and take no tool definition breakpoint. OpenAI models on OpenRouter cache implicitly; for GPT-5.6 breakpoints, use an OpenAI model class with the OpenRouter provider.

See [OpenRouter prompt caching](../models/openrouter.md#prompt-caching) for details.

### Google

Gemini caches prompts implicitly. [`CachePoint`][pydantic_ai.messages.CachePoint] markers are ignored. To reuse a large, fixed context explicitly, create a cached content resource and pass its name in [`google_cached_content`][pydantic_ai.models.google.GoogleModelSettings.google_cached_content].

See [Google context caching](../models/google.md#context-caching-google_cached_content) for details.

### Other providers

Other providers that cache prompts, such as DeepSeek and xAI, do so implicitly. Some take a hint that keeps related requests on the same cache: xAI's [cache sticky routing](../models/xai.md#environment-variable) metadata, Mistral's [`mistral_prompt_cache_key`][pydantic_ai.models.mistral.MistralModelSettings.mistral_prompt_cache_key], and the [prompt cache identity](../models/openai-codex.md#prompt-caching) the OpenAI Codex model derives from the conversation.

## Cost

Caching changes what a request costs. Writing a prefix to the cache costs more than sending it uncached, and reading it back costs much less:

| Provider | Cache write | Cache read |
|---|---|---|
| Anthropic (incl. Bedrock, Vertex AI and Foundry) | 1.25x the input price for the 5-minute cache, 2x for the 1-hour cache | 0.1x |
| OpenAI GPT-5.6 and later | 1.25x | 0.1x |

So a 5-minute cache (1.25x) breaks even after one read and Anthropic's 1-hour cache (2x) after two. A prefix is read back on every request after the first in an agent run with tool calls, and on every turn of a conversation that continues through [`message_history`](../message-history.md). A single request that's never repeated only pays the write premium; for agents that handle many of those, [cache only the stable prefix](#caching-only-the-stable-prefix).

## Unified caching settings

Use the [`Caching`][pydantic_ai.capabilities.Caching] capability to enable caching:

```python {title="caching_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching

agent = Agent('anthropic:claude-opus-5-5', capabilities=[Caching()])
```

You can also set the underlying `cache` field in [`ModelSettings`][pydantic_ai.settings.ModelSettings] directly:

```python {title="unified_cache.py"}
from pydantic_ai import Agent

agent = Agent('anthropic:claude-opus-5-5', model_settings={'cache': True})
```

The [`Caching.retention`][pydantic_ai.capabilities.Caching.retention] value accepts:

- `True` (the capability's default) — cache with the provider's default retention
- `False` — disable library-managed caching, for example to override a `cache` value in the model's default settings
- `'5m'` / `'30m'` / `'1h'` — cache with a specific retention, snapped to the nearest tier the provider supports: down where a shorter tier exists (`'1h'` becomes `'5m'` on a model that only offers 5 minutes), otherwise up to the shortest one (`'5m'` becomes `'30m'` on OpenAI)

These are the same values accepted by the underlying `cache` model setting, which is unset by default. Provider-specific cache settings, such as `anthropic_cache` or `bedrock_cache_instructions`, take precedence: when any of them is set, including to `False`, the unified setting is ignored for that request.

`cache=False` disables caching that Pydantic AI manages. [`CachePoint`][pydantic_ai.messages.CachePoint] markers you add to the message history and provider-specific cache settings still apply, and providers that cache implicitly still do.

### Caching only the stable prefix

By default, caching covers the tool definitions, the static instructions, and the conversation. When an agent handles many short, one-off conversations that share long instructions or tools, writing each conversation to the cache costs more than it saves, because it's never read back. Pass `messages=False` to cache only the stable prefix:

```python {title="caching_stable_prefix.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching

agent = Agent('anthropic:claude-opus-5-5', capabilities=[Caching(messages=False)])
```

The `cache` model setting takes the same option as a [`CacheConfig`][pydantic_ai.settings.CacheConfig], together with an optional retention:

```python {title="unified_cache_stable_prefix.py"}
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-opus-5-5',
    model_settings={'cache': {'retention': '1h', 'messages': False}},
)
```

### Provider translation

Where the provider has an automatic caching mode, the unified setting uses it. Elsewhere, Pydantic AI places cache breakpoints at the end of the tool definitions, the static instructions, and the conversation, so the stable prefix is shared between conversations and each request reads back everything the previous one cached. When explicit `CachePoint`s would push a request over the provider's breakpoint limit, the oldest message breakpoints are dropped first; on OpenAI, which doesn't trim the request client-side, the server instead drops the earliest breakpoints first, starting with the instruction breakpoint (see [OpenAI prompt caching](../models/openai.md#prompt-caching)).

| Provider | `Caching()` | `Caching('1h')` | `Caching(messages=False)` | Notes |
|---|---|---|---|---|
| Anthropic API and Microsoft Foundry | `anthropic_cache='5m'` | `anthropic_cache='1h'` | `anthropic_cache_instructions` and `anthropic_cache_tool_definitions` set to `'5m'` | |
| Anthropic on Bedrock and Vertex AI SDK clients | `anthropic_cache_instructions`, `anthropic_cache_tool_definitions` and `anthropic_cache_messages` set to `'5m'` | The same, set to `'1h'` | Without `anthropic_cache_messages` | On Bedrock, `'1h'` snaps to `'5m'` on the models AWS doesn't grant the 1-hour TTL |
| Bedrock (Claude and Nova) | `bedrock_cache_instructions`, `bedrock_cache_tool_definitions` and `bedrock_cache_messages` set to `True` | The same, set to `'1h'` | Without `bedrock_cache_messages` | `'1h'` snaps to `'5m'` on the models AWS doesn't grant the 1-hour TTL. After a turn that adds more than about 20 content blocks, the end of the previous request gets a cache point too |
| OpenRouter (Anthropic and Gemini) | `openrouter_cache_instructions`, `openrouter_cache_tool_definitions` and `openrouter_cache_messages` set to `'5m'` | The same, set to `'1h'` | Without `openrouter_cache_messages` | Gemini routes take no tool definition breakpoint and no TTL |
| OpenAI (GPT-5.6 and later, on the OpenAI API) | `openai_prompt_cache_options={'mode': 'implicit', 'ttl': '30m'}` and `openai_cache_instructions=True` | The same | `mode='explicit'`, so only the instructions are written | `'30m'` is the only TTL OpenAI accepts. Requests that can't carry an instruction breakpoint, such as those continuing server-side state (`openai_previous_response_id` or `openai_conversation_id`), keep the implicit breakpoint with `messages=False`, so they cache like `Caching()` |
| Providers that only cache implicitly | No effect | No effect | No effect | Such as Google and OpenAI models before GPT-5.6 |

## What invalidates a cache

### Changes under your control

- Minute-precision timestamps or other per-request values in instructions or system prompts. Prefer date-only granularity when the exact time is not required.
- Reordering or changing tool definitions between steps.
- History processors that rewrite already-sent messages on every request.
- Switching models or providers during a conversation. Each provider maintains a separate cache.

On Anthropic models that bind thinking blocks to the prefix that produced them (Claude Fable 5.1, Claude Opus 5.5 and Claude Sonnet 5.5), the same changes mid-conversation cause a 400 error, not just a cache miss: see [Thinking block binding](../models/anthropic.md#thinking-block-binding).

### Provider retention

Provider caches expire after idle gaps. This is unavoidable, but it creates a useful opportunity: schedule history-mutating maintenance for [cache-cold windows](../message-history.md#scheduling-maintenance-into-cache-cold-windows), when the next request would pay the full input price anyway. Each provider's documented default retention is recorded as [`ModelProfile.default_cache_retention`][pydantic_ai.profiles.ModelProfile.default_cache_retention], a longer retention requested through settings is resolved by [`Model.resolve_cache_retention()`][pydantic_ai.models.Model.resolve_cache_retention], and [`prompt_cache_outlook()`][pydantic_ai.profiles.prompt_cache_outlook] takes either to predict whether the next request will find the cache cold.

## Prefix-stability guarantees

Pydantic AI makes the following guarantees about the prompt prefix it sends to providers:

- [Instructions are assembled deterministically](../agent.md#instructions) for each request. Static instructions from `Agent(instructions=...)` always sort before dynamic instructions, preserving the static prefix when dynamic content changes.
- Message history is append-only within a run: Pydantic AI does not rewrite or reorder settled messages. [History processors](../message-history.md#processing-message-history) you add, and capabilities that rewrite history such as [compaction](compaction.md), are the exception.
- Internal bookkeeping such as run IDs, message timestamps, and deferred-tool flags does not reach the provider wire and therefore cannot move the prompt prefix.
- [Vercel AI](../ui/vercel-ai.md) and [AG-UI](../ui/ag-ui.md) adapter round-trips are tested to reconstruct histories that serialize back to the same provider request for the wire-relevant fields (tool call arguments, thinking signatures). Older UI protocol versions can be lossy — for example, AG-UI versions without a reasoning carrier drop thinking parts, which moves the prefix — so keep the client packages current.
- Every recorded provider conversation in Pydantic AI's test suite is checked for wire-level prefix stability, so a framework change that starts moving prefixes fails CI in the pull request that introduces it.

## Monitoring cache efficiency

Every response's [`RequestUsage`][pydantic_ai.usage.RequestUsage] normalizes `cache_read_tokens` and `cache_write_tokens` across providers, and a run's [`RunUsage`][pydantic_ai.usage.RunUsage] sums them. `input_tokens` includes cache reads and writes, so `cache_read_tokens / input_tokens` is a comparable hit ratio per request. Compute a run's ratio only when all its requests went to the same model: a ratio aggregated across models isn't interpretable.

When [instrumentation](../logfire.md) is enabled, model-request spans carry a per-request hit ratio and the size of the established cached prefix. A request that loses a meaningful part of the established prefix is recorded as a cache collapse with a classified reason, such as an expired provider retention window or provider-native compaction, and only an unexpected collapse emits a `pydantic_ai.cache.collapse` span event. A request long enough to cache that goes to a model that needs caching configured, with none configured, is flagged as `pydantic_ai.cache.not_enabled`. See [Prompt-cache health](../logfire.md#prompt-cache-health) for the attributes, thresholds, collapse reasons, and how the established prefix is tracked.

To get the same signals as Python warnings during development and in CI, use Pydantic AI Harness's [Warn On Cache Busts](../harness/warn-on-cache-busts.md) capability: it shares the same detector and classification, and also warns when caching isn't enabled.

## Rules for extension authors

Tool, toolset, and capability authors should preserve the same contract:

- Never mutate the cached prefix during a run.
- Put per-turn dynamic content in the user message, in tool results, or after the last [`CachePoint`][pydantic_ai.messages.CachePoint].
- Keep tool definitions and their ordering stable across steps.
- Use date-only timestamps in instructions.
