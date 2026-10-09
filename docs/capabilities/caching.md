---
description: "Prompt caching in Pydantic AI: why to add the Caching capability to your agents, how each provider caches, what caching costs, what invalidates a cache, and how to check that it works."
---

# Caching

Prompt caching lets a provider reuse the work it did on the beginning of a request it has seen recently (tool definitions, instructions, and the conversation so far) instead of processing it again, which makes long and multi-turn requests cheaper and faster. Because [agent runs](../agent.md#running-agents) resend the whole conversation at every step, caching is often the difference between paying for the history once and paying for it on every turn.

A cache hit reuses an unchanged beginning of the current request, its *prefix*, that the provider cached from an earlier request. Providers read a request in a fixed order: tool definitions, then the system prompt, then messages. Appending new messages keeps the cached prefix usable, but changing anything earlier makes the cache unusable from that point onward, and everything after it is charged again.

**Add the [`Caching`][pydantic_ai.capabilities.Caching] capability to your agents.** Several providers cache nothing unless the request asks them to, so an agent without it pays full price for its instructions, tools and conversation on every request. Caching isn't on by default because it changes what requests cost: writing to the cache costs more than uncached input, so a request whose prefix is never read back costs more with caching than without. For agents that mostly make one-off requests sharing long instructions or tools, [cache only the stable prefix](#caching-only-the-stable-prefix).

## Unified caching settings

Use the [`Caching`][pydantic_ai.capabilities.Caching] capability to enable caching:

```python {title="caching_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching

agent = Agent('anthropic:claude-opus-5-5', capabilities=[Caching()])
```

`model_settings={'cache': True}` is the equivalent through the `cache` field in [`ModelSettings`][pydantic_ai.settings.ModelSettings].

What it does depends on how the provider caches:

| Without `Caching()` | Providers | With `Caching()` |
|---|---|---|
| Nothing is cached | Anthropic (including on Bedrock, Vertex AI and Microsoft Foundry), Bedrock's Claude and Nova models, OpenRouter's Anthropic models | Tool definitions, instructions and the conversation are cached |
| Cached implicitly | OpenAI GPT-5.6 and later, Gemini 2.5 and later on OpenRouter | Explicit breakpoints are added on top, such as one after the instructions so separate conversations share them |
| Cached implicitly | OpenAI models before GPT-5.6, Gemini, DeepSeek, xAI | No effect |

To confirm that requests read from the cache, see [Monitoring cache efficiency](#monitoring-cache-efficiency).

### Configuring caching

The [`Caching.retention`][pydantic_ai.capabilities.Caching.retention] value accepts:

- `True` (the capability's default) — cache with the provider's default retention
- `False` — disable library-managed caching, for example to override a `cache` value in the model's default settings
- `'5m'` / `'30m'` / `'1h'` — cache with a specific retention, snapped to the nearest tier the provider supports: down where a shorter tier exists (`'30m'` becomes `'5m'` on Anthropic, which offers 5 minutes and 1 hour), otherwise up to the shortest one (every value becomes `'30m'` on OpenAI GPT-5.6 and later)

Retention, or TTL (time to live), is how long the provider keeps a cached prefix after its last use. Start with the default. Consider Anthropic's `'1h'` when a conversation commonly continues more than five minutes after its last request: its cache writes cost more, so a prefix needs at least two reads to beat uncached input (see [Cost](#cost)).

These are the same values accepted by the underlying `cache` model setting, which is unset by default. Provider-specific cache settings, such as `anthropic_cache` or `bedrock_cache_instructions`, take precedence: when any of them is set, including to `False`, the unified setting is ignored for that request.

`cache=False` disables caching that Pydantic AI manages. [`CachePoint`][pydantic_ai.messages.CachePoint] markers you add to the message history and provider-specific cache settings still apply, and providers that cache implicitly still do.

### Caching only the stable prefix

By default, caching covers the tool definitions, the static instructions, and the conversation. When an agent handles many short, one-off conversations that share long instructions or tools, writing each conversation to the cache costs more than it saves, because it's never read back. Pass `messages=False` to cache only the stable prefix:

```python {title="caching_stable_prefix.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching

handbook = '...'  # a long document, well above the provider's minimum cacheable length

agent = Agent(
    'anthropic:claude-opus-5-5',
    instructions=f'Answer questions about this employee handbook:\n\n{handbook}',
    capabilities=[Caching(messages=False)],
)

# Two unrelated one-off questions: the second reads the handbook from the cache.
agent.run_sync('Can I expense a home office chair?')
agent.run_sync('Is remote work allowed on Fridays?')
```

On providers where `Caching()` has no effect, `messages=False` doesn't either: it doesn't turn off their implicit caching. The `cache` model setting takes the same option as a [`CacheConfig`][pydantic_ai.settings.CacheConfig], together with an optional retention:

```python {title="unified_cache_stable_prefix.py"}
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-opus-5-5',
    model_settings={'cache': {'retention': '1h', 'messages': False}},
)
```

### Provider translation

Where the provider has an automatic caching mode, the unified setting uses it. Elsewhere, Pydantic AI places cache breakpoints, markers that end the content eligible for caching, at the end of the tool definitions, the static instructions, and the conversation, so the stable prefix is shared between conversations and each request reads back everything the previous one cached. When explicit `CachePoint`s would push a request over the provider's breakpoint limit, the oldest message breakpoints are dropped first; on OpenAI, which doesn't trim the request client-side, the server instead drops the earliest breakpoints first, starting with the instruction breakpoint (see [OpenAI prompt caching](../models/openai.md#prompt-caching)).

| Provider | `Caching()` | `Caching('1h')` | `Caching(messages=False)` | Notes |
|---|---|---|---|---|
| Anthropic API and Microsoft Foundry | `anthropic_cache='5m'` | `anthropic_cache='1h'` | `anthropic_cache_instructions` and `anthropic_cache_tool_definitions` set to `'5m'` | |
| Anthropic on Bedrock and Vertex AI SDK clients | `anthropic_cache_instructions`, `anthropic_cache_tool_definitions` and `anthropic_cache_messages` set to `'5m'` | The same, set to `'1h'` | Without `anthropic_cache_messages` | On Bedrock, `'1h'` snaps to `'5m'` on the models AWS doesn't grant the 1-hour TTL |
| Bedrock (Claude and Nova) | `bedrock_cache_instructions`, `bedrock_cache_tool_definitions` and `bedrock_cache_messages` set to `True` | The same, set to `'1h'` | Without `bedrock_cache_messages` | `'1h'` snaps to `'5m'` on the models AWS doesn't grant the 1-hour TTL |
| OpenRouter (Anthropic and Gemini) | `openrouter_cache_instructions`, `openrouter_cache_tool_definitions` and `openrouter_cache_messages` set to `'5m'` | The same, set to `'1h'` | Without `openrouter_cache_messages` | Gemini routes take no tool definition breakpoint and no TTL |
| OpenAI (GPT-5.6 and later, on the OpenAI API) | `openai_prompt_cache_options={'mode': 'implicit', 'ttl': '30m'}` and `openai_cache_instructions=True` | The same | `mode='explicit'`, so only the instructions are written | `'30m'` is the only TTL OpenAI accepts. Requests that can't carry an instruction breakpoint, such as those continuing server-side state (`openai_previous_response_id` or `openai_conversation_id`), keep the implicit breakpoint with `messages=False`, so they cache like `Caching()` |
| Providers that only cache implicitly | No effect | No effect | No effect | Such as Google and OpenAI models before GPT-5.6 |

### Going beyond the unified setting

The unified setting covers the common case. Reach for the lower-level controls when it can't express what you need:

- Add a [`CachePoint`][pydantic_ai.messages.CachePoint] to a user message after reusable context you supply, such as a long document followed by a changing question, so the document is cached even though it's part of the conversation.
- Use provider-specific settings for behavior the unified setting doesn't cover, such as Anthropic's [per-block message caching](../models/anthropic.md#per-block-message-caching) for gateways that don't support automatic caching, OpenAI's [retention policy](../models/openai.md#prompt-caching) on models before GPT-5.6, or Google's [cached content resources](../models/google.md#context-caching-google_cached_content). Because a provider-specific setting replaces the unified setting for the whole request (see [Configuring caching](#configuring-caching)), set every cache setting you need on that provider.

## How providers cache

Each section below summarizes a provider's native caching behavior and its provider-specific settings. The linked provider pages document those settings in detail.

### Anthropic

Anthropic caches nothing unless the request marks what to cache, with a 5-minute default TTL and a 1-hour opt-in. A request carries at most four cache breakpoints.

[`anthropic_cache`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache] uses Anthropic's [automatic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching), which places the breakpoint on the last cacheable block and moves it forward as the conversation grows. The Anthropic API and Microsoft Foundry support automatic caching; the Bedrock and Vertex AI SDK clients don't, so there `anthropic_cache` falls back to a breakpoint on the last user message. [`anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions], [`anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions] and [`anthropic_cache_messages`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_messages] place per-block breakpoints, and a [`CachePoint`][pydantic_ai.messages.CachePoint] in a user message marks one explicitly.

See [Anthropic prompt caching](../models/anthropic.md#prompt-caching) for details.

### Bedrock

The Bedrock Converse API caches explicitly on Claude and Nova models, at a 5-minute default TTL, with a 1-hour TTL on the Claude models AWS grants it to. Nova doesn't cache tool definitions. Content below the model's minimum token threshold isn't cached, and a request carries at most four cache points.

[`bedrock_cache_instructions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_instructions], [`bedrock_cache_tool_definitions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_tool_definitions] and [`bedrock_cache_messages`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_messages] place cache points, and a [`CachePoint`][pydantic_ai.messages.CachePoint] marks one explicitly. A cache point only finds the previous request's cache entry within about 20 content blocks, so after a turn that adds more than that, `bedrock_cache_messages` also marks the end of the previous request.

See [Bedrock prompt caching](../models/bedrock.md#prompt-caching) for details.

### OpenAI

OpenAI caches prompts implicitly once they pass a minimum length (1,024 tokens on GPT-5.6 and later).

On models before GPT-5.6, how long a cached prefix lives depends on the retention policy, which [`openai_prompt_cache_retention`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_retention] overrides per request.

On GPT-5.6 and later, a cached prefix stays eligible for reuse for 30 minutes after its most recent write or read. [`openai_prompt_cache_options`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_options] sets the TTL and the mode: in the default implicit mode OpenAI also places a breakpoint of its own, in explicit mode only the request's breakpoints are cached. A [`CachePoint`][pydantic_ai.messages.CachePoint] adds a breakpoint after a user content block, and [`openai_cache_instructions`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_cache_instructions] adds one after the static instructions. OpenAI writes at most four breakpoints per request.

On any model, [`openai_prompt_cache_key`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_key] groups requests that share a prefix to improve hit rates.

See [OpenAI prompt caching](../models/openai.md#prompt-caching) for details.

### OpenRouter

OpenRouter passes explicit breakpoints through to Anthropic and Gemini models: [`openrouter_cache_instructions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_instructions], [`openrouter_cache_tool_definitions`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_tool_definitions], [`openrouter_cache_messages`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_cache_messages] and [`CachePoint`][pydantic_ai.messages.CachePoint]. Anthropic routes honor the requested TTL. Gemini routes ignore it, use only the last breakpoint in the conversation, and take no tool definition breakpoint; their cached system instructions can't change, so put dynamic content in a later user message rather than in the instructions. OpenAI models on OpenRouter cache implicitly; for GPT-5.6 breakpoints, use an OpenAI model class with the OpenRouter provider.

See [OpenRouter prompt caching](../models/openrouter.md#prompt-caching) for details.

### Google

Gemini caches prompts implicitly. [`CachePoint`][pydantic_ai.messages.CachePoint] markers are ignored. To reuse a large, fixed context explicitly, create a cached content resource and pass its name in [`google_cached_content`][pydantic_ai.models.google.GoogleModelSettings.google_cached_content]. The resource then owns the system instructions and tools, so the agent's own instructions and tools are left out of those requests.

See [Google context caching](../models/google.md#context-caching-google_cached_content) for details.

### Other providers

Other providers that cache prompts, such as DeepSeek and xAI, do so implicitly. Some take a hint that keeps related requests on the same cache: xAI's [cache sticky routing](../models/xai.md#cache-sticky-routing) metadata, Mistral's [`mistral_prompt_cache_key`][pydantic_ai.models.mistral.MistralModelSettings.mistral_prompt_cache_key], and the [prompt cache identity](../models/openai-codex.md#prompt-caching) the OpenAI Codex model derives from the conversation.

## Cost

On Anthropic and OpenAI GPT-5.6 and later, writing a prefix to the cache costs more than sending it uncached, and reading it back costs much less:

| Provider | Cache write | Cache read |
|---|---|---|
| Anthropic (incl. Bedrock, Vertex AI and Foundry) | 1.25x the input price for the 5-minute cache, 2x for the 1-hour cache | 0.1x |
| OpenAI GPT-5.6 and later | 1.25x | 0.1x |

So a 5-minute cache (1.25x) breaks even after one read and Anthropic's 1-hour cache (2x) after two. A cached prefix is read back by the next request that starts with it before the retention expires: in an agent run with tool calls that's every request after the first, and in a conversation that continues through [`message_history`](../message-history.md) it's every turn that arrives in time. A single request that's never repeated only pays the write premium.

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

To check that caching works, send a request with a long shared prefix, continue the conversation, and compare the two responses' usage:

```python {title="caching_check.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching

handbook = '...'  # a long document, well above the provider's minimum cacheable length

agent = Agent(
    'anthropic:claude-opus-5-5',
    instructions=f'Answer questions about this employee handbook:\n\n{handbook}',
    capabilities=[Caching()],
)

first = agent.run_sync('How many vacation days do new employees get?')
second = agent.run_sync('And after five years?', message_history=first.all_messages())

written = first.response.usage.cache_write_tokens  # the handbook and the first question
read = second.response.usage.cache_read_tokens  # what the first request wrote
```

The first request writes its prefix to the cache and the follow-up reads it back. A prefix below the provider's minimum cacheable length, between about 1,024 and 4,096 tokens depending on the provider and model, is never cached, so both counts stay at zero. Some providers, such as OpenAI before GPT-5.6, report only cache reads.

Every response's [`RequestUsage`][pydantic_ai.usage.RequestUsage] normalizes `cache_read_tokens` and `cache_write_tokens` across providers, and a run's [`RunUsage`][pydantic_ai.usage.RunUsage] (`result.usage`) sums them over all of the run's requests. `input_tokens` includes cache reads and writes, so `cache_read_tokens / input_tokens` (the `cache_hit_ratio` property) is a comparable hit ratio per request. Compute a run's ratio only when all its requests went to the same model: a ratio aggregated across models isn't interpretable.

When [instrumentation](../logfire.md) is enabled, model-request spans carry a per-request hit ratio and the size of the established cached prefix. A request that loses a meaningful part of the established prefix is recorded as a cache collapse with a classified reason, such as an expired provider retention window or provider-native compaction, and only an unexpected collapse emits a `pydantic_ai.cache.collapse` span event. A request long enough to cache that goes to a model that needs caching configured, with none configured, is flagged as `pydantic_ai.cache.not_enabled`. See [Prompt-cache health](../logfire.md#prompt-cache-health) for the attributes, thresholds, collapse reasons, and how the established prefix is tracked.

To get the same signals as Python warnings during development and in CI, use Pydantic AI Harness's [Warn On Cache Busts](../harness/warn-on-cache-busts.md) capability: it shares the same detector and classification, and also warns when caching isn't enabled.

These signals detect that a cache was missed, not why. When the hit rate is lower than expected:

1. Check that caching is enabled for the model, and that no provider-specific cache setting overrides the unified one.
2. Check that the shared prefix is above the provider's minimum cacheable length.
3. Compare each response's `cache_read_tokens` and `cache_write_tokens`: writes without later reads mean the prefix changed or expired before it was read.
4. Look for the [changes under your control](#changes-under-your-control) that move the prefix, and check that conversations continue with their full `message_history`.
5. Check whether the gap between requests exceeds the provider's retention, and whether requests of one conversation reach the same cache (OpenRouter's provider routing, xAI's sticky routing, OpenAI's `openai_prompt_cache_key`).
6. Ask the provider: Anthropic's [cache diagnostics](../models/anthropic.md#cache-diagnostics) and OpenAI's [prompt cache diagnostics](../models/openai.md#prompt-caching) report why a request missed.

No warning doesn't prove that caching works: a request that falls short by less than the detection thresholds, a provider that reports no cache usage, or a cache that was never established all stay silent.

## Rules for extension authors

Tool, toolset, and capability authors should preserve the same contract:

- Never mutate the cached prefix during a run.
- Put per-turn dynamic content in the user message, in tool results, or after the last [`CachePoint`][pydantic_ai.messages.CachePoint].
- Keep tool definitions and their ordering stable across steps.
- Use date-only timestamps in instructions.
