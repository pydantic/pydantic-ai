---
description: "Prompt caching in Pydantic AI: enable it across Anthropic, Bedrock, OpenRouter and OpenAI GPT-5.6 with the Caching capability or the unified cache setting, and what it costs."
---

# Prompt Caching

Prompt caching lets a provider reuse the work it did on a prompt prefix it has seen recently (tool definitions, instructions, and the conversation so far) instead of processing it again, which makes long and multi-turn requests cheaper and faster.

Some providers cache prompts implicitly, without any configuration: OpenAI, Gemini, DeepSeek and xAI, for example. Others cache nothing unless the request opts in: Anthropic (including on Amazon Bedrock, Google Vertex AI and Microsoft Foundry), Amazon Bedrock's Claude and Nova models, and OpenRouter's Anthropic models. OpenAI's GPT-5.6 and later, and Gemini 2.5 and later on OpenRouter, sit in between: they cache implicitly, and also take explicit breakpoints (and on GPT-5.6, a cache TTL), such as one at the end of the instructions so separate conversations share them. On these models, enable caching with the [`Caching`][pydantic_ai.capabilities.Caching] capability:

```python {title="caching_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching

agent = Agent('anthropic:claude-opus-4-7', capabilities=[Caching()])
```

You can also set the underlying `cache` field in [`ModelSettings`][pydantic_ai.settings.ModelSettings] directly:

```python {title="unified_cache.py"}
from pydantic_ai import Agent

agent = Agent('anthropic:claude-opus-4-7', model_settings={'cache': True})
```

Caching changes what a request costs. Writing a prefix to the cache costs more than sending it uncached, and reading it back costs much less:

| Provider | Cache write | Cache read |
|---|---|---|
| Anthropic (incl. Bedrock, Vertex AI and Foundry) | 1.25x the input price for the 5-minute cache, 2x for the 1-hour cache | 0.1x |
| OpenAI GPT-5.6 and later | 1.25x | 0.1x |

So a 5-minute cache (1.25x) breaks even after one read and Anthropic's 1-hour cache (2x) after two. A prefix is read back on every request after the first in an agent run with tool calls, and on every turn of a conversation that continues through [`message_history`](../message-history.md). A single request that's never repeated only pays the write premium.

## Configuring caching

The [`Caching.retention`][pydantic_ai.capabilities.Caching.retention] value accepts:

- `True` (the capability's default) — cache with the provider's default retention
- `False` — disable library-managed caching, for example to override a `cache` value in the model's default settings
- `'5m'` / `'30m'` / `'1h'` — cache with a specific retention, snapped to the nearest tier the provider supports: down where a shorter tier exists (`'1h'` becomes `'5m'` on a model that only offers 5 minutes), otherwise up to the shortest one (`'5m'` becomes `'30m'` on OpenAI)

These are the same values accepted by the underlying `cache` model setting, which is unset by default. Provider-specific cache settings, such as `anthropic_cache` or `bedrock_cache_instructions`, take precedence: when any of them is set, including to `False`, the unified setting is ignored for that request.

`cache=False` disables caching that Pydantic AI manages. [`CachePoint`][pydantic_ai.messages.CachePoint] markers you add to the message history and provider-specific cache settings still apply, and providers that cache implicitly still do.

## What gets cached

Where the provider has a server-managed caching mode, it's used: Anthropic's [automatic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching) places the cache breakpoint on the last cacheable block itself and moves it forward as the conversation grows. Elsewhere, Pydantic AI places explicit cache breakpoints at the end of the tool definitions, the static instructions, and the conversation, so the stable prefix is shared between conversations and each request reads back everything the previous one cached.

On Amazon Bedrock, a breakpoint finds the previous request's cache entry only if it is within about 20 content blocks, so after a turn that adds more than that (such as a dozen parallel tool calls and their results), the end of the previous request gets a breakpoint of its own. With instructions, tools and the conversation, that's at most the four breakpoints a request can carry; when explicit `CachePoint`s would push a request over the limit, the oldest message breakpoints are dropped first.

### Provider translation

| Provider | `Caching()` | `Caching('1h')` | Notes |
|---|---|---|---|
| Anthropic | `anthropic_cache='5m'` | `anthropic_cache='1h'` | Automatic caching on the Anthropic API and Microsoft Foundry |
| Anthropic on Bedrock and Vertex AI SDK clients | `anthropic_cache_instructions`, `anthropic_cache_tool_definitions` and `anthropic_cache_messages` set to `'5m'` | The same, set to `'1h'` | These clients don't support automatic caching. On Bedrock, `'1h'` snaps to `'5m'` on the models AWS doesn't grant the 1-hour TTL |
| Bedrock (Claude and Nova) | `bedrock_cache_instructions`, `bedrock_cache_tool_definitions` and `bedrock_cache_messages` set to `True` | The same, set to `'1h'` | Nova caches the instructions and conversation but not tool definitions, at a 5-minute TTL |
| OpenRouter (Anthropic and Gemini) | `openrouter_cache_instructions`, `openrouter_cache_tool_definitions` and `openrouter_cache_messages` set to `'5m'` | The same, set to `'1h'` | Gemini takes no TTL and caches at its default 5 minutes |
| OpenAI (GPT-5.6 and later) | `openai_prompt_cache_options={'mode': 'implicit', 'ttl': '30m'}` and `openai_cache_instructions=True` | The same | `'30m'` is the only TTL OpenAI accepts. The [instruction breakpoint](../models/openai.md#prompt-caching) is skipped on requests that continue server-side state |

## Checking that caching works

Cache usage is reported as [`cache_write_tokens`][pydantic_ai.usage.RunUsage.cache_write_tokens] and [`cache_read_tokens`][pydantic_ai.usage.RunUsage.cache_read_tokens] on the run's usage, and [instrumented](instrumentation.md) runs record [prompt-cache health](../logfire.md#prompt-cache-health) on each model request, including when a request long enough to cache was sent to a model that needs caching configured, with none configured. To surface cache problems as Python warnings during development and in CI, use Pydantic AI Harness's [Warn On Cache Busts](../harness/warn-on-cache-busts.md) capability.
