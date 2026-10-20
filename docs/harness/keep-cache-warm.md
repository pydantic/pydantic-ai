---
title: Keep Cache Warm
description: "Keep a Pydantic AI conversation's prompt cache from expiring during a long tool call or between turns, by replaying the last model request with max_tokens=1 shortly before the provider's cache retention window ends."
---

# Keep Cache Warm

Keep a conversation's prompt cache from expiring while a long tool call runs, and optionally while the conversation waits for its next turn, so the next model request reads the cache instead of paying to write the whole prefix again.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## How it works

Anthropic keeps a cached prompt prefix for 5 minutes by default, timed from the start of the last request that used it, and [restarts that timer each time the prefix is read](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). A conversation that sits idle for longer than that pays the full cache-write price (1.25x the input price) on its next request. Reading the prefix back costs 0.1x.

After each model request, `KeepCacheWarm` starts a background task for the conversation. Shortly before the cache would expire, the task sends the same request again (same messages, instructions, tools and settings) with `max_tokens=1`, and discards the response. The replay reads the cached prefix, which restarts the provider's timer. The conversation's next model request cancels the pending refresh.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching
from pydantic_ai_harness import KeepCacheWarm

# Keeps the cache warm while a run's tools take longer than the cache lasts.
agent = Agent('{{ anthropic:medium }}', capabilities=[Caching(), KeepCacheWarm()])
```

The refresh doesn't change the conversation: its response isn't added to the message history, and its usage isn't part of any run's usage. `keep_warm.usage` holds the combined usage of every refresh sent. A refresh calls the model directly, so other capabilities' model request hooks don't see it.

## Between runs

By default the cache is kept warm only while a run is in progress, where the next model request is certain to come: once a long tool call finishes, the run asks the model again. Between runs the next turn may never come, so keeping the cache warm there is opt-in with `between_runs=True`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching
from pydantic_ai_harness import KeepCacheWarm

keep_warm = KeepCacheWarm(between_runs=True)
agent = Agent('{{ anthropic:medium }}', capabilities=[Caching(), keep_warm])


async def main():
    result = await agent.run('Summarize the repository layout.')
    # The cache stays warm while the user reads the answer...
    result = await agent.run('Now list the test directories.', message_history=result.all_messages())
    # ...and on shutdown, cancel any pending refresh.
    await keep_warm.stop()
```

Each refresh costs a cache read of the whole prefix. At Anthropic's 0.1x read and 1.25x write prices, the six refreshes of a 30-minute `max_idle` cost about half of the one rewrite they avoid, so warming between runs pays off when conversations usually continue within `max_idle`. Newer Claude models read their cache for less (0.05x or 0.025x), which lowers that bar.

## Which models it warms

A request is kept warm when:

- the model's cache expires a fixed time after its last use, which each read restarts: a profile with a `'5m'` cache tier, as on Anthropic, Claude and Nova on Amazon Bedrock, and OpenRouter's Anthropic routes;
- its cache retention is known: the retention the request's settings ask for (such as `Caching(retention='1h')`), else the provider's documented default;
- the response reported cached tokens.

OpenAI isn't warmed: GPT-5.6 and later keep a cached prefix for *at least* 30 minutes, and may keep it longer, so a refresh buys nothing certain. Models that cache implicitly, or that publish no retention, aren't warmed either. Neither is a `FallbackModel`, since which provider's cache a replay would hit isn't known.

## Options

| Option | Default | Meaning |
|---|---|---|
| `between_runs` | `False` | Also keep the cache warm after a run ends, until the conversation's next run or `max_idle`. |
| `max_idle` | 30 minutes | How long after the conversation's last real request to keep its cache warm. With a 5-minute retention this is about six refreshes. A retention at least this long, such as `'1h'`, is never refreshed. |
| `margin` | 30 seconds | How long before the cache would expire to send the refresh. |

## When warming stops

- **The next real request**, which uses the cache itself.
- **`max_idle` has passed** since the conversation's last real request.
- **A late refresh.** A refresh that can't start within half of `margin` of its due time (the event loop was blocked, or the process was suspended) is dropped, since the cache may already be gone and the refresh would pay to write it again.
- **The run ends**, unless `between_runs` is set.
- **A refresh that isn't a cheap read.** Warming stops with a `CacheWarmingStoppedWarning` when a refresh request:
    - raises, as with Anthropic extended thinking, where `max_tokens` must exceed the thinking `budget_tokens`;
    - reads back less than 95% of the prefix the real request cached, so it paid to write the cache again;
    - produces more than 16 output tokens, so the provider ignored `max_tokens=1`.
- **A real request that took nearly the whole retention window** (at the defaults, a response that streamed for more than 4m45s): its refresh is already late when the response arrives, so none is sent.
- **`await keep_warm.stop()`**, which cancels every pending refresh.

## Limits

- **It needs a running event loop.** The refresh runs on the loop the run ran on, so it works in an async application. Between `run_sync()` calls the loop isn't running, so only refreshes during a run (such as during a long tool call) are sent.
- **It isn't for durable execution**, where model requests run in a separate activity or step.
- **State is in-process memory**, keyed by `conversation_id`. Reuse one instance across the runs of a conversation. Runs of one conversation are expected one at a time: two that overlap can cancel each other's refreshes.

## Telemetry

Each refresh is a root `keep_cache_warm` span, since it outlives the run that started it, with these attributes:

| Attribute | Meaning |
|---|---|
| `keep_cache_warm.cache_read_tokens` | Cached tokens the refresh read back. |
| `keep_cache_warm.cache_write_tokens` | Tokens the refresh wrote to the cache. |
| `keep_cache_warm.retention_seconds` | The retention the refresh was timed against. |

A refresh request that raises records the exception on the span.

## API reference

- [`pydantic_ai_harness.keep_cache_warm` source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/keep_cache_warm/)
- [Prompt caching](../capabilities/caching.md)
- [Warn On Cache Busts](warn-on-cache-busts.md)

::: pydantic_ai_harness.keep_cache_warm.KeepCacheWarm

::: pydantic_ai_harness.keep_cache_warm.CacheWarmingStoppedWarning
