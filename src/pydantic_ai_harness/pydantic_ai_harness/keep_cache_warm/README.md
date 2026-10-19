# Keep Cache Warm

Keep a conversation's prompt cache from expiring while the agent waits for its next turn or for a long tool call, so the next model request reads the cache instead of paying to write the whole prefix again.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/keep_cache_warm/)

## How it works

Anthropic keeps a cached prompt prefix for 5 minutes by default, timed from the start of the last request that used it, and [restarts that timer each time the prefix is read](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). A conversation that sits idle for longer than that pays the full cache-write price (1.25x the input price) on its next request. Reading the prefix back costs 0.1x.

After each model request, `KeepCacheWarm` starts a background task for the conversation. Shortly before the cache would expire, the task sends the same request again (same messages, instructions, tools and settings) with `max_tokens=1`, and discards the response. The replay reads the cached prefix, which restarts the provider's timer. The conversation's next model request cancels the pending refresh.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching
from pydantic_ai_harness import KeepCacheWarm

keep_warm = KeepCacheWarm()
agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[Caching(), keep_warm])


async def main():
    result = await agent.run('Summarize the repository layout.')
    # The cache stays warm while the user reads the answer...
    result = await agent.run('Now list the test directories.', message_history=result.all_messages())
    # ...and on shutdown, cancel any pending refresh.
    await keep_warm.stop()
```

The refresh doesn't change the conversation: its response isn't added to the message history, and its usage isn't part of any run's usage. `keep_warm.usage` holds the combined usage of every refresh sent.

## Which models it warms

A request is kept warm when:

- the model's caching is configured on the request (its profile's `supports_cache`), as on Anthropic, Claude on Amazon Bedrock, and OpenRouter's Anthropic routes;
- its cache retention is known: the retention the request's settings ask for (such as `Caching(retention='1h')`), else the provider's documented default;
- the response reported cached tokens.

Models that cache implicitly, or that publish no retention, aren't warmed. Neither is a `FallbackModel`, since which provider's cache a replay would hit isn't known. OpenAI's GPT-5.6 and later keep a cached prefix for at least 30 minutes, which already covers the default `max_idle`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `max_idle` | 30 minutes | How long after the conversation's last real request to keep its cache warm. With a 5-minute retention this is about six refreshes. A retention at least this long, such as `'1h'`, is never refreshed. |
| `margin` | 30 seconds | How long before the cache would expire to send the refresh. |

## When warming stops

- **The next real request**, which uses the cache itself.
- **`max_idle` has passed** since the conversation's last real request.
- **A late refresh.** A refresh that can't start within half of `margin` of its due time (the event loop was blocked, or the process was suspended) is dropped, since the cache may already be gone and the refresh would pay to write it again.
- **A refresh that doesn't refresh.** A refresh request that raises, or that reads no cached tokens back, stops warming the conversation and emits a `CacheWarmingStoppedWarning`. That happens on a provider that rejects `max_tokens=1`, or with Anthropic extended thinking, where `max_tokens` must exceed the thinking `budget_tokens`.
- **A real request that took nearly the whole retention window** (at the defaults, a response that streamed for more than 4m45s): its refresh is already late when the response arrives, so none is sent.
- **`await keep_warm.stop()`**, which cancels every pending refresh.

## Limits

- **It needs a running event loop.** The refresh runs on the loop the run ran on, so it works in an async application. Between `run_sync()` calls the loop isn't running, so only refreshes during a run (such as during a long tool call) are sent.
- **It isn't for durable execution**, where model requests run in a separate activity or step.
- **State is in-process memory**, keyed by `conversation_id`. Reuse one instance across the runs of a conversation.

## Telemetry

Each refresh is a root `keep_cache_warm` span, since it outlives the run that started it, with these attributes:

| Attribute | Meaning |
|---|---|
| `keep_cache_warm.cache_read_tokens` | Cached tokens the refresh read back. |
| `keep_cache_warm.cache_write_tokens` | Tokens the refresh wrote to the cache. |
| `keep_cache_warm.retention_seconds` | The retention the refresh was timed against. |

A refresh request that raises records the exception on the span.

## API reference

- [Prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/)
- [Warn On Cache Busts](../warn_on_cache_busts/)

The public module exports `KeepCacheWarm` and `CacheWarmingStoppedWarning`.
