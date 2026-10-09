---
title: Warn On Cache Busts
description: "Catch prompt cache busts in Pydantic AI: warn when the cached prefix read back collapses between model requests or turns, unless the provider's cache retention window explains it."
---

# Warn On Cache Busts

Warn when a conversation's prompt cache hit collapses between model requests, within a run or across the runs that continue it, so a moved cacheable prefix surfaces instead of quietly re-charging tokens the provider could have served from cache.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## What it watches

Prompt caching pays off only while the cacheable prefix (tools, then system instructions, then message history) stays byte-stable across a run's consecutive requests. When something moves that prefix -- reordered tools, a timestamp injected into instructions, a serialization-level block hop -- the provider re-charges tokens it could have served from cache.

This is the **observe** signal: it reads the provider's own verdict rather than guessing from the structured request. On each response it reads `usage.cache_read_tokens` and tracks the cacheable prefix the conversation has established (`cache_read_tokens + cache_write_tokens`; it grows with the prefix and re-baselines after a collapse), keyed by the response's `(provider_name, provider_url, model_name)`. Because message history is append-only, a stable prefix means each request for that model reads back at least what the previous one cached, so any real shortfall is the observable signature of a collapse -- including a partial one, where a change deep in the history moves only the tail of the prefix.

When a request falls short of the established prefix by more than `min_missed_ratio` of it (5%) and by at least `min_missed_tokens` (2,000), the thresholds Claude Code uses for a prompt-cache miss, the monitor classifies the collapse against the provider's cache retention window (see below) and emits a `CacheBustWarning` unless the window explains it. The mark then re-baselines to what the collapsing request established, so an intentional bust is judged once rather than against a stale high-water mark, and the monitor stays quiet about the collapse until a healthy read-back re-stabilizes the cache. A prefix that moves on every request, so the provider keeps writing a cache nothing reads back, therefore warns once, not on every request.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import WarnOnCacheBusts

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[WarnOnCacheBusts()])
result = await agent.run('...')  # a CacheBustWarning fires if a cached prefix collapses mid-run
# ...and on the next turn, if the prefix the first turn cached no longer reads back:
await agent.run('...', message_history=result.all_messages())
```

The verdict is cross-provider for free -- pyai normalizes every provider into the `cache_read_tokens` / `cache_write_tokens` fields on `RequestUsage`.

## Retention and classification

A collapse has two shapes the token counts can't tell apart: the cacheable prefix moved, or the provider's cache expired under an unchanged prefix. The provider's retention window separates them. It is the retention the request's settings ask for, such as `anthropic_cache='1h'`, as resolved by `Model.resolve_cache_retention()`, or else the provider's documented `ModelProfile.default_cache_retention`, extended by any `CachePoint` TTLs. Each collapse is classified against it, timed from the same model's previous request:

| `reason` | Meaning | Warns |
|----------|---------|-------|
| `unexpected` | The retention window should still have been active, so the prefix moved. | Yes |
| `unknown` | The provider publishes no retention window, so a cache expiry can't be ruled out. | Yes |
| `ttl_expired` | The gap since the same model's previous request exceeded the retention window. | No |
| `compacted` | Provider-native compaction replaced the history before a `CompactionPart` with its summary, which shrinks the prefix by design. | No |
| `unreported` | The response reported no cache usage at all: caching was off for that request, or (on providers that only report reads) the cache fully missed. | No |

The warning carries the classification as `CacheBustWarning.reason`, alongside `established_tokens`, `cache_read_tokens`, and `missed_tokens`, and its message says how long ago the previous request for that model was.

This is the detector Pydantic AI's instrumentation uses for its [prompt-cache health](../logfire.md#prompt-cache-health) span attributes, so the two classify every collapse the same way. Instrumentation's `pydantic_ai.cache.collapse` span event fires only for `unexpected` collapses; this warning also fires for `unknown` ones, so it still catches a moved prefix on providers that publish no retention window (Google and DeepSeek, for example) and under a `FallbackModel`, whose serving model's profile isn't known when the response arrives. Instrumentation and this capability each keep their own marks, so with both on one agent a collapse is reported through both, and neither misses one because the other already updated a shared mark.

## Model switches

Keying per provider, endpoint, and model means a mid-run model switch does not warn: a `FallbackModel` failover or a per-step model change uses a different cache key, so the monitor starts a fresh mark for it instead of comparing against the previous model's. Marks are kept per key rather than reset, so switching back to an earlier model still compares against that model's prefix, and its retention window is timed from its own previous request, not whatever ran in between.

## Native tool calls

A response that ran a native tool, such as web search or code execution, may report cache reads summed over the provider's internal model calls, so the total can be several times the prefix the next request reads back. Such a response is still judged, so a low total can warn, but it doesn't raise the established prefix or confirm that a collapsed cache recovered; the monitor keeps the earlier prefix and waits for an ordinary request. A response that reports a single call to the main model and no compaction, or Gemini's separate tool-use prompt count, is ordinary cache accounting and updates the prefix as usual. Instrumentation's prompt-cache health attributes treat these responses the same way.

## Conversations

Marks are kept per conversation (`RunContext.conversation_id`), not per run. A run that continues an earlier one via `message_history` -- including history that was serialized and loaded back, which carries the conversation id with it -- is judged against the prefix the earlier run established, so the first request of the next turn is checked against what the previous turn cached. That is where a moved prefix most often hides: history rewritten between turns, or a tool or instruction that differs from one turn to the next. A run that starts a new conversation (no history, or `conversation_id='new'`) starts from a clean mark. The warning says whether the mark it compared against came from this run or from an earlier run of the conversation.

A continuation that comes back after the retention window has elapsed is classified `ttl_expired` and doesn't warn. A conversation's marks are forgotten once it has been idle for 24 hours, longer than any provider documents keeping a cache, or when more than 4,096 conversations on the same instance have been active more recently.

## Caching not enabled

Models such as Anthropic's (including on the Bedrock and Vertex AI SDK clients), Bedrock's Claude and Nova, and OpenRouter's Anthropic routes only cache what the request asks them to. OpenAI's models and OpenRouter's Gemini routes, which cache implicitly, are never reported. When a request of at least 4,096 input tokens goes to such a model with no caching configured (neither the unified `cache` setting nor a provider-specific one), no `CachePoint` in its history, and no cache usage reported, the monitor emits a `CacheNotEnabledWarning` pointing to `cache=True` and the `Caching()` capability, once per conversation and model. Setting `cache=False` says caching was left off on purpose, so it doesn't warn. Silence or escalate it with the `warnings` filters, like `CacheBustWarning`.

## Options

- `min_missed_ratio` (default `0.05`): only warn when a request falls short of the established prefix by more than this fraction of it. The default only keeps provider rounding out. It must be at least `0.0` and less than `1.0` -- a request can't miss more than the whole prefix.
- `min_missed_tokens` (default `2000`): only warn when a request falls short of the established prefix by at least this many tokens, which keeps small partial misses on small prefixes out. Lower both to catch smaller regressions, such as in a test with a small prompt.
- `collapse_ratio`: deprecated, with a `HarnessDeprecationWarning`. It warned when a request read back less than this fraction of the established prefix, and is still honored as `min_missed_ratio=1 - collapse_ratio`; pass that instead.
- `min_prefix_tokens`: deprecated, with a `HarnessDeprecationWarning`, and still honored as a floor on the established prefix. `min_missed_tokens` implies a prefix at least that large; pass it instead.
- `cache_ttl_seconds`: deprecated and ignored, with a `HarnessDeprecationWarning`. The retention window now comes from the model, as described above; remove the argument. To classify against a longer window, request it through the model's settings (such as `anthropic_cache='1h'`) or a `CachePoint(ttl='1h')`.

## Silencing and escalation

There is no bespoke suppression API. Use the stdlib `warnings` machinery, exactly as you would manage any other `UserWarning`:

```python
import warnings
from pydantic_ai_harness.warn_on_cache_busts import CacheBustWarning

# Silence the whole category:
warnings.filterwarnings('ignore', category=CacheBustWarning)

# Silence one intentional bust, scoped to the operation that causes it:
with warnings.catch_warnings():
    warnings.simplefilter('ignore', CacheBustWarning)
    result = agent.run_sync('...')  # e.g. a step that switches models or adds a file

# Treat every bust as an error (dev/CI enforcement):
warnings.filterwarnings('error', category=CacheBustWarning)
```

In tests, assert an intentional bust with `pytest.warns(CacheBustWarning)`, or silence a legitimately-busting test with `@pytest.mark.filterwarnings('ignore::pydantic_ai_harness.warn_on_cache_busts.CacheBustWarning')`.

## Logfire

Logfire bridges the stdlib `logging` module, not the `warnings` module, so a `CacheBustWarning` does not reach your traces on its own. To route busts into Logfire, redirect Python warnings to the `logging` system once at startup:

```python
import logging

logging.captureWarnings(True)  # warnings.warn(...) -> the 'py.warnings' logger -> Logfire
```

The monitor's signal is the `CacheBustWarning`; routing it through `logging` is how it reaches Logfire.

## Composition

- The monitor only implements `for_run` and `after_model_request`; it adds no tools, instructions, or model settings, so it composes with any other capability, toolset, or `ToolSearch` setup without interference.
- The marks live on the `WarnOnCacheBusts` instance the agent was built with, keyed by conversation; `for_run` binds each run to its conversation's marks. Reuse one instance across many `Agent.run` calls: runs of the same conversation share a mark, runs of different conversations are judged apart.
- With Pydantic AI's instrumentation enabled on the same agent, each keeps its own marks and both report the same collapses: instrumentation as span attributes (and, for `unexpected` collapses, a span event), this capability as a warning.

## Scope

- **Observational only.** It reports that a cached prefix collapsed and whether the provider's retention window explains it, not what moved the prefix. The structural explanation ("what moved the prefix this turn") is a separate job.
- **Fires only when caching is enabled and reported.** A run that never establishes a cache never warns.
- **History rewritten by a history processor can warn.** Harness compaction strategies (such as `ClearToolResults` or `SummarizingCompaction`) rewrite messages the provider already cached, which moves the prefix like any other rewrite; only provider-native compaction, recognized by its `CompactionPart`, is classified `compacted`. Silence the warning around the runs where compaction is expected, as shown above.
- **A mid-run model switch does not warn.** Marks are per `(provider_name, provider_url, model_name)`, so a `FallbackModel` failover starts a fresh mark rather than collapsing the previous model's.
- **Marks are in-process memory.** They are held on the capability instance, so a conversation continued through a different instance, a different process, or a different worker starts from a clean mark; nothing is persisted or shared.

## API reference

- [`pydantic_ai_harness.warn_on_cache_busts` source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/warn_on_cache_busts/)
- [Pydantic AI capabilities](../capabilities/overview.md)
- [Pydantic AI hooks](../hooks.md)

The public module exports `WarnOnCacheBusts` and `CacheBustWarning`. Import them from `pydantic_ai_harness.warn_on_cache_busts`.

::: pydantic_ai_harness.warn_on_cache_busts.WarnOnCacheBusts

::: pydantic_ai_harness.warn_on_cache_busts.CacheBustWarning
