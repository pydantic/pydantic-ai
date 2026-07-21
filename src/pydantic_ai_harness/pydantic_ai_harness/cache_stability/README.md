# Cache Stability Monitor

Warn when a run's prompt cache hit collapses between model requests.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/cache_stability/)

Prompt caching pays off only while the cacheable prefix (tools, then system
instructions, then message history) stays byte-stable across a run's consecutive
requests. When something moves that prefix -- reordered tools, a timestamp
injected into instructions, a serialization-level block hop -- the provider
re-charges tokens it could have served from cache. `CacheStabilityMonitor` makes
that collapse visible.

This is the **observe** signal: it reads the provider's own verdict rather than
guessing from the structured request. On each response it reads
`usage.cache_read_tokens` and tracks the largest cacheable prefix the run has
established (`cache_read_tokens + cache_write_tokens`, a high-water mark), keyed by
the response's `(provider_name, model_name)`. Because message history is
append-only, a stable prefix means each request for that model reads back at least
what the previous one cached; a large drop is the observable signature of a
collapse.

When a request reads back less than `collapse_ratio` of the established prefix, the
monitor warns once and latches that key, staying quiet about the collapse until a
healthy read-back re-stabilizes the cache. A sustained collapse -- caching toggled
off mid-run (`read == 0, write == 0`), or a prefix that moves every request so the
provider keeps writing a cache nothing reads back -- therefore warns once, not on
every request.

Keying per provider and model means a mid-run model switch does not warn: a
`FallbackModel` failover or a per-step model change uses a different cache key, so
the monitor starts a fresh mark for it instead of comparing against the previous
model's. Marks are kept per key rather than reset, so switching back to an earlier
model within its cache TTL still compares against that model's prefix.

A collapse has two shapes the monitor cannot tell apart, so the warning names both:
the cacheable prefix moved, or the provider's cache expired under an unchanged prefix
(a gap between requests longer than the cache TTL -- Anthropic's default is 5 minutes,
refreshed on each hit). When the gap since the same model's previous request exceeds
`cache_ttl_seconds`, the message reports the gap so a long tool or approval pause
isn't mistaken for a moved prefix. The gap is timed per model, so switching away and
back measures the returning model's own idle time, not whatever ran in between.

The verdict is cross-provider for free -- pyai normalizes every provider into the
`cache_read_tokens` / `cache_write_tokens` fields on `RequestUsage`.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.cache_stability import CacheStabilityMonitor
> ```

Cache Stability Monitor is a released, non-experimental capability. Pydantic AI
Harness is still on 0.x releases, so the API may change between minor releases. See
the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).
It is the opt-in observe arm of the broader prompt-cache-prefix-stability work.

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.cache_stability import CacheStabilityMonitor

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[CacheStabilityMonitor()])
await agent.run('...')  # a CacheBustWarning fires if a cached prefix collapses mid-run
```

The monitor is silent when caching is off or unreported (`cache_read_tokens`
stays 0), so it never fires spuriously in runs that don't use caching. That is the
honest scope of a runtime signal -- the deterministic, always-on structural catch
belongs at the wire level in tests, not here.

## Options

- `collapse_ratio` (default `0.5`): warn when a request reads back less than this
  fraction of the established prefix. Conservative by default so ordinary rounding
  or a partial miss does not fire; raise toward `1.0` to warn on smaller
  regressions. It must be greater than `0.0` -- a ratio of `0.0` could never warn,
  so it is rejected rather than treated as a silent disable switch.
- `min_prefix_tokens` (default `1024`): only judge collapse once the established
  prefix reaches this many tokens. Below a provider's minimum cacheable size
  (Anthropic's is 1024) `cache_read_tokens` is noisy or zero.
- `cache_ttl_seconds` (default `300`): the assumed provider cache TTL. Message-only
  -- when the gap since the same model's previous request exceeds it, the warning
  notes the collapse may be a cache expiry rather than a moved prefix. It does not
  change whether a warning fires. Lower it for providers with a shorter cache lifetime.

## Silencing and escalation

There is no bespoke suppression API. Use the stdlib `warnings` machinery, exactly
as you would manage any other `UserWarning`:

```python
import warnings
from pydantic_ai_harness.cache_stability import CacheBustWarning

# Silence the whole category:
warnings.filterwarnings('ignore', category=CacheBustWarning)

# Silence one intentional bust, scoped to the operation that causes it:
with warnings.catch_warnings():
    warnings.simplefilter('ignore', CacheBustWarning)
    result = agent.run_sync('...')  # e.g. a step that switches models or adds a file

# Treat every bust as an error (dev/CI enforcement):
warnings.filterwarnings('error', category=CacheBustWarning)
```

In tests, assert an intentional bust with `pytest.warns(CacheBustWarning)`, or
silence a legitimately-busting test with
`@pytest.mark.filterwarnings('ignore::pydantic_ai_harness.cache_stability.CacheBustWarning')`.

## Logfire

Logfire bridges the stdlib `logging` module, not the `warnings` module, so a
`CacheBustWarning` does not reach your traces on its own. To route busts into
Logfire, redirect Python warnings to the `logging` system once at startup:

```python
import logging

logging.captureWarnings(True)  # warnings.warn(...) -> the 'py.warnings' logger -> Logfire
```

The monitor's signal is the `CacheBustWarning`; routing it through `logging` is how
it reaches Logfire.

## Composition

- The monitor only implements `for_run` and `after_model_request`; it adds no
  tools, instructions, or model settings, so it composes with any other
  capability, toolset, or `ToolSearch` setup without interference.
- Per-run state (the per-key marks and timing) is materialized in `for_run`, so one
  `CacheStabilityMonitor` instance can be reused across many `Agent.run` calls --
  each run is judged independently.

## Scope

- **Observational only.** It reports that a cached prefix collapsed, not why -- a
  moved prefix and a provider-side cache expiry look the same from the token counts,
  so the warning names both. The structural explanation ("what moved the prefix this
  turn") is a separate job.
- **Fires only when caching is enabled and reported.** A run that never establishes
  a cache never warns.
- **A mid-run model switch does not warn.** Marks are per `(provider_name,
  model_name)`, so a `FallbackModel` failover starts a fresh mark rather than
  collapsing the previous model's.
