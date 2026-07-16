# Prompt Caching

Provider prompt caching reuses a processed prompt prefix and charges a steeply discounted rate for tokens read from the cache, often around 10% of the normal input-token price. A cache hit requires the serialized request to be an exact byte prefix of an earlier request in the provider's cache order: tool definitions, then the system prompt, then messages. Moving one byte early in the request silently causes everything after it to be charged again. Because [agent runs](agent.md#running-agents) resend the whole conversation at every step, caching is often the difference between paying for the history once and paying for it on every turn.

## Enabling prompt caching

| Provider | Behavior | Configuration |
| --- | --- | --- |
| [OpenAI](models/openai.md#prompt-caching) | Implicit for prompts of at least 1,024 tokens; retention depends on the account's policy | Nothing to enable; [`openai_prompt_cache_retention`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_prompt_cache_retention] selects the retention policy |
| [Anthropic](models/anthropic.md#prompt-caching) | Explicit, with a 5-minute default TTL and a 1-hour opt-in | Cache settings or [`CachePoint`][pydantic_ai.messages.CachePoint] |
| [Bedrock](models/bedrock.md#prompt-caching) | Explicit; minimum-token thresholds apply | [`CachePoint`][pydantic_ai.messages.CachePoint] and cache settings |
| [OpenRouter](models/openrouter.md#prompt-caching) | Passes explicit caching through to Anthropic and Gemini models | Cache settings or [`CachePoint`][pydantic_ai.messages.CachePoint] |
| [Google](models/google.md#context-caching-google_cached_content) | Implicit caching is automatic; no [`CachePoint`][pydantic_ai.messages.CachePoint] support | [`google_cached_content`][pydantic_ai.models.google.GoogleModelSettings.google_cached_content] for explicit cached-content resources |
| Other providers | Typically implicit where supported | Consult the provider documentation |

The provider pages linked in the table document the configuration mechanics — for example, Anthropic's [`anthropic_cache`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache], [`anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions], [`anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions], and [`anthropic_cache_messages`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_messages] settings, Bedrock's minimum-token thresholds, and OpenRouter's per-downstream-provider differences.

## Prefix-stability guarantees

Pydantic AI makes the following guarantees about the prompt prefix it sends to providers:

- [Instructions are assembled deterministically](agent.md#instructions) for each request. Static instructions from `Agent(instructions=...)` always sort before dynamic instructions, preserving the static prefix when dynamic content changes.
- Message history is append-only within a run: Pydantic AI does not rewrite or reorder settled messages. User-controlled [history processors](message-history.md#processing-message-history) are the exception.
- Internal bookkeeping such as run IDs, message timestamps, and deferred-tool flags does not reach the provider wire and therefore cannot move the prompt prefix.
- [Vercel AI](ui/vercel-ai.md) and [AG-UI](ui/ag-ui.md) adapter round-trips are tested to reconstruct histories that serialize back to the same provider request bytes for the wire-relevant fields (tool call arguments, thinking signatures). Older UI protocol versions can be lossy — for example, AG-UI versions without a reasoning carrier drop thinking parts, which moves the prefix — so keep the client packages current.
- Every recorded provider conversation in CI is checked for wire-level prefix stability, so a framework change that starts moving prefixes fails CI in the pull request that introduces it.

## What invalidates a cache

### Changes under your control

- Minute-precision timestamps or other per-request values in instructions or system prompts. Prefer date-only granularity when the exact time is not required.
- Reordering or changing tool definitions between steps.
- History processors that rewrite already-sent messages on every request.
- Switching models or providers during a conversation. Each provider maintains a separate cache.

### Provider retention

Provider caches expire after idle gaps. This is unavoidable, but it creates a useful opportunity: schedule history-mutating maintenance for [cache-cold windows](message-history.md#scheduling-maintenance-into-cache-cold-windows), when the next request would pay the full input price anyway. That section documents provider retention expectations and shows how [`prompt_cache_outlook()`][pydantic_ai.profiles.prompt_cache_outlook] can identify those windows.

## Monitoring cache efficiency

Every response's [`RequestUsage`][pydantic_ai.usage.RequestUsage] normalizes `cache_read_tokens` and `cache_write_tokens` across providers. `input_tokens` always includes cached reads, so `cache_read_tokens / input_tokens` is a comparable hit ratio per request and, through [`RunUsage`][pydantic_ai.usage.RunUsage], per run.

When [instrumentation](logfire.md) is enabled, model-request spans carry `pydantic_ai.cache.hit_ratio` and `pydantic_ai.cache.established_tokens`. A cache collapse also records `pydantic_ai.cache.collapsed`, `pydantic_ai.cache.wasted_tokens`, and `pydantic_ai.cache.collapse_reason`, whose value is `'unexpected'`, `'ttl-expired'`, `'unknown'`, or `'unreported'`. Only an `'unexpected'` collapse — one that happens while the provider's documented retention window should still be active — additionally emits a `pydantic_ai.cache.collapse` span event; model switches never register as collapses at all, since each provider and model's cache is tracked separately. See the [Logfire documentation](logfire.md#prompt-cache-health) for the authoritative attribute definitions.

## Rules for extension authors

Tool, toolset, and capability authors should preserve the same contract:

- Never mutate the cached prefix during a run.
- Put per-turn dynamic content in the user message, in tool results, or after the last [`CachePoint`][pydantic_ai.messages.CachePoint].
- Keep tool definitions and their ordering stable across steps.
- Use date-only timestamps in instructions.
