# Prompt Caching

Provider prompt caching reuses a processed prompt prefix and charges a steeply discounted rate for tokens read from the cache, often around 10% of the normal input-token price. A cache hit requires the serialized request to be an exact byte prefix of an earlier request in the provider's cache order: tool definitions, then the system prompt, then messages. Moving one byte early in the request silently causes everything after it to be charged again. Because [agent runs](agent.md#running-agents) resend the whole conversation at every step, caching is often the difference between paying for the history once and paying for it on every turn.

## Enabling prompt caching

| Provider | Behavior | Configuration |
| --- | --- | --- |
| [OpenAI](models/openai.md) | Implicit for prompts of at least 1,024 tokens | Nothing to enable |
| [Anthropic](models/anthropic.md#prompt-caching) | Explicit, with a 5-minute default TTL and a 1-hour opt-in | Cache settings or [`CachePoint`][pydantic_ai.messages.CachePoint] |
| [Bedrock](models/bedrock.md#prompt-caching) | Explicit; minimum-token thresholds apply | [`CachePoint`][pydantic_ai.messages.CachePoint] and cache settings |
| [OpenRouter](models/openrouter.md#prompt-caching) | Passes explicit caching through to Anthropic and Gemini models | Cache settings or [`CachePoint`][pydantic_ai.messages.CachePoint] |
| [Google](models/google.md) | Implicit caching is automatic | Nothing to enable |
| Other providers | Typically implicit where supported | Consult the provider documentation |

OpenAI applies caching automatically to eligible prompts. No model setting or marker is required.

Anthropic supports `anthropic_cache`, `anthropic_cache_instructions`, `anthropic_cache_tool_definitions`, and `anthropic_cache_messages`, as well as explicit [`CachePoint`][pydantic_ai.messages.CachePoint] markers. The default TTL is 5 minutes; use `'1h'` where supported to opt into a 1-hour TTL. See [Anthropic prompt caching](models/anthropic.md#prompt-caching) for configuration details.

Bedrock represents [`CachePoint`][pydantic_ai.messages.CachePoint] markers as Converse `cachePoint` blocks, and applies provider-specific minimum-token thresholds. See [Bedrock prompt caching](models/bedrock.md#prompt-caching) for the available settings and constraints.

OpenRouter passes cache settings and [`CachePoint`][pydantic_ai.messages.CachePoint] markers through to supported Anthropic and Gemini models. See [OpenRouter prompt caching](models/openrouter.md#prompt-caching) for the differences between downstream providers.

Google Gemini applies implicit caching automatically. Explicit `CachedContent` is separate provider infrastructure and is not covered here.

Other providers, including Groq and DeepSeek, typically apply caching implicitly where they support it. Consult the provider's documentation for eligibility and retention details.

## Prefix-stability guarantees

Pydantic AI makes the following guarantees about the prompt prefix it sends to providers:

- [Instructions are assembled deterministically](agent.md#instructions) for each request. Static instructions from `Agent(instructions=...)` always sort before dynamic instructions, preserving the static prefix when dynamic content changes.
- Message history is append-only within a run: Pydantic AI does not rewrite or reorder settled messages. User-controlled [history processors](message-history.md#processing-message-history) are the exception.
- Internal bookkeeping such as run IDs, message timestamps, and deferred-tool flags does not reach the provider wire and therefore cannot move the prompt prefix.
- [Vercel AI](ui/vercel-ai.md) and [AG-UI](ui/ag-ui.md) adapter round-trips reconstruct histories that serialize back to the same provider request bytes.
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

When [instrumentation](logfire.md) is enabled, model-request spans carry `pydantic_ai.cache.hit_ratio` and `pydantic_ai.cache.established_tokens`. A cache collapse also records `pydantic_ai.cache.collapsed`, `pydantic_ai.cache.wasted_tokens`, and `pydantic_ai.cache.collapse_reason`, whose value is `'ttl-expired'`, `'unknown'`, or `'unexpected'`. An unexpected collapse — one not explained by TTL expiry or a model switch — additionally emits a `pydantic_ai.cache.collapse` span event. See the [Logfire documentation](logfire.md) for the authoritative attribute definitions.

## Rules for extension authors

Tool, toolset, and capability authors should preserve the same contract:

- Never mutate the cached prefix during a run.
- Put per-turn dynamic content in the user message, in tool results, or after the last [`CachePoint`][pydantic_ai.messages.CachePoint].
- Keep tool definitions and their ordering stable across steps.
- Use date-only timestamps in instructions.
