# Frequently Asked Questions

## How do deferred tools and prompt caching work together?

Prompt caching keys on a **stable prefix**: providers cache the longest unchanged run of tokens from the start of the request, in roughly the order tool definitions → system/instructions → message history. A change at any layer invalidates the cache for that layer and everything after it — so **changing, adding, removing, or reordering a tool definition invalidates the cache**, because tool definitions sit at the very front.

[Tool search](tools-advanced.md#tool-search) and [deferred loading](toolsets.md#deferred-loading) exist to grow the set of tools the model can reach *without* editing that prefix. Pydantic AI picks the strategy per provider automatically, and the trade-off is about caching:

- **Native tool search** (Anthropic, OpenAI Responses) keeps deferred tools out of the cached prefix and appends discovered tools to the *end* of the conversation, so **discovery preserves the cache**.
- **The local `search_tools` fallback** (every other provider) reveals a discovered tool by adding it to the tools array, which **costs one cache invalidation per discovery turn**.

This is covered in depth, with the per-provider execution modes, under [Tool Search](tools-advanced.md#tool-search).

!!! note "Deferring saves context — it is not dynamic registration"
    With either strategy, every tool the model can ever reach must be **declared in the request up front**. Deferring keeps unused definitions out of the model's context (and, natively, out of the cached prefix); it does not let you register a brand-new, never-declared tool mid-conversation. Introducing a genuinely new tool changes the tools array, which invalidates the cache from that point on.

For a genuinely open-ended tool universe, route everything through a single, stable tool. The harness [`CodeMode`](harness/code-mode.md) capability collapses many tools into one `run_code` tool whose definition stays byte-stable; newly discovered tools are surfaced as callables inside the sandbox rather than as new tool schemas, keeping the tool-definitions prefix — and its cache — intact across discoveries.

### Related caching controls

- Restricting the *active* tools with [`tool_choice`](tools-advanced.md#tool-choice) can also invalidate the cache when Pydantic AI has to trim the array client-side — see [Prompt caching implications](tools-advanced.md#tool-choice-caching) for the per-provider breakdown and the cache-preserving alternatives (`allowed_tools`, `allowed_function_names`, `ToolOrOutput`).
- To place explicit cache breakpoints on messages, use [`CachePoint`][pydantic_ai.messages.CachePoint] (honored by Anthropic, Bedrock, and OpenRouter). Anthropic's tool, system, and instruction caching settings are documented under [Anthropic prompt caching](models/anthropic.md#prompt-caching).

### Does changing a tool definition always break the cache?

On providers that cache tool definitions at the front of the prefix — Anthropic, OpenAI, and xAI — yes: editing a single tool's description is enough to invalidate the cached prefix. Google's *implicit* cache is prefix-based on a different layout (its `system_instruction` is a separate field ahead of the tool block), so a large stable system instruction can keep cache hits even when the tool list changes; an explicit [`CachedContent`](models/google.md) instead fixes the tools as an immutable part of the cache by construction.

| Provider | Native tool search | Cache-preserving tool restriction |
|----------|:------------------:|-----------------------------------|
| Anthropic | ✓ (`defer_loading`) | `tool_choice` (single tool) |
| OpenAI Responses | ✓ (`tool_search`) | `allowed_tools` |
| OpenAI Chat Completions | — (local fallback) | — |
| Google | — (local fallback) | `allowed_function_names` |
| xAI | — (local fallback) | `tool_choice` |
