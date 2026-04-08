# Native Provider Tool Search — Follow-up PR Plan

## Context

PR #4090 (`ToolSearchToolset`) implements portable client-side tool search. Follow-up PRs add
native Anthropic (#4167) and OpenAI (#4566) tool search. This plan captures the simplified
architecture after deciding to **drop the cross-provider metadata bridge**.

## Key Decision: No Cross-Provider Discovery Bridge

### The scenario

1. Anthropic + native tool search discovers `find_tickers`, calls it, gets results
2. Fallback to OpenAI
3. OpenAI sees `ToolCallPart('find_tickers')` + `ToolReturnPart('find_tickers', results)` in history
4. But `find_tickers` is NOT in OpenAI's tool definitions (still deferred — native discovery used `BuiltinToolReturnPart`, not `ToolReturnPart(tool_name='search_tools')`, so `_parse_discovered_tools()` doesn't see it)
5. OpenAI has `search_tools` available to rediscover `find_tickers` if needed

### Why this is fine

- OpenAI already has the **results** of `find_tickers` — it doesn't need the tool definition to understand what happened
- If it needs `find_tickers` again (unlikely — it already has results), one `search_tools` call rediscovers it
- More likely: it wants the *next* tool (e.g., `get_ticker_price`) and discovers that via `search_tools`
- APIs don't validate historical tool calls against current definitions — `ToolCallPart`/`ToolReturnPart` are just history records
- Context optimization: fallback provider doesn't carry stale tool definitions it may never need

### What this eliminates

- ~~`_agent_graph.py`: scan `BuiltinToolReturnPart(tool_name='tool_search')`, extract discovered names, set `ModelRequest.metadata`~~
- ~~`_searchable.py`: extend `_parse_discovered_tools()` to read `ModelRequest.metadata`~~
- Zero changes to `_agent_graph.py` or `_searchable.py` for native PRs

Native PRs become **purely model-adapter work** + `builtin_tools.py` + model profiles.

## Provider Comparison

| Aspect | Anthropic | OpenAI |
|--------|-----------|--------|
| Search tool type | `tool_search_tool_regex/bm25_20251119` | `tool_search` |
| What `defer_loading` hides | Everything (name + desc + schema) | Only schema (name + desc still visible) |
| Execution | Always server-side | Server (hosted) or client |
| Response blocks | `server_tool_use` + `tool_search_tool_result` | `tool_search_call` + `tool_search_output` |
| SDK status | Needs bump: 0.80.0 → ≥0.86.0 | Already supported (≥2.25.0) |
| Model support | Sonnet 4.0+ / Opus 4.0+ | gpt-5.4+ |

## Architecture

### Builtin part mapping (same as web search pattern)

- Anthropic: `server_tool_use` → `BuiltinToolCallPart(provider_name='anthropic', tool_name='tool_search')`
- Anthropic: `tool_search_tool_result` → `BuiltinToolReturnPart(provider_name='anthropic', tool_name='tool_search')`
- OpenAI: `tool_search_call` → `BuiltinToolCallPart(provider_name='openai', tool_name='tool_search')`
- OpenAI: `tool_search_output` → `BuiltinToolReturnPart(provider_name='openai', tool_name='tool_search')`

Provider-gated: silently skipped when replayed on a different provider.

### `ToolSearchToolset` always wraps — native is additive

When native + `ToolSearchToolset` coexist, two discovery paths:
- Native (faster, cached, provider-specific)
- `search_tools` (portable fallback, always available)

On provider switch, only `search_tools` survives. This is by design.

## Scope

### #4167 — Anthropic native
- `models/anthropic.py`: `_map_tool_definition` (defer_loading), parse `server_tool_use`/`tool_search_tool_result`, request mapping, streaming
- `pyproject.toml`: anthropic SDK bump ≥0.86.0
- `builtin_tools.py`: new `ToolSearchTool` (kind='tool_search')
- Model profiles: capability flag

### #4566 — OpenAI native (hosted mode only)
- `models/openai.py`: `_map_tool_definition` (defer_loading), parse `ResponseToolSearchCall`/`ResponseToolSearchOutputItem`, request mapping
- `builtin_tools.py`: same `ToolSearchTool` reused
- Model profiles: capability flag
- Client-executed mode: out of scope

### NOT touched (the simplification)
- `_agent_graph.py`: no changes
- `toolsets/_searchable.py`: no changes

## Message History (provider switch)

```
ModelResponse (from Anthropic):
┌────────────────────────────────────────────────────────────┐
│ BuiltinToolCallPart(provider='anthropic', name='tool_search') │ Anthropic sees; others skip
│ BuiltinToolReturnPart(provider='anthropic', content=...)      │ Anthropic sees; others skip
│ ToolCallPart(name='find_tickers', args={...})                 │ ALL providers see
└────────────────────────────────────────────────────────────┘

ModelRequest:
┌────────────────────────────────────────────────────────────┐
│ ToolReturnPart(name='find_tickers', content=[...])            │ ALL providers see
└────────────────────────────────────────────────────────────┘

On OpenAI fallback:
- Skips Builtin* parts (provider mismatch)
- Sees find_tickers call + results in history (understands what happened)
- find_tickers NOT in tool definitions (still deferred)
- search_tools IS available → can rediscover if needed
- Most likely: moves to next action (get_ticker_price) via search_tools
```

## Session Action Items

1. Update `local-notes/wh1isper-message-draft.md` — remove metadata bridge details, simplify to "native discovery doesn't bridge across providers; `search_tools` rediscovers on fallback"
2. Track docs note for the two-path choice
