# Plan: ToolSearch Capability + Native Provider Support

## Context

PR #4090 implements `ToolSearchToolset` — portable client-side tool search with substring matching.
This plan adds:
1. A `ToolSearch` capability following the `BuiltinOrLocalTool` pattern (like `WebSearch`, `ImageGeneration`)
2. A `strategy` field with `'auto'` default that routes to the best available strategy per provider
3. Native Anthropic and OpenAI tool search support via builtin tools

Supersedes the original native-only plan and the pluggable-search-strategy plan by consolidating
both into the capabilities framework.

## Design Decisions (from Slack call 2026-04-07)

- **Single search tool**: providers (Anthropic, OpenAI) expose one search tool, not multiple. We follow suit.
- **`strategy` field**: `Literal['auto', 'substring'] | Callable`, default `'auto'`. Mutually exclusive options on one key.
- **`'auto'` routes via provider mapping**: central `_DEFAULT_STRATEGY` dict maps providers to their best strategy.
- **Capability pattern**: subclass `BuiltinOrLocalTool` — routing between native and local is already handled by `prefer_builtin` on `ToolDefinition`.
- **No separate config data class**: options are kwargs on the capability.
- **Custom callable**: returns `Sequence[str]` (tool names), description from docstring.
- **Plan stays as PR artifact**: never merged to main.

## Provider Strategies

| Provider | Native strategies | `defer_loading` hides | SDK requirement |
|----------|------------------|----------------------|-----------------|
| Anthropic | `regex`, `bm25` (versioned `_20251119`) | name + desc + schema | anthropic >= 0.86.0 |
| OpenAI | `tool_search` (single, no variants) | schema only (name + desc visible) | openai >= 2.25.0 |
| Others | none | n/a | n/a |

Local strategy: `'substring'` — split query into terms, `any(term in searchable for term in terms)`.

### Default strategy mapping

```python
_DEFAULT_STRATEGY: dict[str, str] = {
    'anthropic': 'bm25',
    'openai': 'tool_search',
    # all others: fall back to 'substring'
}
```

Stored centrally with the TST. `'auto'` resolves via this mapping at request time.
We can update what `'auto'` resolves to without breaking changes.

## Architecture

### ToolSearch capability (subclass of BuiltinOrLocalTool)

```python
@dataclass(init=False)
class ToolSearch(BuiltinOrLocalTool[AgentDepsT]):
    strategy: Literal['auto', 'substring'] | Callable[..., Sequence[str]] = 'auto'

    def _default_builtin(self) -> ToolSearchTool | None:
        return ToolSearchTool()

    def _builtin_unique_id(self) -> str:
        return ToolSearchTool.kind  # 'tool_search'

    def _default_local(self) -> Tool | AbstractToolset | None:
        # return local substring search tool/toolset
        ...

    def _requires_builtin(self) -> bool:
        # strategy values that only work with builtin
        return isinstance(self.strategy, str) and self.strategy in ('bm25', 'regex', 'tool_search')
```

### How routing works (inherited from BuiltinOrLocalTool)

1. Capability registers both builtin (`ToolSearchTool`) and local (substring search) tools
2. Local tool gets `prefer_builtin='tool_search'` via `PreparedToolset`
3. Model provider checks `profile.supported_builtin_tools` for `ToolSearchTool`
4. If supported: builtin sent, local removed. If not: builtin removed, local sent.
5. Zero custom routing logic needed.

### Builtin part mapping (same pattern as web search)

- Anthropic: `server_tool_use` -> `BuiltinToolCallPart(provider_name='anthropic', tool_name='tool_search')`
- Anthropic: `tool_search_tool_result` -> `BuiltinToolReturnPart(provider_name='anthropic', tool_name='tool_search')`
- OpenAI: `tool_search_call` -> `BuiltinToolCallPart(provider_name='openai', tool_name='tool_search')`
- OpenAI: `tool_search_output` -> `BuiltinToolReturnPart(provider_name='openai', tool_name='tool_search')`

Provider-gated: silently skipped when replayed on a different provider.

### Wrapping order

```
1. PreparedToolset (agent-level prepare_tools)
2. ToolSearchToolset (tool search wrapping)
3. Capability wrappers (outermost — capabilities can change anything)
```

TST wraps before capability wrappers per DouweM's review.

## Cross-provider fallback (no metadata bridge)

When native + `ToolSearchToolset` coexist and provider switches:

- Native discovery results stay in history as `BuiltinToolCallPart`/`BuiltinToolReturnPart` — skipped by other providers
- Tool calls/returns from discovered tools (`ToolCallPart`/`ToolReturnPart`) are visible to all providers
- Fallback provider has `search_tools` available to rediscover tools if needed
- No metadata bridge needed — APIs don't validate historical tool calls against current definitions

## Scope

### Phase 1: ToolSearch capability + strategy field

- New `pydantic_ai_slim/pydantic_ai/capabilities/tool_search.py`
  - `ToolSearch(BuiltinOrLocalTool)` with `strategy` field
  - `_default_builtin` -> `ToolSearchTool()`
  - `_default_local` -> local substring search
- `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py`: export `ToolSearch`
- `pydantic_ai_slim/pydantic_ai/builtin_tools.py`: `ToolSearchTool` (kind='tool_search')
- `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`: `_DEFAULT_STRATEGY` mapping
- `pydantic_ai_slim/pydantic_ai/agent/__init__.py`: TST wraps before capability wrappers
- Model profiles: `ToolSearchTool` in `supported_builtin_tools` for Anthropic/OpenAI
- Tests + docs

### Phase 2: Anthropic native

- `models/anthropic.py`:
  - `_map_tool_definition`: respect `defer_loading`, emit Anthropic tool search tool type
  - Parse `server_tool_use` / `tool_search_tool_result` response blocks
  - Request mapping for `BuiltinToolCallPart`/`BuiltinToolReturnPart` with provider='anthropic'
  - Streaming support
- `pyproject.toml`: anthropic SDK bump >= 0.86.0
- Strategy mapping: `'auto'` -> `'bm25'` for Anthropic

### Phase 3: OpenAI native (hosted mode only)

- `models/openai.py`:
  - `_map_tool_definition`: respect `defer_loading`, emit `tool_search` type
  - Parse `ResponseToolSearchCall` / `ResponseToolSearchOutputItem`
  - Request mapping
- Strategy mapping: `'auto'` -> `'tool_search'` for OpenAI
- Client-executed mode: out of scope for now

### NOT touched

- `_agent_graph.py`: no changes
- `toolsets/_searchable.py`: no changes (this is the existing `SearchableToolset`, unrelated)

## User-facing API

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

# Default: auto-routes to best available per provider
agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch()])

# Force local substring search
agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch(strategy='substring')])

# Custom search function (description from docstring)
def my_search(keywords: str, entries: list[ToolSearchEntry]) -> Sequence[str]:
    '''Search tools using semantic similarity.'''
    ...

agent = Agent('openai:gpt-5.4', capabilities=[ToolSearch(strategy=my_search)])
```

## Open Questions

1. Should `strategy` expand to include `'bm25'` / `'regex'` as explicit user-facing options, or only resolve via `'auto'`? Leaning toward making them explicit so users can force a specific native strategy.
2. `ToolSearchFunc` signature: `(keywords: str, entries: Sequence[ToolSearchEntry]) -> Sequence[str]` — pass raw string (custom fns may tokenize differently). Confirm this is right.
3. When `strategy='auto'` and provider has no native support, should we use `'substring'` silently or warn? Silently seems right (like `WebSearch` falling back to DuckDuckGo).

## Verification

1. `make format && make lint && make typecheck`
2. `uv run pytest tests/test_tool_search.py -x` (existing tests still pass)
3. VCR tests for Anthropic native (record with `--record-mode=rewrite`)
4. VCR tests for OpenAI native
5. Test capability routing: model with native support uses builtin, model without uses local
6. Test cross-provider fallback: Anthropic native -> OpenAI local (history survives)
7. `make docs-serve` — verify capability docs
