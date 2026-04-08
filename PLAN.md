# Plan: Pluggable Tool Search Strategy

## Context

Follow-up to PR #4090 (tool search for deferred-loaded tools). This plan makes the search strategy pluggable via a `ToolSearch` capability, allowing users to swap in custom matching logic or let `'auto'` route to the best available strategy per provider.

## Design Decisions

Sources: DouweM review on #4960, Slack call 2026-04-07, PR #4090 review threads.

- **`strategy` field, not separate `search_fn`**: all search options are mutually exclusive → single key. `Literal['auto', 'substring', 'regex', 'bm25', 'tool_search'] | Callable[..., Sequence[str]]`, default `'auto'`. Native strategy names included from the start per DouweM review — sets us up for native implementation without API changes.
- **BuiltinOrLocalTool subclass** (not config-holder): `ToolSearch` subclasses `BuiltinOrLocalTool`, inheriting automatic routing between provider-native (builtin) and local (substring) implementations via `prefer_builtin` on `ToolDefinition`. Same pattern as `WebSearch`, `ImageGeneration`.
- **Central `_DEFAULT_STRATEGY` mapping**: `dict[str, str]` mapping provider names to their best native strategy. `'auto'` resolves via this mapping at request time. Stored with the TST so we can update defaults without breaking changes.
- **No separate config data class**: options are kwargs on the capability directly.
- **Custom callable**: returns `Sequence[str]` (tool names to include), description read from docstring (same as tool functions).
- **TST wraps before capability wrappers**: capabilities are outermost so they can change anything.
- **Use `visit_and_replace`** (not a custom `_find_tool_search` helper) for extracting config from the capability tree.
- **Naming**: `search_guidance` (not `keywords_description`) for the model-facing prompt on how to formulate search input. `tool_description` (not `description`) for the search tool's description text.

### Provider native strategies (researched)

| Provider | Strategies | Notes |
|----------|-----------|-------|
| Ours | `substring` | split terms, `any(term in searchable)` |
| Anthropic | `regex`, `bm25` (versioned `_20251119`) | server-side, hides name + desc + schema |
| OpenAI | `tool_search` (single, no variants) | server or client execution, hides schema only |

### Default strategy mapping

```python
_DEFAULT_STRATEGY: dict[str, str] = {
    'anthropic': 'bm25',
    'openai': 'tool_search',
    # all others: fall back to 'substring'
}
```

### How `strategy` maps to BuiltinOrLocalTool routing

| strategy value | builtin | local | effect |
|---|---|---|---|
| `'auto'` (default) | `True` (default builtin) | substring TST | provider with native support → builtin; without → local substring |
| `'substring'` | `False` | substring TST | always local, any provider |
| `'bm25'`, `'regex'` | `True` + specific config | substring TST | force specific Anthropic native strategy; error if provider doesn't support |
| `'tool_search'` | `True` + specific config | substring TST | force OpenAI native strategy; error if provider doesn't support |
| `Callable` | `False` | custom search via callable | always local with user's function |

## Changes

### 1. Make `_SearchIndexEntry` public as `ToolSearchEntry`

**File:** `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`

Rename `_SearchIndexEntry` → `ToolSearchEntry`. This is the type users need to implement custom search functions.

**Export from:** `pydantic_ai/toolsets/__init__.py`

### 2. Define `ToolSearchFunc` type

**File:** `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`

```python
ToolSearchFunc = Callable[[str, Sequence[ToolSearchEntry]], Sequence[str]]
```

`(keywords: str, entries) -> tool_names`. Pass raw `str` (not pre-split) — custom fns may tokenize differently (e.g. semantic search treats whole phrase as input).

**Export from:** `pydantic_ai/toolsets/__init__.py`

### 3. Add `search_fn` + `max_results` + prompt overrides to `ToolSearchToolset`

**File:** `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`

```python
@dataclass
class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    search_fn: ToolSearchFunc | None = None
    max_results: int = _MAX_SEARCH_RESULTS
    tool_description: str | None = None
    search_guidance: str | None = None
```

Also add `_DEFAULT_STRATEGY` mapping here.

- In `_search_tools`: if `self.search_fn` is set, call it and cap results at `self.max_results`. Otherwise current substring logic using `self.max_results`.
- In `get_tools`: use `self.tool_description` / `self.search_guidance` if set, otherwise fall back to current hardcoded defaults.

The `keywords` parameter name on the search tool stays fixed — changing schema field names would require changing arg parsing. `search_guidance` like `'Natural language description of the tools you need'` gets 90% of the benefit.

### 4. Create `ToolSearch` capability

**New file:** `pydantic_ai_slim/pydantic_ai/capabilities/tool_search.py`

```python
@dataclass(init=False)
class ToolSearch(BuiltinOrLocalTool[AgentDepsT]):
    strategy: Literal['auto', 'substring', 'regex', 'bm25', 'tool_search'] | Callable[..., Sequence[str]] = 'auto'
    max_results: int = 10
    tool_description: str | None = None
    search_guidance: str | None = None

    def _default_builtin(self) -> ToolSearchTool | None:
        return ToolSearchTool()

    def _builtin_unique_id(self) -> str:
        return ToolSearchTool.kind  # 'tool_search'

    def _default_local(self) -> Tool | AbstractToolset | None:
        # return local substring search toolset
        ...

    def _requires_builtin(self) -> bool:
        # strategy values that only work with native providers
        return isinstance(self.strategy, str) and self.strategy in ('bm25', 'regex', 'tool_search')
```

`__init__` sets `builtin`/`local` based on `strategy`:
- `'auto'` → `builtin=True`, `local=<default>`
- `'substring'` → `builtin=False`, `local=<default>`
- `Callable` → `builtin=False`, `local=<custom>`

**Export from:** `pydantic_ai/capabilities/__init__.py` (add to `__all__`, NOT to `CAPABILITY_TYPES`)

### 5. Move TST before capability wrappers

**File:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

Reorder in `_get_toolset`:
```
BEFORE:                          AFTER:
1. PreparedToolset               1. PreparedToolset
2. capability wrappers           2. ToolSearchToolset (with extracted config)
3. ToolSearchToolset             3. capability wrappers (outermost)
```

### 6. Agent wiring: extract config via `visit_and_replace`

**File:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

Use `visit_and_replace` (per DouweM review) instead of a custom `_find_tool_search` helper to locate `ToolSearch` in the capability tree and extract its config for the TST.

Also: need a canonical pattern for "auto inject unless already injected" capabilities (DouweM's note). This applies to TST — if user already has a `ToolSearch` capability, don't double-inject.

### 7. Tests

**File:** `tests/test_tool_search.py`

- `test_custom_search_fn`: Pass custom fn to `ToolSearchToolset`, verify it's called and results used
- `test_custom_max_results`: Set `max_results=2`, verify capping
- `test_custom_tool_description`: Set `tool_description` + `search_guidance`, verify they appear in the search tool def
- `test_tool_search_capability_integration`: Agent with `capabilities=[ToolSearch(strategy=...)]` + deferred tools, verify custom search is used
- `test_tool_search_capability_prompt_override`: Agent with `capabilities=[ToolSearch(tool_description=..., search_guidance=...)]`, verify the search_tools tool def uses custom prompts
- `test_default_behavior_preserved`: Agent with `capabilities=[ToolSearch()]` (no custom fn), verify substring matching unchanged
- `test_capability_wrapping_over_tst`: Verify a capability's `get_wrapper_toolset` wraps outside TST

### 8. Documentation (deferred until after review)

Per PR flow, docs/docstrings left as placeholders until logic is confirmed.

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

# Custom prompts
agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        ToolSearch(
            tool_description='Search for tools by describing what you need.',
            search_guidance='A natural language description of the capability you need.',
        )
    ],
)
```

## Open Questions

1. `ToolSearchFunc` signature: `(keywords: str, entries: Sequence[ToolSearchEntry]) -> Sequence[str]` — confirm return type is tool names (not filtered entries).
2. Canonical "auto inject unless already injected" pattern for capabilities — does this need a general solution in this PR or just the TST-specific case?
3. When `strategy='auto'` and provider has no native support, should we fall back to `'substring'` silently or warn? Silently seems right (like `WebSearch` falling back to local).

## File Manifest

| File | Action |
|------|--------|
| `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py` | Make entry public, add type + params + `_DEFAULT_STRATEGY` |
| `pydantic_ai_slim/pydantic_ai/toolsets/__init__.py` | Export `ToolSearchEntry`, `ToolSearchFunc` |
| `pydantic_ai_slim/pydantic_ai/capabilities/tool_search.py` | **New** — `ToolSearch(BuiltinOrLocalTool)` capability |
| `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` | Export `ToolSearch` |
| `pydantic_ai_slim/pydantic_ai/agent/__init__.py` | Reorder TST, extract ToolSearch config via `visit_and_replace` |
| `tests/test_tool_search.py` | Custom search tests |

## Verification

1. `make format && make lint && make typecheck`
2. `uv run pytest tests/test_tool_search.py -x`
3. Verify custom search fn replaces substring matching (unit test)
4. Verify capability wrapper order (capability wraps outside TST)
