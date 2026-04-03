# Plan: Pluggable Tool Search Strategy

## Context

Follow-up to PR #4090 (tool search for deferred-loaded tools). DouweM noted that making ToolSearchToolset a capability is "not worth doing now, but in the future." This plan implements that future direction: a pluggable search strategy via a `ToolSearch` capability, allowing users to swap in semantic search, BM25, or any custom matching logic.

Research in `local-notes/tool-search-research.md` shows the community unanimously moving toward semantic/vector search for tool discovery (97%+ hit rates vs substring's inherent limitations). Spring AI's biggest win is making the strategy pluggable. This plan brings that flexibility to Pydantic AI.

## Design: Config-holder capability (Approach A)

- `ToolSearchToolset` gets `search_fn` + `max_results` params (defaulting to current substring behavior)
- New `ToolSearch` capability holds config, does NOT wrap itself
- Agent detects `ToolSearch` in capabilities, extracts config, passes to hardcoded TST
- TST moves before capability wrappers so capabilities can override search_tools behavior
- No auto-injection, no double-wrapping, backward compatible

## Changes

### 1. Make `_SearchIndexEntry` public as `ToolSearchEntry`

**File:** `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`

Rename `_SearchIndexEntry` → `ToolSearchEntry` (drop underscore). This is the type users need to implement custom search functions.

**Export from:** `pydantic_ai/toolsets/__init__.py`

### 2. Define `ToolSearchFunc` type

**File:** `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`

```python
ToolSearchFunc = Callable[[str, Sequence[ToolSearchEntry]], Sequence[ToolSearchEntry]]
```

`(keywords: str, entries) -> matches`. Sync — search is fast in-memory.

**Design decision:** Pass raw `str` keywords (not pre-split `list[str]`). Rationale: custom search fns may want to do their own tokenization (e.g., semantic search treats the whole phrase as input, not individual terms). The default substring impl splits internally.

**Export from:** `pydantic_ai/toolsets/__init__.py`

### 3. Add `search_fn` + `max_results` + prompt overrides to `ToolSearchToolset`

**File:** `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py`

```python
@dataclass
class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    search_fn: ToolSearchFunc | None = None
    max_results: int = _MAX_SEARCH_RESULTS
    description: str | None = None
    keywords_description: str | None = None
```

- In `_search_tools`: if `self.search_fn` is set, call it and cap results at `self.max_results`. Otherwise current substring logic using `self.max_results`.
- In `get_tools`: use `self.description` / `self.keywords_description` if set, otherwise fall back to current hardcoded defaults. This lets users tailor the prompt to their search strategy (e.g. semantic search works better with natural language phrases than space-separated keywords).

**Note:** The `keywords` parameter name stays fixed — changing the schema field name would require changing arg parsing. A custom `keywords_description` like `'Natural language description of the tools you need'` gets 90% of the benefit without that complexity. Full schema override can be a future extension if needed.

### 4. Create `ToolSearch` capability

**New file:** `pydantic_ai_slim/pydantic_ai/capabilities/tool_search.py`

```python
@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    """Customize tool search strategy for deferred tool discovery."""
    search_fn: ToolSearchFunc | None = None
    max_results: int = 10
    description: str | None = None
    keywords_description: str | None = None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # callable → not serializable
```

Config-holder — no `get_wrapper_toolset`, no `get_toolset`.

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

This fulfills DouweM's review request: "capabilities need to be able to change basically anything."

### 6. Agent wiring: extract config from ToolSearch capability

**File:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

Helper to find `ToolSearch` in capability tree:
```python
def _find_tool_search(cap):
    if isinstance(cap, ToolSearch): return cap
    if isinstance(cap, CombinedCapability):
        for c in cap.capabilities:
            if found := _find_tool_search(c): return found
    if isinstance(cap, WrapperCapability):
        return _find_tool_search(cap.wrapped)
    return None
```

In `_get_toolset`:
```python
ts_cap = _find_tool_search(run_capability)
toolset = ToolSearchToolset(
    wrapped=toolset,
    search_fn=ts_cap.search_fn if ts_cap else None,
    max_results=ts_cap.max_results if ts_cap else _MAX_SEARCH_RESULTS,
    description=ts_cap.description if ts_cap else None,
    keywords_description=ts_cap.keywords_description if ts_cap else None,
)
```

### 7. Documentation

**File:** `docs/tools-advanced.md`

Add subsection under "Tool Search" for custom search strategy:

```markdown
### Custom Search Strategy

By default, tool search uses substring matching on space-separated keywords. Use the
[`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability to customize the search
function and/or the prompt shown to the model:

\```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.toolsets import ToolSearchEntry

def semantic_search(keywords: str, entries: list[ToolSearchEntry]) -> list[ToolSearchEntry]:
    # your embedding/vector logic here
    ...

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        ToolSearch(
            search_fn=semantic_search,
            description='Search for tools by describing what you need in natural language.',
            keywords_description='A natural language description of the capability you need.',
        )
    ],
)
\```
```

**File:** `docs/capabilities.md` — add `ToolSearch` to built-in capabilities table.

### 8. Tests

**File:** `tests/test_tool_search.py`

- `test_custom_search_fn`: Pass custom fn to `ToolSearchToolset`, verify it's called and results used
- `test_custom_max_results`: Set `max_results=2`, verify capping
- `test_custom_description`: Set `description` + `keywords_description`, verify they appear in the search tool def shown to the model
- `test_tool_search_capability_integration`: Agent with `capabilities=[ToolSearch(search_fn=...)]` + deferred tools, verify custom search is used
- `test_tool_search_capability_prompt_override`: Agent with `capabilities=[ToolSearch(description=..., keywords_description=...)]`, verify the search_tools tool def uses custom prompts
- `test_default_behavior_preserved`: Agent with `capabilities=[ToolSearch()]` (no custom fn), verify substring matching unchanged
- `test_capability_wrapping_over_tst`: Verify a capability's `get_wrapper_toolset` wraps outside TST (i.e., TST is now inner)

## File manifest

| File | Action |
|------|--------|
| `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py` | Make entry public, add type + params |
| `pydantic_ai_slim/pydantic_ai/toolsets/__init__.py` | Export `ToolSearchEntry`, `ToolSearchFunc` |
| `pydantic_ai_slim/pydantic_ai/capabilities/tool_search.py` | **New** — `ToolSearch` capability |
| `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` | Export `ToolSearch` |
| `pydantic_ai_slim/pydantic_ai/agent/__init__.py` | Reorder TST, extract ToolSearch config |
| `docs/tools-advanced.md` | Custom search strategy section |
| `docs/capabilities.md` | Add ToolSearch to table |
| `tests/test_tool_search.py` | Custom search tests |

## Verification

1. `make format && make lint && make typecheck`
2. `uv run pytest tests/test_tool_search.py -x`
3. Verify custom search fn actually replaces substring matching (unit test)
4. Verify capability wrapper order (capability wraps outside TST)
5. `make docs-serve` — verify new docs section
