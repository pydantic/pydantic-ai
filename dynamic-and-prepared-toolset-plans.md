<!-- Intentionally included in PR for discussion. Delete this file before merging. -->

# Plan: Fix DynamicToolset + SearchableToolset issues (claims 3 & 4)

## Context

Devin bot review on PR #4090 raised two valid architectural issues with the SearchableToolset wrapping logic in `Agent._get_toolset()`. Both relate to how/when SearchableToolset gets applied.

## Issue 1: DynamicToolset silently hides deferred tools (claim 3)

**Problem**: `DynamicToolset` inherits `has_deferred_tools() -> False`. Its inner toolset is created at runtime so it can't know at construction time if deferred tools exist. `_get_toolset()` checks `has_deferred_tools()` at setup time → never wraps DynamicToolset in SearchableToolset → deferred tools silently hidden.

**Key insight**: `SearchableToolset.get_tools()` already handles the no-deferred-tools case gracefully (line 80-81 of `_searchable.py`: `if not deferred: return all_tools`). So always-wrapping is safe.

**Fix**: In `_get_toolset()`, also wrap when any DynamicToolset exists in the hierarchy.

```python
# agent/__init__.py, _get_toolset()
has_dynamic = False
def copy_dynamic_toolsets(toolset):
    nonlocal has_dynamic
    if isinstance(toolset, DynamicToolset):
        has_dynamic = True
        return toolset.copy()
    return toolset

toolset = toolset.visit_and_replace(copy_dynamic_toolsets)

if toolset.has_deferred_tools() or has_dynamic:
    toolset = SearchableToolset(wrapped=toolset)
```

Piggyback on the existing `visit_and_replace` traversal — no extra traversal needed.

## Issue 2: PreparedToolset can filter out search_tools (claim 4)

**Problem**: Current wrapping order is `PreparedToolset(SearchableToolset(inner))`. User's `prepare_tools` callback sees and can filter out the `search_tools` synthetic tool.

**Fix**: Swap wrapping order so SearchableToolset wraps *outside* PreparedToolset:

```python
# Before (current):
if toolset.has_deferred_tools() or has_dynamic:
    toolset = SearchableToolset(wrapped=toolset)
if self._prepare_tools:
    toolset = PreparedToolset(toolset, self._prepare_tools)

# After (fixed):
if self._prepare_tools:
    toolset = PreparedToolset(toolset, self._prepare_tools)
if toolset.has_deferred_tools() or has_dynamic:
    toolset = SearchableToolset(wrapped=toolset)
```

Result: `SearchableToolset(PreparedToolset(inner))` — `prepare_tools` never sees `search_tools`, SearchableToolset adds it on top after preparation.

**Note**: `has_deferred_tools()` is called on `PreparedToolset` now. `PreparedToolset` extends `WrapperToolset` which delegates `has_deferred_tools()` to `wrapped` — so this still works correctly.

## Files to modify

- `pydantic_ai_slim/pydantic_ai/agent/__init__.py` — `_get_toolset()` (~lines 1541-1555)

## Tests

- Add test: `DynamicToolset` returning `FunctionToolset(defer_loading=True)` → SearchableToolset is applied and search works
- Add test: `prepare_tools` + deferred tools → `search_tools` not visible to prepare callback, still works
- Check existing tests still pass

## Verification

1. `make typecheck` — confirm type safety
2. `uv run pytest tests/test_searchable_toolset.py` — existing + new tests
3. `make lint` — formatting

## Docs

No doc changes needed — these are internal fixes, not user-facing API changes.
