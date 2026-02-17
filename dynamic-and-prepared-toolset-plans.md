<!-- Intentionally included in PR for discussion. Delete this file before merging. -->

# Plan: Fix DynamicToolset + SearchableToolset issues (claims 3 & 4)

## Context

Devin bot review on PR #4090 raised two valid architectural issues with the SearchableToolset wrapping logic in `Agent._get_toolset()`. Both relate to how/when SearchableToolset gets applied.

## Resolution

Both issues were resolved by DouweM's review feedback: **drop `has_deferred_tools()` entirely and always wrap in `SearchableToolset`**.

`SearchableToolset.get_tools()` already handles the no-deferred-tools case gracefully (returns all tools without adding `search_tools`), so always-wrapping is safe and eliminates the need for `has_deferred_tools()` on every toolset class.

### What was done

1. Removed `has_deferred_tools()` from `AbstractToolset`, `FunctionToolset`, `CombinedToolset`, `WrapperToolset`, `FastMCPToolset`, and `MCPServer`
2. `_get_toolset()` now unconditionally wraps in `SearchableToolset`
3. Wrapping order swapped: `SearchableToolset(PreparedToolset(inner))` — so `prepare_tools` never sees `search_tools`

This is simpler than the original plan's `has_deferred_tools() or has_dynamic` approach and handles all cases including `DynamicToolset`.

## Files modified

- `pydantic_ai_slim/pydantic_ai/agent/__init__.py` — `_get_toolset()`
- `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py` — removed `has_deferred_tools()`
- `pydantic_ai_slim/pydantic_ai/toolsets/function.py` — removed override
- `pydantic_ai_slim/pydantic_ai/toolsets/combined.py` — removed override
- `pydantic_ai_slim/pydantic_ai/toolsets/wrapper.py` — removed override
- `pydantic_ai_slim/pydantic_ai/toolsets/fastmcp.py` — removed override
- `pydantic_ai_slim/pydantic_ai/mcp.py` — removed override
