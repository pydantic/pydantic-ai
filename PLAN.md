# Builtin-to-Custom Tool Fallback (#3212)

## Context

When using `FallbackModel` or testing across models, agents with builtin tools (e.g. `WebSearchTool`) fail with `UserError` on models that don't support them. Users need function tools as automatic fallbacks.

**Core insight**: send BOTH the builtin tool AND its fallback function tool in `ModelRequestParameters`. Each model's `prepare_request` independently decides which to keep via `self.profile.supported_builtin_tools`. Zero changes to `FallbackModel` — each inner model makes the right choice.

## Changes

### 1. `ToolDefinition` — add `prefer_builtin` field
**File**: `pydantic_ai_slim/pydantic_ai/tools.py`

Add after `timeout`:
```python
prefer_builtin: str | None = None
```
Matches `AbstractBuiltinTool.unique_id` (e.g. `'web_search'`, `'code_execution'`). Signals: 'if model supports this builtin, prefer it over me'.

### 2. `Tool` — add `prefer_builtin` param
**File**: `pydantic_ai_slim/pydantic_ai/tools.py`

- Add field: `prefer_builtin: str | None`
- Add `__init__` param: `prefer_builtin: AbstractBuiltinTool | None = None`
- Resolve: `self.prefer_builtin = prefer_builtin.unique_id if prefer_builtin is not None else None`
  - `unique_id` defaults to `kind` for most builtins, but `MCPServerTool` returns `f'mcp_server:{id}'`
- Flow to `tool_def` property: `prefer_builtin=self.prefer_builtin`

### 3. `Model.prepare_request` — swap logic
**File**: `pydantic_ai_slim/pydantic_ai/models/__init__.py` (lines ~766-774)

Replace the current 'Check if builtin tools are supported' block:

```
if params.builtin_tools:
    supported_types = self.profile.supported_builtin_tools

    # Partition builtins into supported/unsupported
    supported_builtins = [t for t in params.builtin_tools if isinstance(t, tuple(supported_types))]
    unsupported_ids = {t.unique_id for t in params.builtin_tools if not isinstance(t, tuple(supported_types))}

    if unsupported_ids:
        # Check which unsupported builtins have fallbacks
        fallback_ids = {td.prefer_builtin for td in params.function_tools if td.prefer_builtin}
        without_fallback = unsupported_ids - fallback_ids
        if without_fallback:
            # Raise error only for unsupported builtins with no fallback (preserves existing behavior)
            <existing UserError logic, scoped to without_fallback>

    # Remove fallback function tools whose preferred builtin IS supported
    supported_ids = {t.unique_id for t in supported_builtins}
    function_tools = [td for td in params.function_tools if not td.prefer_builtin or td.prefer_builtin not in supported_ids]

    params = replace(params, builtin_tools=supported_builtins, function_tools=function_tools)
```

### 4. Thread `prefer_builtin` through decorators

**`FunctionToolset.tool()`** + **`FunctionToolset.add_function()`** in `pydantic_ai_slim/pydantic_ai/toolsets/function.py`:
- Add `prefer_builtin: AbstractBuiltinTool | None = None` param to both overloads + implementation
- Pass through to `Tool(...)` constructor

**`Agent.tool()`** + **`Agent.tool_plain()`** in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`:
- Add `prefer_builtin: AbstractBuiltinTool | None = None` param to keyword overloads + implementation
- Pass through to `self._function_toolset.add_function(...)`
- NOT added to the bare-decorator overload (no-parens: `@agent.tool` — no kwargs possible)

## What does NOT change

- `FallbackModel` — each inner model's `prepare_request` handles the swap independently
- `_agent_graph.py` — already builds both lists separately
- `AbstractToolset` / `ToolManager` — `ToolDefinition`s flow through unchanged
- Provider model implementations — they call base `prepare_request`
- `AbstractBuiltinTool` — no `fallback` param (deferred)
- `ModelProfile` — already has `supported_builtin_tools`

## Deferred to follow-up PRs

- `AbstractBuiltinTool(fallback=...)` — builtin carries its own fallback
- `MCPServer`/`FastMCPToolset` `prefer_builtin` — toolset-level support
- `TavilySearchTool(prefer_builtin=True)` — convenience on common tools
- `BuiltinToolset` — wrapping builtins in the toolset system

## Testing

- Unit tests for `prepare_request` swap logic:
  - Supported builtin + fallback -> builtin kept, fallback removed
  - Unsupported builtin + fallback -> builtin removed, fallback kept
  - Unsupported builtin + no fallback -> `UserError` (backward compat)
  - Mix of supported/unsupported with partial fallbacks
- Integration test with `FallbackModel` + `FunctionModel`/`TestModel` end-to-end
- `make typecheck`, `make lint`

## Docs/memory

- Placeholder docstrings on new params (per PR flow — defer full docs until after review)
- Update `CLAUDE.local.md` with branch/issue/PR info after creating branch
