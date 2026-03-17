# Builtin-to-Custom Tool Fallback (#3212)

## Context

When using `FallbackModel` or testing across models, agents with builtin tools (e.g. `WebSearchTool`) fail with `UserError` on models that don't support them. Users need function tools as automatic fallbacks.

**Core insight**: send BOTH the builtin tool AND its fallback function tool in `ModelRequestParameters`. Each model's `prepare_request` independently decides which to keep via `self.profile.supported_builtin_tools`. Zero changes to `FallbackModel` — each inner model makes the right choice.

**Important**: `prefer_builtin` on a function tool does NOT auto-register the corresponding builtin tool with the agent. The user must still include it in `builtin_tools=[...]`. This is by design — the upcoming `capabilities=[WebSearch()]` API (see Relationship to Deferred Loading below) will handle both sides automatically via `get_toolset()` + `get_builtin_tools()`. This PR provides the low-level primitive that capabilities will build on.

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

- Store field: `prefer_builtin: str | None` (resolved from the input)
- Add `__init__` param: `prefer_builtin: AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT] | None = None`
  - `BuiltinToolFunc` (already at `tools.py:129`) is `Callable[[RunContext[AgentDepsT]], Awaitable[AbstractBuiltinTool | None] | AbstractBuiltinTool | None]`
- Resolve in `prepare_tool_def()` (async, already handles `ToolPrepareFunc`):
  - If `AbstractBuiltinTool` instance: `prefer_builtin = inst.unique_id`
  - If `Callable`: invoke with `RunContext` to get `AbstractBuiltinTool | None`, then `prefer_builtin = result.unique_id if result else None`
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
- Add `prefer_builtin: AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT] | None = None` param to both overloads + implementation
- Pass through to `Tool(...)` constructor

**`Agent.tool()`** + **`Agent.tool_plain()`** in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`:
- Add `prefer_builtin: AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT] | None = None` param to keyword overloads + implementation
- Pass through to `self._function_toolset.add_function(...)`
- NOT added to the bare-decorator overload (no-parens: `@agent.tool` — no kwargs possible)

## What does NOT change

- `FallbackModel` — each inner model's `prepare_request` handles the swap independently
- `_agent_graph.py` — already builds both lists separately
- `AbstractToolset` / `ToolManager` — `ToolDefinition`s flow through unchanged
- Provider model implementations — they call base `prepare_request`
- `AbstractBuiltinTool` — no `fallback` param (deferred)
- `ModelProfile` — already has `supported_builtin_tools`

## Relationship to deferred tool loading (#4090, #4167, #3666)

DouweM notes that deferred tool loading (Anthropic tool search) needs the ability to send fundamentally different tool lists depending on model support — e.g. all tools with `defer_loading=True` for native support vs a `search_tools` meta-tool + already-discovered tools for framework-level support. `prefer_builtin` alone doesn't cover that case because it's a 1:1 swap (one function tool ↔ one builtin), not a structural transformation of the entire tool list.

**Why this is fine for our scope**: `prefer_builtin` is the right primitive for builtin-to-custom fallback (web search, code execution, etc.). Deferred tool loading will need an additional mechanism (likely `defer_loading: bool` on `ToolDefinition` + model-level list restructuring), which is a separate feature. These are complementary, not conflicting — both follow the same architectural pattern of model-level resolution in `prepare_request`.

**Forward compatibility**: the `capabilities=[WebSearch()]` API will compose both: `WebSearch.get_toolset()` returns function tools with `prefer_builtin` set, `WebSearch.get_builtin_tools()` returns the builtin. For deferred loading, `SearchableToolset` will add its own fields. Our `prepare_request` swap logic is additive and won't conflict with future deferred loading logic.

## Deferred to follow-up PRs

- `AbstractBuiltinTool(fallback=...)` — builtin carries its own fallback
- `MCPServer`/`FastMCPToolset` `prefer_builtin` — toolset-level support
- `TavilySearchTool(prefer_builtin=True)` — convenience on common tools
- `BuiltinToolset` — wrapping builtins in the toolset system
- Deferred tool loading / `SearchableToolset` (#4090, #4167)

## Testing

Integration tests via `Agent.run()` using `TestModel`/`FunctionModel` with custom `ModelProfile(supported_builtin_tools=frozenset())` to simulate unsupported builtins. Snapshot `all_messages()` for assertions.

**Cases**:
- Model supports builtin → builtin used, fallback tool removed from request
- Model doesn't support builtin + fallback exists → fallback tool used, no error
- Model doesn't support builtin + no fallback → `UserError` (backward compat)
- `FallbackModel` with mixed-support inner models → each model handles independently
- Callable `prefer_builtin` → dynamic resolution via `RunContext`

**Reference patterns**: `tests/models/test_fallback.py`, `tests/test_builtin_tools.py`

**Also**: `make typecheck`, `make lint`

## Docs/memory

- Placeholder docstrings on new params (per PR flow — defer full docs until after review)
- Update `CLAUDE.local.md` with branch/issue/PR info after creating branch
