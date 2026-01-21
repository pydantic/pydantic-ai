# Code Mode: Prompt Flexibility & Eval Goals

## Completed (Staged)

1. ✅ Move files to `tests/code_mode/` and `demos/code_mode/`
2. ✅ Refactor `code_mode.py` - extract `build_code_mode_prompt()`, add `prompt_builder` param
3. ✅ Public exports in `toolsets/__init__.py`
4. ✅ Create `demos/code_mode/CLAUDE.md`
5. ✅ Create `code-mode-plus-tool-search.spec.md`

## TODO

### 1. MCP Tool Prefix for Name Clash Prevention

**Problem:** MCP tools need prefixing to avoid name clashes with user-defined functions.
- Currently `_get_tool_signature()` uses bare `tool.name`
- If MCP server has `arbitrary_query` and user defines `arbitrary_query`, they clash

**Solution:** Prefix MCP tool signatures with server name (e.g., `logfire__arbitrary_query` or `logfire_arbitrary_query`)

**Questions:**
- How do we detect if a tool comes from MCP vs native?
- What separator? `__` (dunder) or `_` (single underscore)?
- Should the prefix be configurable?

### 2. Update CLAUDE.local.md

Update current state section after changes are finalized.

---

## Decisions (Reference)

- **Callback API**: `prompt_builder(*, signatures: list[str]) -> str` (kwargs-only for future expansion)
- **Prompt parts**: Single `build_code_mode_prompt()` function, users override entirely
- **Smart routing**: Future scope (see spec file)

---

## Open Questions: Handling Unspecified MCP Schemas

Many MCP tools have incomplete JSON schemas (e.g., `items: {}` for arrays, meaning `list[Any]`).
This was discovered when testing with Logfire MCP - their `arbitrary_query` tool returns `list[Any]`.

### Questions

1. **Prompt guidance for `Any` types**: Should the base prompt instruct the LLM to retrieve sample data when it encounters an `Any` type, so it can learn the actual structure?

2. **Automatic schema discovery**: Should CodeModeToolset offer to automatically call tools with sample inputs to discover actual return structures?

3. **User opt-in**: If we add this, should it be:
   - Always on (automatic)
   - Opt-in via parameter (e.g., `discover_schemas=True`)
   - Prompt-only (just tell LLM to do it, don't automate)

4. **Schema caching**: If we discover schemas at runtime, should we cache them for the session?

### Context

- Logfire MCP declares `{result: list[Any]}` but doesn't specify array item structure
- LLM generates code assuming structure, which may fail at runtime
- Sample call could reveal: `[{span_name: str, duration: float, ...}]`

### Potential API

```python
@dataclass
class IntrospectUnknownTypes:
    """Config for runtime schema discovery of `Any` types."""
    enabled: bool = True
    max_depth: int = 3  # how deep to recurse into nested structures
    sample_array_items: int = 1  # how many array items to inspect
    cache: bool = True  # cache discovered schemas for session

# Usage
code_toolset = CodeModeToolset(
    wrapped=mcp_server,
    introspect_unknown_types=IntrospectUnknownTypes(max_depth=2),
)
```
