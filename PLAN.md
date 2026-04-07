# Plan: Unify python-signatures-v2 (#4851) + return_schema (#3865)

## Goal

Combine these two PRs into one that delivers complete `FunctionSignature` generation for all tools, using the `return_schema` field on `ToolDefinition` to get return types via the schema-based path — eliminating the need for the function-based `from_function` path entirely.

## Background & Context

This work is infrastructure for **code mode** (`CodeExecutionToolset`), which presents tools to the LLM as Python function signatures. The consumer exists in pydantic-harness (#98) and was previously in this repo (#4153, #4358). The harness PR explicitly lists this as its #1 blocking dependency.

### Who is @DouweM

@DouweM is a core maintainer of Pydantic AI. He is driving this PR to completion after taking it over from an external contributor (@adtyavrdhn) whose original version (#4755) was accidentally merged and reverted.

## How Code Mode Consumes Signatures

Code mode (`CodeExecutionToolset`) in pydantic-harness wraps another toolset and presents all its tools as Python function signatures in a single `run_code` tool description. The key consumer flow:

1. For each wrapped tool, gets `tool_def.function_signature`
2. **Deep copies it** (important — prevents mutation leaks)
3. Sets `sig.name` to a sanitized Python identifier
4. Sets `sig.is_async` based on parallelism mode
5. Calls `dedup_referenced_types(signatures)` across all tools
6. Calls `collect_unique_referenced_types(signatures)` for rendering
7. Renders everything into the `run_code` tool's description:
   ```
   # Available types:
   class User(TypedDict):
       name: str

   # Available functions:
   async def get_user(*, id: int) -> User:
       """Get a user by ID"""
       ...
   ```

**Key implication**: Code mode always overrides `name` and `is_async` on the signature. So staleness of these fields after `replace()` doesn't matter in practice — code mode sets them at render time. The `render()` method's `name`/`description`/`is_async` overrides serve the same purpose for non-code-mode consumers.

## Current State

### This branch (python-signatures-v2, #4851)
- `function_signature.py` module with `FunctionSignature`, `TypeExpr` tree, `from_function`/`from_schema` classmethods
- `ToolDefinition.function_signature` field (stored, computed by `__post_init__` from schema if not provided)
- `FunctionSchema.function_signature` cached_property (function-based, richer types)
- `Tool.tool_def` passes the function-based signature from `FunctionSchema`
- `FunctionSignature.render()` with `name`/`description`/`is_async` overrides for staleness after `replace()`
- Missing: return types in schema-based path (returns `Any`)

### PR #3865 (return_schema)
- Adds `return_schema: ObjectJsonSchema | None` field on `ToolDefinition`
- Adds `include_return_schema: bool | None` field on `ToolDefinition`
- Adds `return_schema` on `FunctionSchema` (generated from return type annotation)
- Adds `ToolReturn[T]` generic for explicit return type control
- Google Gemini native `return_schema` support
- Description injection fallback for models without native support
- ~2600 lines, 20 files, extensive test snapshots
- Has unresolved review feedback from @DouweM on design

## Target Architecture

After unification:

1. **`ToolDefinition.return_schema`** — new field, populated by:
   - Function tools: from `FunctionSchema.return_schema` (inspects return type annotation)
   - MCP tools: from `outputSchema`
   - Direct construction: optional

2. **`ToolDefinition.function_signature`** — stored field, computed by `__post_init__`:
   - Always uses `FunctionSignature.from_schema()` with `return_schema=self.return_schema`
   - No `from_function` path needed — JSON schema + return_schema has all the same info
   - `FunctionSchema.function_signature` can be removed or kept as internal convenience

3. **`FunctionSignature.from_schema()`** — already accepts `return_schema`, converts to proper `TypeExpr` return type

4. **Schema generation** — ensure `use_attribute_docstrings=True` is applied when generating return schemas, so field descriptions flow through

5. **No `original_func` on ToolDefinition**, no function-based path needed

## Key Insight

The `from_function` path exists because `from_schema` lacked return type info. With `return_schema` on `ToolDefinition`, `from_schema` gets the return type too. Pydantic's JSON schema already preserves:
- Real class names in `$defs` (User, Address, etc.)
- Field descriptions (via `Field(description=...)` or `use_attribute_docstrings`)
- Union types, generics, nested models

So `from_schema` + `return_schema` ≈ `from_function` for all practical purposes.

## Design Decisions & Tradeoffs (from extensive iteration)

This section captures the reasoning behind the current design. We went through many iterations to arrive here — future sessions should understand WHY, not just WHAT.

### How `function_signature` should live on `ToolDefinition`

We tried several approaches, each with different tradeoffs:

**1. `cached_property` on ToolDefinition (schema-based only)**
- Clean, lazy, recomputes correctly after `dataclasses.replace()`
- But: loses return type info (JSON schema for tool params doesn't include return types)
- But: loses richer function-based type info

**2. `cached_property` + `__dict__` pre-population hack**
- `Tool.tool_def` pre-populates `td.__dict__['function_signature']` with the function-based signature
- `replace()` creates fresh `__dict__`, falls back to schema-based
- Problem: shared object identity causes mutation leak from `dedup_referenced_types`
- Problem: needed ugly `_use_schema_signature` flag for `Tool.from_schema`
- **Rejected**: too hacky, mutation leak is a real bug

**3. `original_func` field on ToolDefinition + `cached_property` dispatch**
- `cached_property` checks `self.original_func` → `from_function` or `from_schema`
- `replace()` preserves `original_func`, recomputes with new name/description
- Problem: `original_func` (a callable) can't survive `pydantic_core.to_json()` serialization
- Problem: Temporal workflow → activity serialization drops it
- **Viable but fragile**: serialization concerns are real for Temporal users

**4. Stored `function_signature` field + `__post_init__` (CURRENT)**
- `function_signature` is a regular dataclass field, default `None`
- `__post_init__` computes from schema if not provided
- `Tool.tool_def` passes the function-based signature from `FunctionSchema`
- `Tool.from_schema` and MCP tools let `__post_init__` compute from schema
- Problem: after `replace()`, the stored signature has stale name/description
- Solution: `render(body, *, name=, description=, is_async=)` accepts overrides
- Code mode already deepcopies and overrides name/is_async, so staleness is handled
- **Current approach**: simple, explicit, survives serialization

**5. PLANNED: Eliminate `from_function` entirely via `return_schema`**
- With `return_schema` on ToolDefinition (#3865), `from_schema` gets return types
- `__post_init__` always uses `from_schema(return_schema=self.return_schema)`
- No `from_function`, no `original_func`, no dual paths
- `FunctionSchema` generates `return_schema` from the return type annotation
- **Target architecture**: one path, clean, complete

### Why `from_function` exists (and why we want to eliminate it)

`from_function` uses `inspect.signature()` + `get_type_hints()` to build richer signatures than `from_schema`. The advantages:
- **Return types**: `from_function` has the return annotation, `from_schema` doesn't (JSON schema for tool params doesn't include return types)
- **Real type names**: `from_function` gets `User` from `get_type_hints()`, `from_schema` gets them from `$defs` — but these are the same names since Pydantic puts class names in `$defs`
- **Field descriptions**: Both paths get them — Pydantic includes descriptions in JSON schema via `Field(description=...)` and `use_attribute_docstrings=True`

The ONLY meaningful difference is return types. With `return_schema` (#3865), that gap closes. So `from_function` becomes redundant.

**Important for return_schema**: ensure `use_attribute_docstrings=True` is applied when generating the return schema in `_function_schema.py`, so field descriptions in return types flow through to the signature. The parameter schema already does this (line 112: `ConfigDict(title=function.__name__, use_attribute_docstrings=True)`). The return schema generation in #3865 uses `TypeAdapter(schema_type).json_schema()` which inherits the model's own config — so models with `use_attribute_docstrings=True` get descriptions, others don't. This matches `from_function` behavior.

### `FunctionSignature.render()` and staleness

After `dataclasses.replace(td, name='new_name')`, the stored `function_signature` field has the old name. This is a known tradeoff of storing signatures as fields.

Solution: `render()` accepts `name`, `description`, and `is_async` overrides:
```python
sig.render('...', name=td.name, description=td.description, is_async=True)
```

Code mode already deepcopies signatures and overrides `name`/`is_async`, so this fits naturally. The `_UNSET` sentinel on `description` allows distinguishing "not provided" from "explicitly None".

### TypeExpr should not use strings

Early versions used `str` in the `TypeExpr` union (e.g. `'str'`, `'Any'`, `'None'`). @DouweM requested proper types so the data model could render to TypeScript or other languages. Now:
- `SimpleTypeExpr('str')` instead of `'str'`
- `LiteralTypeExpr(['a', 'b'])` instead of `"Literal['a', 'b']"`
- `TypeSignature.__str__()` returns just the name (for type expressions); `render_definition()` for full TypedDict output

### `FunctionSignature` naming (not `python_signature`)

Originally named `python_signature`. Renamed to `function_signature` because the data model is language-agnostic — it could render TypeScript, etc. The module is `function_signature.py` (public, no underscore prefix).

### `FunctionSchema` as the home for function-based signature generation

`FunctionSchema` already has `function`, `name`, `description`, `json_schema`, `is_async`. It's the natural place to build the `FunctionSignature`. Currently has a `cached_property` that calls `FunctionSignature.from_function()`. After unification with #3865, this can be simplified to just providing `return_schema`, and `ToolDefinition.__post_init__` handles the rest.

### Serialization concerns

`ToolDefinition` is serialized via `pydantic_core.to_json()` in tests and by Temporal for workflow → activity communication. Callables (`original_func`) and complex objects (`FunctionSignature`) can't survive this.

- **Callables**: Can't be serialized at all. This is why `original_func` on ToolDefinition was problematic.
- **FunctionSignature as a field**: Goes over the wire. @DouweM accepted this tradeoff ("I do not care about the size of the data going over the wire"). Tests use `pydantic_core.to_json(td, fallback=lambda v: None)` to handle unserializable nested types.
- **After eliminating `from_function`**: `FunctionSignature` is derived entirely from JSON schema fields that DO serialize. So even if the `FunctionSignature` object itself doesn't survive serialization, it can be recomputed from `parameters_json_schema` + `return_schema` on the other side.

### `dedup_referenced_types` mutation concern

`dedup_referenced_types` mutates `TypeSignature.name` in place and replaces `sig.referenced_types`. If the same `FunctionSignature` object is shared (e.g. between `FunctionSchema` cache and `ToolDefinition` field), mutations leak.

Code mode handles this by deepcopying signatures before dedup. The stored-field approach (option 4) means `Tool.tool_def` passes the same object from `FunctionSchema`, so the leak is possible. But since code mode deepcopies, it's safe in practice. After eliminating `from_function`, `ToolDefinition` always computes its own signature in `__post_init__`, so there's no sharing.

### `*args` and `**kwargs` handling

`_build_function_params` skips `**kwargs` (maps to `additionalProperties` in JSON schema) and wraps `*args: T` as `list[T]`, matching `_function_schema.py` behavior. After eliminating `from_function`, this code goes away.

### `functools.partial` unwrapping

`from_function` unwraps `partial` objects to get the original function's type hints. After eliminating `from_function`, this goes away — the schema is generated from the unwrapped function in `_function_schema.py` already.

## Implementation Steps

### Phase 1: Merge #3865 into this branch

1. `git fetch origin` and create a merge of #3865's branch
2. Resolve conflicts in ~13 files (mostly `tools.py`, `_function_schema.py`, test files)
3. Key conflicts:
   - `ToolDefinition` fields: reconcile `function_signature` (ours) with `return_schema`/`include_return_schema` (theirs)
   - `_function_schema.py`: reconcile `FunctionSchema.function_signature` (ours) with `FunctionSchema.return_schema` (theirs)
   - `Tool.tool_def`: reconcile our signature passing with their `return_schema` passing
   - Test snapshots: extensive updates needed

### Phase 2: Wire return_schema into function_signature generation

4. Update `ToolDefinition.__post_init__` to pass `self.return_schema` to `FunctionSignature.from_schema()`
5. Remove the `(self.metadata or {}).get('output_schema')` hack (replaced by `self.return_schema`)
6. Verify MCP tools set `return_schema` from `outputSchema` (done in #3865)

### Phase 3: Eliminate from_function path

7. Remove `FunctionSignature.from_function()` classmethod
8. Remove `FunctionSchema.function_signature` cached_property
9. `Tool.tool_def` no longer passes a pre-computed signature — `__post_init__` handles it
10. Remove the `from_function` helpers: `_collect_referenced_types`, `_build_function_params`, `_annotation_to_type_expr`, `_get_type_name`, etc.
11. Update `function_signature.py` `__all__` to remove `from_function`-only types if any

### Phase 4: Ensure schema quality matches function path

12. Verify `use_attribute_docstrings=True` is applied when generating return schemas in `_function_schema.py`
13. Compare output of `from_schema` vs old `from_function` for representative tools to ensure parity
14. If gaps exist (e.g. return type annotation nuances), address them in `from_schema`

### Phase 5: Address @DouweM's review feedback on #3865

15. Move description injection logic to `Model.prepare_request` (where `profile` is available)
16. Apply `include_return_schema` via `PreparedToolset` in `Agent._get_toolset`
17. MCP tools should NOT default to `include_return_schema=True`
18. Evaluate if `ToolReturn[T]` / `ToolReturnContent` changes are in scope or should be split out

### Phase 6: Tests and cleanup

19. Update all test snapshots (large portion of #3865's diff)
20. Remove tests specific to `from_function` path
21. Add tests verifying return types flow through `return_schema` → `function_signature`
22. Coverage: ensure 100%
23. Verify doc examples pass

## Risk Assessment

- **Merge conflicts**: Extensive but mechanical. The two PRs touch overlapping files.
- **Schema quality**: Need to verify that `from_schema` with `return_schema` produces equivalent output to `from_function`. Key area: field descriptions from docstrings.
- **#3865 review feedback**: Some design changes requested by @DouweM are non-trivial (where description injection lives, `PreparedToolset` integration).
- **Scope**: This is a large combined PR. Consider whether #3865's `ToolReturn[T]` and Google Gemini native support should be split out.

## Files to Modify (estimated)

Core:
- `function_signature.py` — remove `from_function` path, simplify
- `tools.py` — add `return_schema`/`include_return_schema`, simplify `function_signature` generation
- `_function_schema.py` — add `return_schema` generation, remove `function_signature` cached_property
- `_agent_graph.py` — `ToolReturn` handling (from #3865)
- `_tool_manager.py` — description injection (from #3865, may need redesign per review)

Models:
- `models/__init__.py` — return_schema in request building
- `models/google.py` — native return_schema support
- `profiles/__init__.py`, `profiles/google.py` — capability flag

Toolsets:
- `toolsets/function.py` — pass return_schema
- `toolsets/fastmcp.py` — pass return_schema from outputSchema
- `mcp.py` — pass return_schema from outputSchema

Tests:
- `test_function_signature.py` — remove from_function tests, add return_schema tests
- `test_tools.py` — snapshot updates, return_schema tests
- `test_toolsets.py` — snapshot updates
- `test_model_request_parameters.py` — snapshot updates
- `test_gemini.py`, `test_google.py` — native return_schema tests
- `test_logfire.py` — span attribute updates

## Starting Point

Work off the `python-signatures-v2` branch. Cherry-pick or merge `return-schema` branch (#3865), resolve conflicts, then execute the phases above.
