# Typecheck Fixes Summary

After `uv sync --all-extras --all-packages` and library upgrades (groq 1.0.0, pytest 9.0.2), we have **134 remaining typecheck errors** to fix.

## Error Breakdown

- **83 errors**: `reportUnknownVariableType` - `list[Unknown]`, `dict[Unknown, Unknown]`, `set[Unknown]`
- **21 errors**: `reportUnknownMemberType` - accessing unknown types on library objects
- **15 errors**: `reportUnknownArgumentType` - passing unknown types as arguments
- **8 errors**: `reportAttributeAccessIssue` - MCP content type attribute access
- **4 errors**: `reportCallIssue` - mistral SDK breaking changes
- **2 errors**: `reportDeprecated` - MCP deprecated `streamablehttp_client`
- **1 error**: `reportArgumentType` - fallback exceptions tuple

## Root Cause

Using `field(default_factory=list)` without explicit type parameters in strict pyright mode causes `list[Unknown]` inference. The fix is to use `lambda: []` or `lambda: list[T]()` to preserve type information.

## Fixes Needed

### 1. MCP Errors (14 errors)

**Files**: `pydantic_ai_slim/pydantic_ai/mcp.py`, `pydantic_ai_slim/pydantic_ai/_mcp.py`

**Issue 1**: Deprecated `streamablehttp_client` → use `streamable_http_client`
- Line 35 in `mcp.py`: Change import
- Line 1285 in `mcp.py`: Change usage

**Issue 2**: Attribute access on MCP content union types
Lines 34-123 in `_mcp.py` - need type narrowing with `isinstance()` checks or type ignores

### 2. Dataclass Field Defaults (83+ errors)

**Pattern**: Change `field(default_factory=list)` to `field(default_factory=lambda: [])`

**Core Files**:
- `pydantic_ai_slim/pydantic_ai/_agent_graph.py` - lines 88, 188, 191, 192
- `pydantic_ai_slim/pydantic_ai/_function_schema.py` - line 45
- `pydantic_ai_slim/pydantic_ai/_parts_manager.py` - lines 60, 62
- `pydantic_ai_slim/pydantic_ai/_run_context.py` - lines 39, 49
- `pydantic_ai_slim/pydantic_ai/_tool_manager.py` - line 36
- `pydantic_ai_slim/pydantic_ai/format_prompt.py` - lines 85, 87, 89
- `pydantic_ai_slim/pydantic_ai/models/__init__.py` - lines 540, 541, 545
- `pydantic_ai_slim/pydantic_ai/models/openai.py` - lines 819-821
- `pydantic_ai_slim/pydantic_ai/models/openrouter.py` - line 577
- `pydantic_ai_slim/pydantic_ai/direct.py` - line 290 (Queue type)

**Example Files**:
- `examples/pydantic_ai_examples/ag_ui/api/agentic_generative_ui.py` - line 30
- `examples/pydantic_ai_examples/ag_ui/api/shared_state.py` - lines 64, 71
- `examples/pydantic_ai_examples/question_graph.py` - lines 35, 36
- `examples/pydantic_ai_examples/slack_lead_qualifier/modal.py` - **FIXED** (type ignores added)

**Test Files**: ~30 test files with same pattern

### 3. Model Implementation Errors (8 errors)

**openai.py** (5 errors):
- Lines 819-821: Change `field(default_factory=list)` → `field(default_factory=lambda: [])`
- Lines 1832, 1840, 1848: Walrus operator with `cast()` needs `# type: ignore[reportUnknownMemberType]`

**openrouter.py** (1 error):
- Line 577: Change `field(default_factory=list)` → `field(default_factory=lambda: [])`

**outlines.py** (1 error):
- Line 59: Add `# type: ignore[reportUnknownVariableType]` to `from_transformers` import

**fallback.py** (1 error):
- Line 52: Add `# type: ignore[reportUnknownArgumentType]`

### 4. Output Module Generic Type Issues (27 errors)

**File**: `pydantic_ai_slim/pydantic_ai/_output.py`

Complex covariant generic type handling where type checker can't narrow `OutputSpec[T]` parameters through runtime checks.

**Solution**: Add `# pyright: ignore[reportUnknownVariableType]` or `# pyright: ignore[reportUnknownArgumentType]` at:
- Lines 252, 256, 277, 281 (output detection)
- Lines 310, 312 (append operations)
- Lines 909, 915 (ObjectOutputProcessor)
- Lines 1002-1003, 1010, 1020-1021, 1028, 1030, 1034 (recursive type flattening)

### 5. Utils Module (3 errors)

**File**: `pydantic_ai_slim/pydantic_ai/_utils.py`

- Line 223: `Task[Unknown]` - add `# pyright: ignore[reportArgumentType]`
- Line 434: `dict[Unknown, Unknown]` - add `# pyright: ignore[reportUnknownVariableType]`

### 6. Mistral SDK Breaking Changes (4 errors)

**File**: `tests/models/test_mistral.py`

Lines 2193, 2195, 2208, 2210 - `ModelHTTPError` constructor changed. Need to check mistral SDK version and update test calls.

## Implementation Strategy

### Option 1: Systematic File-by-File Fixes

1. Fix MCP deprecated imports (2 files, simple)
2. Fix model implementations (4 files, straightforward)
3. Fix core dataclass fields (10 files, repetitive but clear)
4. Fix output module generic types (1 file, add type ignores)
5. Fix utils module (1 file, add type ignores)
6. Fix all test files (batch operation, same pattern)
7. Fix example files (5 files)
8. Fix mistral tests (investigate SDK changes)

### Option 2: Automated Script

Create a Python script to:
1. Read each file
2. Apply regex replacements for common patterns
3. Handle edge cases manually

## Estimated Scope

- **Simple fixes**: ~100 instances of `field(default_factory=list)` → `field(default_factory=lambda: [])`
- **Type ignores**: ~40 instances where static analysis is insufficient
- **MCP fixes**: 2 simple renames
- **Mistral SDK**: Investigate breaking changes and update 4 test lines

Total: ~140-150 individual edits across 50+ files.

## Next Steps

Given the scope, recommend:
1. Create and run an automated script for the repetitive `default_factory` pattern
2. Manually fix the special cases (MCP, type ignores, mistral)
3. Run `make typecheck` to verify
4. Run `make test` to ensure no regressions

Would you like me to:
- **A)** Create an automated script to fix the bulk of these systematically?
- **B)** Continue with manual fixes file by file?
- **C)** Provide you with specific sed/perl commands to run?
