Implement Phase 3 of PLAN.md — Local Text Editor + Apply Patch.

Read PLAN.md §5.3, §5.4, §5.8, §5.9, and §14 Phase 3 for full spec. Read PROGRESS.md
for Phase 1 and Phase 2 context. Read the coding guidelines in agent_docs/index.md and the
directory-specific AGENTS.md files.

Summary of what Phase 3 adds:

1. TextEditorToolset in new file `toolsets/text_editor.py`:
   - `TextEditorCommand` discriminated union of TypedDicts (`ViewCommand`, `StrReplaceCommand`,
     `CreateCommand`, `InsertCommand`) for type-safe command dispatch
   - `TextEditorOutput` dataclass (`output: str`, `success: bool`)
   - `TextEditorExecuteFunc` callback type
   - `TextEditorToolset(AbstractToolset)` with `execute` callback, `max_characters`,
     `tool_name='str_replace_based_edit_tool'`
   - `get_tools()` emits `ToolDefinition` with `native_definition=TextEditorNativeDefinition(max_characters=...)`
   - `call_tool()` dispatches to the execute callback

2. ApplyPatchToolset in new file `toolsets/apply_patch.py`:
   - `ApplyPatchOperation` dataclass (`operation_type`, `path`, `diff`, `content`)
   - `ApplyPatchOutput` dataclass (`status: Literal['completed', 'failed']`, `output`)
   - `ApplyPatchExecuteFunc` callback type
   - `ApplyPatchToolset(AbstractToolset)` with `execute` callback
   - `get_tools()` emits `ToolDefinition` with `native_definition=ApplyPatchNativeDefinition()`
   - `call_tool()` dispatches to the execute callback

3. Anthropic adapter (`models/anthropic.py`):
   - In `_get_tools()`: when `native_definition` is `TextEditorNativeDefinition` and
     `supports_native_text_editor_tool`, emit `BetaToolTextEditor20250728Param(
     type='text_editor_20250728', name='str_replace_based_edit_tool',
     max_characters=native_def.max_characters)`
   - Register name mapping: `'str_replace_based_edit_tool'` → toolset's `tool_name`
   - Response/streaming: already handled — `BetaToolUseBlock` with native name is
     mapped back via `_native_tool_names` dict (same Phase 2 mechanism)
   - The `BetaToolTextEditor20250728Param` import already exists in the try block but
     needs to be added

4. OpenAI adapter (`models/openai.py`):
   - In `_get_tools()`: when `native_definition` is `ApplyPatchNativeDefinition` and
     `supports_native_apply_patch_tool`, emit `ApplyPatchToolParam(type='apply_patch')`
   - Register name mapping: `'apply_patch'` → toolset's `tool_name`
   - In `_process_response()`: handle `ResponseApplyPatchToolCall` — when the tool
     name is in `_native_tool_names`, create `ToolCallPart` with operation info
   - In streaming: handle `ResponseApplyPatchToolCall` in both
     `ResponseOutputItemAddedEvent` and `ResponseOutputItemDoneEvent`
   - Round-trip in `_map_messages()`: `ToolCallPart` for apply_patch → emit
     `ApplyPatchCallParam` input item; `ToolReturnPart` → `ApplyPatchCallOutputParam`
   - Import `ApplyPatchToolParam`, `ResponseApplyPatchToolCall`,
     `ResponseApplyPatchToolCallOutput` from openai SDK

5. Profile flags:
   - `supports_native_text_editor_tool: bool = False` on `ModelProfile`
   - `supports_native_apply_patch_tool: bool = False` on `ModelProfile`
   - Set `supports_native_text_editor_tool=True` in `anthropic_model_profile()`
   - Set `supports_native_apply_patch_tool=True` in `openai_model_profile()`

6. Top-level exports:
   - Add `TextEditorToolset` and `ApplyPatchToolset` to `toolsets/__init__.py` and
     `pydantic_ai/__init__.py` with `__all__` entries

7. Fallback warnings:
   - Same `_warn_native_tool_fallback()` pattern from Phase 2 — already in both adapters,
     just needs the new `isinstance` branches for `TextEditorNativeDefinition` and
     `ApplyPatchNativeDefinition`

8. Tests:
   - Unit tests in `tests/test_shell_toolset.py` (or a new `tests/test_native_toolsets.py`):
     type tests for `TextEditorCommand`, `ApplyPatchOperation`, toolset `get_tools()` and
     `call_tool()` with mock executors
   - VCR cassette integration tests (via `doppler run --`):
     - Anthropic text_editor: non-streaming + streaming (in `test_anthropic.py`)
     - OpenAI apply_patch: non-streaming + streaming (in `test_openai.py`)
   - Fallback warning test for text_editor on OpenAI, apply_patch on Anthropic
   - Snapshot updates for any affected inline snapshots

9. Docs:
   - Update `docs/native-tools.md`: add TextEditorToolset and ApplyPatchToolset sections
     with usage examples, provider support table, and safety patterns
   - Update `docs/api/toolsets.md`: add TextEditorToolset, ApplyPatchToolset, and their
     helper types to API reference

Key architectural notes:

- Phase 2 already established the `NativeToolDefinition` → adapter pattern. Phase 3
  extends it with two new definition kinds: `TextEditorNativeDefinition` (already defined
  in `tools.py`) and `ApplyPatchNativeDefinition` (already defined in `tools.py`).
- Anthropic `text_editor_20250728` returns `BetaToolUseBlock` with
  `name='str_replace_based_edit_tool'` — same response type as regular tools and
  Phase 2 shell, handled by existing `_native_tool_names` mapping.
- OpenAI `apply_patch` returns `ResponseApplyPatchToolCall` — a NEW response type
  that doesn't exist in Phase 1 or 2 handling. The adapter needs new branches in
  `_process_response()` and streaming.
- OpenAI `apply_patch` response has `operation` field (union of `OperationCreateFile`,
  `OperationDeleteFile`, `OperationUpdateFile`) — map to `ApplyPatchOperation`.
- OpenAI returns `environment=None` for local tools (learned in Phase 2) — no
  `ResponseLocalEnvironment` check needed.
- `TextEditorNativeDefinition.max_characters` passes through to
  `BetaToolTextEditor20250728Param.max_characters`.
- `TextEditorToolset` is Anthropic-only for native format; falls back to function tool
  on OpenAI and others.
- `ApplyPatchToolset` is OpenAI-only for native format; falls back to function tool
  on Anthropic and others.

SDK types to use:
- Anthropic: `BetaToolTextEditor20250728Param` (already importable from `anthropic.types.beta`)
- OpenAI: `ApplyPatchToolParam`, `ResponseApplyPatchToolCall`,
  `ResponseApplyPatchToolCallOutput` (from `openai.types.responses`)
- OpenAI input round-trip: check for `ApplyPatchCall` / `ApplyPatchCallOutput` in
  `openai.types.responses.response_input_item_param`

We use jj not git. Use `doppler run --` for API keys when recording cassettes.
