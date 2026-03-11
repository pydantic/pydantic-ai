# Provider-Native Shell Tools — Implementation Progress

> Tracking doc for [PLAN.md](PLAN.md) implementation.
> Issue refs: #3365, #3963, #3794 | PRs: #4600

---

## Phase 1: Remote Shell + Skills — COMPLETE

All items from PLAN.md §4 and §14 Phase 1 are implemented and verified.

### Per-Phase Checklist

- [x] `__all__` exports; public types re-exported from `pydantic_ai/__init__.py`
- [x] `kw_only=True` on all new dataclasses (matches existing builtin tool pattern)
- [x] Private helpers prefixed with `_` and excluded from `__all__`
- [x] `make format` and `make typecheck` pass (0 errors)
- [x] `make lint` passes (all checks passed)
- [ ] `make testcov` passes with 100% coverage (not yet verified — new adapter code paths need cassette coverage)
- [x] Docs updated (`docs/builtin-tools.md` ShellTool section)
- [x] Docstrings on all public types and methods
- [ ] PR created with template and issue refs
- [x] VCR cassettes recorded (4 cassettes: Anthropic + OpenAI, streaming + non-streaming)

### Deliverables

| Item | File(s) | Status |
|------|---------|--------|
| `ShellTool` builtin | `builtin_tools.py` | Done |
| `SkillReference` type | `builtin_tools.py` | Done |
| `CodeExecutionNetworkPolicy` type | `builtin_tools.py` | Done |
| `UploadedFile.target` field | `messages.py` | Done |
| `UploadedFileTarget` type alias | `messages.py`, `__init__.py` | Done |
| `supports_shell_network_policy` profile flag | `profiles/__init__.py` | Done |
| OpenAI profile flag set | `profiles/openai.py` | Done |
| Anthropic adapter: ShellTool → 20260120 | `models/anthropic.py` | Done |
| Anthropic: skills beta, container uploads | `models/anthropic.py` | Done |
| Anthropic: bash/text_editor result blocks | `models/anthropic.py` | Done |
| Anthropic: round-trip + streaming | `models/anthropic.py` | Done |
| OpenAI adapter: ShellTool → container_auto | `models/openai.py` | Done |
| OpenAI: skills, network_policy, file mounting | `models/openai.py` | Done |
| OpenAI: shell_call/shell_call_output mapping | `models/openai.py` | Done |
| OpenAI: round-trip + streaming | `models/openai.py` | Done |
| OpenAI: `openai_shell_uploaded_files` setting | `models/openai.py` | Done |
| OpenAI: `openai_shell_container` setting | `models/openai.py` | Done |
| Validation: ShellTool + CodeExecutionTool mutual exclusion | Both adapters | Done |
| Validation: container target without ShellTool | Both adapters | Done |
| Validation: network_policy on unsupported provider | `models/__init__.py` | Done |
| Top-level exports | `__init__.py` | Done |
| Docs: ShellTool section in builtin-tools.md | `docs/builtin-tools.md` | Done |
| Unit tests (types, serialization, validation) | `tests/test_builtin_tools.py`, `tests/test_messages.py` | Done |
| VCR cassettes (Anthropic non-stream + stream) | `tests/models/cassettes/test_anthropic/` | Done |
| VCR cassettes (OpenAI non-stream + stream) | `tests/models/cassettes/test_openai/` | Done |

### Test Summary

- 639 tests pass (16 new: 12 unit + 4 integration)
- Cassettes recorded with `claude-sonnet-4-6` (Anthropic) and `gpt-5.4` (OpenAI)

### SDK Versions at Time of Implementation

- `anthropic`: 0.80.0
- `openai`: 2.26.0

### Notes

- Used `BetaCodeExecutionTool20260120Param` (GA) for Anthropic — no beta header needed at HTTP level, but SDK still places it under `beta` module
- `UploadedFile.part_kind` from PLAN.md §4.2 was not needed — `UploadedFile` already has `kind: Literal['uploaded-file']` which serves the same purpose in the `ModelResponsePart` discriminated union
- `anthropic_container` setting already existed in the codebase prior to Phase 1

---

## Phase 2: Local Shell — NOT STARTED

> PLAN.md §5 and §14 Phase 2

### Scope

1. `NativeToolDefinition` types (`ShellNativeDefinition`, `TextEditorNativeDefinition`, `ApplyPatchNativeDefinition`) + union
2. `native_definition: NativeToolDefinition | None` field on `ToolDefinition`
3. `ShellToolset` + `ShellExecutor` protocol + `_LocalShellExecutor` in `toolsets/shell.py`
4. Anthropic adapter: `bash_20250124` emission for native shell ToolDefinition
5. OpenAI adapter: `shell` with `environment: {type: "local"}` emission
6. Profile flag: `supports_native_shell_tool`
7. Fallback warning on unsupported providers
8. Tests (VCR cassettes, fallback unit test) + docs (`native-tools.md`)

### Key Architectural Note

- Phase 1 `ShellTool` = **builtin** (remote, server-executed, `AbstractBuiltinTool`)
- Phase 2 `ShellToolset` = **toolset** (local, client-executed, `AbstractToolset`)
- OpenAI uses the same `shell` tool type for both — differentiated by `environment` type (`container_auto` vs `local`)
- Streaming constructors need a native-name-to-toolset-name lookup dict passed through

---

## Phase 3: Local Text Editor + Apply Patch — NOT STARTED

> PLAN.md §5.3, §5.4, and §14 Phase 3

### Scope

1. `TextEditorToolset` with `TextEditorCommand` union in `toolsets/text_editor.py`
2. `ApplyPatchToolset` with `ApplyPatchOperation` in `toolsets/apply_patch.py`
3. Anthropic adapter: `text_editor_20250728`
4. OpenAI adapter: `apply_patch`
5. Profile flags: `supports_native_text_editor_tool`, `supports_native_apply_patch_tool`
6. Tests + docs

---

## Future Work (Out of Scope)

- Execution Environment Integration (post-PR #4393)
- Capabilities Layer (#4303)
