# PLAN: Remote Code Tools — ShellTool Builtin, Background Mode, Native Tool Adapters

> **Issue refs:** #3365 (Anthropic/OpenAI Skills), #3963 (Shell/Bash builtin), #3794 (Text Editor tool)
> **Related PRs:** #4600 (Anthropic Skills draft — absorbed into this plan)
> **Stack:** `continuation-support` -> `skill-support-v2` (this) -> `local-tools`
>
> **Depends on:** `continuation-support` (ContinueRequestNode, ModelResponseState, fallback continuation pinning)

---

## Scope

This change implements **remote/provider-hosted code execution** support, **OpenAI background mode**, and the **model adapter infrastructure** for both remote and local native tools. The continuation infrastructure it builds on is in the parent change (`continuation-support`).

### What's in this change

1. **`ShellTool` builtin** — provider-hosted shell with skills, network policy, container management, file uploads
2. **OpenAI background mode** — async execution with polling for Responses API (produces `suspended` state consumed by `ContinueRequestNode`)
3. **Anthropic `pause_turn` handling** — produces `suspended` state for long-running operations
4. **`NativeToolDefinition` types** — typed discriminated union on `ToolDefinition` for provider-native tool presentation
5. **Model adapter native tool infrastructure** — Anthropic and OpenAI adapters handle both remote and local native tool formats
6. **Profile flags** — `supports_shell_network_policy`, `supports_native_shell_tool`, `supports_native_text_editor_tool`, `supports_native_apply_patch_tool`
7. **`UploadedFile` changes** — `target` field, `part_kind`, `ModelResponsePart` union membership
8. **Toolset base implementations** — `ShellToolset`, `TextEditorToolset`, `ApplyPatchToolset` source files (needed for import resolution; dedicated tests are in the follow-up)

### What's in `continuation-support` (parent)

- `ContinueRequestNode` in agent graph
- `ModelResponseState` type on `ModelResponse` and `StreamedResponse`
- Fallback model continuation pinning
- All fallback continuation tests

### What's in `local-tools` (follow-up)

- `tests/test_shell_toolset.py` — dedicated unit tests for all three toolsets
- VCR cassettes for local toolset integration tests
- `docs/native-tools.md`, `docs/builtin-tools.md`, `docs/api/toolsets.md` — user-facing docs

---

## 1. New Types

### `builtin_tools.py`

- `ShellTool(AbstractBuiltinTool)` — provider-hosted shell with `skills`, `network_policy`
- `SkillReference` — reference to a provider-hosted skill (`skill_id`, `version`, `source`)
- `CodeExecutionNetworkPolicy` — network access policy (`mode`, `allowed_domains`)

### `tools.py` — NativeToolDefinition

- `ShellNativeDefinition`, `TextEditorNativeDefinition`, `ApplyPatchNativeDefinition`
- `NativeToolDefinition` union type
- `native_definition` field on `ToolDefinition`

### `messages.py` — UploadedFile Changes

- `UploadedFileTarget = Literal['message', 'container', 'both']`
- `UploadedFile.target` and `UploadedFile.part_kind` fields
- `UploadedFile` added to `ModelResponsePart` union

---

## 2. Anthropic Adapter

- `ShellTool` -> `BetaCodeExecutionTool20260120Param` (GA)
- Skills -> `BetaSkillParams` in `container.skills` with `skills-2025-10-02` beta
- Container ID persistence via `provider_details['container_id']`
- Response blocks: `BetaBashCodeExecutionToolResultBlock`, `BetaTextEditorCodeExecutionToolResultBlock`
- File outputs -> `UploadedFile` parts
- `UploadedFile(target='container')` -> `BetaContainerUploadBlockParam`
- Native local tools: `bash_20250124`, `text_editor_20250728` emission and name mapping
- Container error retry: auto-recover from expired/missing containers
- `pause_turn` -> `state='suspended'` (consumed by `ContinueRequestNode` from parent)

---

## 3. OpenAI Adapter

- `ShellTool` -> `shell` tool with `container_auto` environment
- Skills via `client.containers.create(skills=[...])`
- Network policy, file mounting, container reuse
- `shell_call` / `shell_call_output` mapping (discriminated by environment type)
- `apply_patch` tool emission and response handling
- **Background mode**: async execution with polling (`_get_continuation_info`, `_responses_retrieve`)
  - Produces `state='suspended'` (consumed by `ContinueRequestNode` from parent)
- Container expired auto-recovery with targeted retry
- New settings: `openai_shell_uploaded_files`, `openai_shell_container`

---

## 4. Validation

- `ShellTool` + `CodeExecutionTool` = `UserError` (mutually exclusive)
- `ShellTool.network_policy` on unsupported provider = `UserError`
- `UploadedFile(target='container')` without `ShellTool` = `UserError`

---

## 5. Capability Matrix

| Feature                | Anthropic                    | OpenAI Responses                          | OpenAI Chat | Others |
| ---------------------- | ---------------------------- | ----------------------------------------- | ----------- | ------ |
| Remote shell (hosted)  | `code_execution_20260120`    | `shell` with `container_auto`             | No          | No     |
| Skills                 | `skills-2025-10-02` beta     | Via containers                            | No          | No     |
| Local bash (native)    | `bash_20250124`              | No                                        | No          | No     |
| Local shell (native)   | No                           | `shell` with `local` env                  | No          | No     |
| Local text editor      | `text_editor_20250728`       | No                                        | No          | No     |
| Local apply_patch      | No                           | `apply_patch` (GPT-5.1+)                  | No          | No     |
| Function tool fallback | Yes                          | Yes                                       | Yes         | Yes    |
