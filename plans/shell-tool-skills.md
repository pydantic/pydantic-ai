# Plan: Provider-Native Shell/Skills Support (`ShellTool`, `SkillReference`)

> **Status:** Draft for maintainer discussion
> **Branch:** `anthropic-skills-support`
> **Related:** PR #4393 (Execution Environments), PR #4153 (CodeExecutionToolset), PR #4233 (Traits/Capabilities Research), Issue #4303 (Capabilities Abstraction)

## Problem

Anthropic and OpenAI both now offer provider-hosted "workspace" execution environments that go beyond simple code interpreters:

- **Anthropic** `code_execution_20250825` beta: container-based execution with provider-hosted skills, bash tool, text editor tool, and `container_upload` for file mounting.
- **OpenAI Responses** `shell` tool: hosted shell with skills, network policy, `container_auto` environments, and file mounting via `file_ids`.

Pydantic AI's existing `CodeExecutionTool` maps to the simpler sandbox (Anthropic `code_execution_20250522`, OpenAI `code_interpreter`). There is no way for users to access the newer workspace-style features through the builtin tool abstraction.

## Proposed Solution

Introduce a new `ShellTool` builtin tool and supporting types that provide a **provider-agnostic interface** to these workspace execution environments.

### New Public API

```python
# pydantic_ai/builtin_tools.py

@dataclass
class SkillReference:
    skill_id: str
    version: str | int | None = None
    source: Literal['custom', 'provider'] = 'custom'

@dataclass
class CodeExecutionNetworkPolicy:
    mode: Literal['disabled', 'allowlist']
    allowed_domains: Sequence[str] = ()

@dataclass
class ShellTool(AbstractBuiltinTool):
    skills: Sequence[SkillReference] = ()
    network_policy: CodeExecutionNetworkPolicy | None = None
    kind: str = 'shell'
```

`ShellTool` is mutually exclusive with `CodeExecutionTool` (validated at the model layer).

### Provider Mapping

| `ShellTool` field | Anthropic | OpenAI Responses |
|---|---|---|
| `skills` | `container.skills` (beta `skills-2025-10-02`) | `environment.skills` on `FunctionShellToolParam` |
| `network_policy` | N/A (raises if set) | `environment.network_policy` on `ContainerAutoParam` |
| File mounting | `container_upload` block via `UploadedFile(target='container')` | `environment.file_ids` on `ContainerAutoParam` |
| Container reuse | `anthropic_container` setting (auto-extracts `container_id` from history) | `openai_shell_container` setting (`container_reference` env) or implicit via message history round-tripping |

### Message History

- Anthropic's new `BetaBashCodeExecutionToolResultBlock` and `BetaTextEditorCodeExecutionToolResultBlock` are mapped to `BuiltinToolCallPart` / `BuiltinToolReturnPart` with tool names `bash_code_execution` and `text_editor_code_execution`, and round-tripped correctly.
- OpenAI's `ResponseFunctionShellToolCall` / `ResponseFunctionShellToolCallOutput` are mapped to `BuiltinToolCallPart` / `BuiltinToolReturnPart` with tool name `shell`.
- Both streaming and non-streaming paths are updated.

### `UploadedFile` Changes

`UploadedFile` gains a `target` field:

```python
UploadedFileTarget = Literal['message', 'container', 'both']
```

- `'message'` (default, backward compatible): file sent as model-visible content.
- `'container'`: file mounted into the execution environment only.
- `'both'`: both message content and container mount.

`UploadedFile` also gains `part_kind` so it can appear in `ModelResponsePart` (discriminated union) when returned by provider tool results that include file references.

### OpenAI-Specific

- `OpenAIResponsesModelSettings.openai_shell_uploaded_files`: convenience for mounting files into hosted shell without adding them to the prompt.
- `OpenAIResponsesModelSettings.openai_shell_container`: explicit container reuse across turns. Accepts a container ID string to use `container_reference`, `False` to force a fresh container, or `None` (default) for standard `container_auto` behavior. Cannot be combined with skills, network policy, or uploaded files (which require `container_auto`).
- Raw `shell` tools in `openai_builtin_tools` are auto-normalized (default `container_auto` environment).
- `_OpenAICodeExecutionContext` tracks whether the request uses shell transport and collects container-target `UploadedFile` references to merge into `environment.file_ids`.

### Vercel AI / UI

- `UploadedFile` with `target='container'` is skipped in UI rendering (container-only, not user-visible).
- `UploadedFile` round-trips `target` through `provider_metadata`.
- `FileChunk` gains optional `provider_metadata` to carry `UploadedFile` context.

### Model Profile

- `ModelProfile.supports_shell_network_policy: bool` added (default `False`); OpenAI Responses sets it to `True`.

## Relationship to PR #4393 (Execution Environments)

PR #4393 introduces `ExecutionEnvironment` ABC + `ExecutionEnvironmentToolset` — a **framework-level** abstraction that wraps local/Docker/memory environments and exposes `shell`, `read_file`, `write_file`, `replace_str` as regular function tools. It does **not** touch `builtin_tools.py`, model files, or messages.

This plan is **complementary, not overlapping**:

| Concern | This plan (`ShellTool`) | PR #4393 (`ExecutionEnvironmentToolset`) |
|---|---|---|
| Where code runs | Provider-hosted (Anthropic container, OpenAI hosted shell) | Framework-managed (local, Docker, memory) |
| How tools appear | Builtin tools (provider-native, server-side) | Function tools (framework-managed, client-side) |
| Files touched | `builtin_tools.py`, `models/anthropic.py`, `models/openai.py`, `messages.py`, `ui/` | `environments/`, `toolsets/`, `messages.py` (minor) |
| `messages.py` overlap | Adds `UploadedFile.target`, `UploadedFile.part_kind` | Adds `infer_media_type_from_path()` utility |

The only file both touch is `messages.py`, and the changes are to different parts. The `UploadedFile.target` field added here could eventually be useful to `ExecutionEnvironmentToolset` too, but there's no conflict.

## Scope of Changes

- `pydantic_ai_slim/pydantic_ai/builtin_tools.py` — `ShellTool`, `SkillReference`, `CodeExecutionNetworkPolicy`
- `pydantic_ai_slim/pydantic_ai/__init__.py` — re-exports
- `pydantic_ai_slim/pydantic_ai/models/__init__.py` — validation (`ShellTool` + `CodeExecutionTool` mutual exclusion, network policy support check)
- `pydantic_ai_slim/pydantic_ai/models/anthropic.py` — `code_execution_20250825` beta, skills, bash/text-editor tool result mapping, container upload blocks, `UploadedFile` in response parts
- `pydantic_ai_slim/pydantic_ai/models/openai.py` — `FunctionShellToolParam`, shell tool synthesis, `_OpenAICodeExecutionContext`, file mounting, shell call/output mapping
- `pydantic_ai_slim/pydantic_ai/messages.py` — `UploadedFile.target`, `UploadedFile.part_kind`, `UploadedFileTarget`
- `pydantic_ai_slim/pydantic_ai/profiles/__init__.py` — `supports_shell_network_policy`
- `pydantic_ai_slim/pydantic_ai/_agent_graph.py` — handle `UploadedFile` in response part iteration
- `pydantic_ai_slim/pydantic_ai/result.py` — no functional change (already handles `UploadedFile`)
- `pydantic_ai_slim/pydantic_ai/ui/` — `UploadedFile` target-aware rendering and round-tripping
- `docs/` — builtin-tools, input, anthropic, openai docs updated
- `tests/` — anthropic, openai, openai_responses, builtin_tools, vercel AI, model_request_parameters

## Relationship to Capabilities/Traits v2 (PR #4233, Issue #4303)

DouweM's proposed "capabilities" abstraction (née "traits") is the planned v2 unification of tools, instructions, hooks, guardrails, history processors, and model middleware into a single composable `Capability` class. The target API from issue #4303:

```python
Agent(
    'gateway/anthropic:claude-opus-4-6',
    capabilities=[
        FileSystem(), Shell(), WebSearch(), WebFetch(),
        Skills.from_directory('.jak/skills'),
        # ...
    ],
).to_tui()
```

The research report (PR #4233) explicitly names `ShellTrait` and `FileSystemTrait` as planned built-in capabilities, and notes that existing `AbstractBuiltinTool` subclasses are "a natural fit" for the capabilities abstraction — they could implement the `Capability` interface alongside their current role.

### How this plan fits

This plan's `ShellTool` is **the provider-native builtin tool layer** that a future `Shell` capability would delegate to when the model supports hosted execution (Anthropic containers, OpenAI hosted shell). The capability would add:

- **Intelligent fallback**: Use provider-native `ShellTool` when available, fall back to `ExecutionEnvironmentToolset` (PR #4393) with local/Docker environments when not
- **Instructions**: Shell usage guidance, working directory context
- **Hooks**: `before_tool_call` for destructive command approval (the `ApprovalTrait` pattern)
- **History processing**: Compaction of verbose shell output

The layering is:

```
Shell (Capability, v2)
├── delegates to ShellTool (this plan) when provider supports hosted execution
├── delegates to ExecutionEnvironmentToolset (PR #4393) for local/Docker
├── adds instructions, hooks, guardrails on top
└── composes with other capabilities (Approval, Compaction, etc.)
```

Similarly, `SkillReference` maps naturally to the `SkillsTrait` concept — provider-hosted skills are one source of skills alongside filesystem-based skill definitions.

### Design considerations for forward compatibility

1. **`ShellTool` should remain a thin builtin tool**, not grow into a capability itself. It's the provider-native plumbing; the capability layer adds the intelligence.

2. **`SkillReference` and `CodeExecutionNetworkPolicy`** are provider-configuration types that will be consumed by capabilities but don't need to become capabilities themselves.

3. **The `conflicts_with` pattern** from the traits proposal (e.g., `ShellTrait` conflicts with non-sandboxed execution) maps well to the existing `CodeExecutionTool` / `ShellTool` mutual exclusion validation. When capabilities land, this validation can move from the model layer into the capability composition runtime.

4. **`UploadedFile.target`** is a message-level concern that sits below both the builtin tool and capability layers — it should be stable regardless of how the capability abstraction evolves.

## Open Questions for Maintainers

1. **Naming**: `ShellTool` vs `WorkspaceTool` vs `HostedShellTool`? Current name matches OpenAI's terminology; Anthropic calls it "code execution" with different tool types.

2. **`CodeExecutionTool` / `ShellTool` mutual exclusion**: Currently enforced at the model validation layer. Should they be allowed together (e.g., `code_interpreter` + `shell` on OpenAI)?

3. **`UploadedFile.target`**: Is `Literal['message', 'container', 'both']` the right shape? Could this be a `set` instead for extensibility?

4. **`SkillReference.source`**: Anthropic uses `'custom'` vs `'provider'`; OpenAI currently ignores it. Should this be Anthropic-specific or kept cross-provider?

5. **Network policy**: Currently only OpenAI supports it. Should we gate it differently or let it be a silent no-op on unsupported providers instead of raising?

6. **Anthropic tool name constants**: `bash_code_execution` / `text_editor_code_execution` are derived from the Anthropic SDK types. Are these stable enough for message history, or should we normalize to `shell`?

7. **Does this need to wait for or coordinate with PR #4393?** The changes are complementary but if `ExecutionEnvironmentToolset` is landing soon, we may want to align on `UploadedFile.target` semantics.

8. **Capabilities v2 timing**: Issue #4303 has a 2026-03 milestone. Should `ShellTool` land as a standalone builtin tool now (and later be wrapped by a `Shell` capability), or should it wait for the capabilities abstraction so it can be designed as a capability from the start? The current approach (thin builtin tool) seems forward-compatible with either path, but maintainer input on the intended layering would help.
