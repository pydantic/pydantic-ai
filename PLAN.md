# PLAN: Provider-Native Shell Tools — Capability-Driven Execution with Skills

> **Issue refs:** #3365 (Anthropic/OpenAI Skills), #3963 (Shell/Bash builtin), #3794 (Text Editor tool)
> **Related PRs:** #4393 (Execution Environments), #4505 (ShellTool PLAN), #4513 (TextEditorTool builtin), #4600 (Anthropic Skills draft — absorbed into this plan)
> **Upstream vision:** #4303 (Capabilities abstraction), #4050 (Code Mode)
> **In-flight work:** Capabilities PR (Douwe, pending), unified built-in tool call/return parts PR, built-in tool fallback issue (`prefers_builtin`), tool search deferred loading
>
> **Discussion log:** Architecture reviewed with Douwe Maan and David Sanchez on 2026-03-13. Key decisions: capability-driven routing (not user-picks), local-by-default for data privacy, memory tool pattern rejected as anti-pattern, capability hooks for built-in ↔ tool call bridging.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background & Landscape](#2-background--landscape)
3. [Architecture: Capability-Driven Routing](#3-architecture-capability-driven-routing)
4. [Implementation Layer: Remote Shell — `ShellTool` Builtin](#4-implementation-layer-remote-shell--shelltool-builtin)
5. [Implementation Layer: Local Shell — Native Toolsets](#5-implementation-layer-local-shell--native-toolsets)
6. [Provider Adapter Changes](#6-provider-adapter-changes)
7. [Agent Loop & Message Changes](#7-agent-loop--message-changes)
8. [Error Handling](#8-error-handling)
9. [Security Considerations](#9-security-considerations)
10. [Relationship to PR #4393](#10-relationship-to-pr-4393)
11. [Relationship to Other PRs & In-Flight Work](#11-relationship-to-other-prs--in-flight-work)
12. [Test Strategy](#12-test-strategy)
13. [Documentation](#13-documentation)
14. [Rollout & Phasing](#14-rollout--phasing)
15. [Open Questions](#15-open-questions)

---

## 1. Problem Statement

Modern LLMs ship with **native tool harnesses** for shell execution and text editing. These models are trained on their provider's exact tool schemas and perform significantly worse with generic function-tool equivalents — a finding confirmed by both Anthropic and OpenAI at developer panel talks and validated by multiple third-party providers building on these APIs.

Pydantic AI currently cannot:

1. **Use provider-hosted shell/workspace environments** — Anthropic's `code_execution_20250825` (container-based shell with skills) and OpenAI's hosted `shell` (`FunctionShellToolParam` with `container_auto`) are inaccessible.
2. **Use provider-native local tools** — Anthropic's `bash_20250124` raises `NotImplementedError`; OpenAI's `shell` (local mode) and `apply_patch` are silently dropped.
3. **Use provider-native text editors** — Anthropic's `text_editor_20250728` is unsupported.
4. **Access provider-hosted skills** — Both Anthropic and OpenAI now support skills (packaged tool environments) that run inside hosted containers.

### Why This Needs Two Tracks

These capabilities fall into two fundamentally different execution models:


|                         | Remote (Provider-Hosted)                                                                                                                                                               | Local (Client-Executed)                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Who runs it**         | Provider's container infrastructure                                                                                                                                                    | Developer's local machine / Docker                                                              |
| **Pydantic AI pattern** | `AbstractBuiltinTool` (server-executed)                                                                                                                                                | `AbstractToolset` (client-executed)                                                             |
| **Skills support**      | Yes — skills run in the container                                                                                                                                                      | OpenAI: Yes (`LocalEnvironmentParam` has `skills` field); Anthropic: No                         |
| **Anthropic**           | `code_execution_20260120` (GA) / `code_execution_20250825` (beta) — provides `bash_code_execution` + `text_editor_code_execution` server-side sub-tools, plus `skills-2025-10-02` beta | `bash_20250124`, `text_editor_20250728` (fully client-executed, separate from hosted container) |
| **OpenAI**              | `shell` with `container_auto` / `container_reference` (server-executed)                                                                                                                | `shell` with `environment: {type: "local"}` (client-executed, same tool type!), `apply_patch`   |
| **Use case**            | "Let the provider run code for me"                                                                                                                                                     | "I want to run shell commands locally with optimal model performance"                           |


**Important architectural note for Anthropic**: The hosted `bash_code_execution` (runs in Anthropic's container, returns `server_tool_use` blocks) is **completely separate** from the local `bash_20250124` (runs on client's machine, returns `tool_use` blocks). State is NOT shared between them. A model can use both simultaneously — they appear as two different execution environments.

### Why Tool Names and Parameters Matter

These models are **specifically fine-tuned** on their own tool harnesses:


| Provider  | Local Shell                                                                                                     | Local Text Editor                                                        | Local Patch                                 | Remote Shell (Hosted)                                                                                                                              |
| --------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Anthropic | `bash` (`bash_20250124`) — client-executed, separate tool type                                                  | `str_replace_based_edit_tool` (`text_editor_20250728`) — client-executed | N/A                                         | `code_execution_20260120` (GA) / `code_execution_20250825` (beta) — server-executed `bash_code_execution` + `text_editor_code_execution` sub-tools |
| OpenAI    | `shell` with `environment: {type: "local"}` — client-executed (same `shell_call`/`shell_call_output` as hosted) | N/A                                                                      | `apply_patch` (V4A diffs) — client-executed | `shell` with `container_auto`/`container_reference` — server-executed (same tool type as local!)                                                   |


Using a generic `execute_command` function tool instead of `bash` with `command: str` causes Claude to produce less reliable outputs. The same logical capability must be sent in the exact format each provider expects.

---

## 2. Background & Landscape

### 2.1 Anthropic Native Tools

#### Bash Tool (Client-Executed)

```json
{"type": "bash_20250124", "name": "bash"}
```

- **Input**: `{"command": "ls -la"}` or `{"restart": true}`
- **Execution**: Client-side — model sends tool call, developer executes and returns result
- **Cost**: 245 additional input tokens per request
- **No beta header required**

#### Text Editor Tool (Client-Executed, Claude 4.x)

```json
{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10000}
```

- **Commands**: `view`, `str_replace`, `create`, `insert` (no `undo_edit` in Claude 4)
- **Cost**: 700 additional input tokens per request
- **No beta header required**

#### Code Execution — Hosted Shell (Remote, Server-Executed)

Anthropic's code execution tool provides a **fully server-side hosted shell** inside an Anthropic-managed container. The client never executes anything — results are returned in the same API response.


| Version                   | Beta Header                 | Notes                               |
| ------------------------- | --------------------------- | ----------------------------------- |
| `code_execution_20260120` | None (GA)                   | Latest, recommended, no beta needed |
| `code_execution_20250825` | `code-execution-2025-08-25` | Original hosted shell beta          |
| `code_execution_20250522` | `code-execution-2025-05-22` | Legacy Python-only sandbox          |


The `20250825`+ versions provide two **server-side sub-tools**:

- `bash_code_execution` — runs shell commands in the container
- `text_editor_code_execution` — file operations (view, create, str_replace, insert)

Response blocks use `server_tool_use` type (IDs prefixed `srvtoolu`_), with results as `bash_code_execution_tool_result` and `text_editor_code_execution_tool_result`.

**Key capabilities:**

- Container: Linux x86_64, Python 3.11.12, 5GiB RAM, 5GiB disk, 1 CPU, no internet
- Container ID persists across turns via `response.container.id`
- Skills via `container.skills` with `skills-2025-10-02` beta
- File outputs via `BetaCodeExecutionResultBlock` with `file_id`s
- File uploads via `BetaContainerUploadBlockParam`
- Programmatic tool calling: `allowed_callers: ["code_execution_20250825"]`
- Pricing: $0.05/hour/container

**Important**: `bash_code_execution` (server-side, in container) is architecturally distinct from `bash_20250124` (client-side, on user's machine). State is NOT shared between them. Both can be used simultaneously — the model sees them as two separate execution environments.

#### Anthropic Skills

- Open standard (agentskills.io), adopted by multiple providers
- Referenced via `SkillReference` with `skill_id`, `version`, `source` (`'custom'` or `'provider'`)
- Mounted in the container as `container.skills` array
- Require `skills-2025-10-02` beta header

### 2.2 OpenAI Native Tools

#### Shell Tool — Unified Hosted & Local

The OpenAI `shell` tool supports **both** hosted and local execution modes using the **same tool type and output items** (`shell_call` / `shell_call_output`). Available through the Responses API only (not Chat Completions).

**Hosted mode** — OpenAI manages the container:

```json
{
  "type": "shell",
  "environment": {
    "type": "container_auto",
    "network_policy": {"type": "allowlist", "allowed_domains": [...]},
    "file_ids": ["file-xxx"]
  }
}
```

- Server-executed in OpenAI's Debian 12 container (Python 3.11, Node 22, Java 17, etc.)
- Working directory: `/mnt/data`
- Container expires after 20 minutes of inactivity
- Container reuse via `container_reference` with a `container_id`
- Skills attached to containers (created separately via `client.containers.create(skills=[...])`)
- Network policy with domain allowlist + domain secrets for auth
- File mounting via `file_ids`

**Local mode** — client executes commands:

```json
{"type": "shell", "environment": {"type": "local"}}
```

- Model returns `shell_call` output items with commands
- Client executes in their own runtime, captures stdout/stderr/exit code
- Client returns `shell_call_output` in next request
- Same `shell_call` / `shell_call_output` item types as hosted mode
- Agents SDK provides shell executor helpers for this mode
- Supported on **all models that support the shell tool** (not just codex-mini)

**Key insight**: Unlike Anthropic (where hosted `bash_code_execution` and local `bash_20250124` are completely separate tools), OpenAI uses a **single `shell` tool type** for both modes. The execution mode is determined by the `environment` type, not the tool type.

#### Apply Patch (Client-Executed)

```json
{"type": "apply_patch"}
```

- Supported on GPT-5.1+
- Returns `apply_patch_call` with V4A format diffs
- Operations: `create_file`, `update_file`, `delete_file`

### 2.3 Current Pydantic AI Architecture

**Builtin tools** (`AbstractBuiltinTool`) — server-executed:

- Passed in `ModelRequestParameters.builtin_tools`
- Provider executes; returns `BuiltinToolCallPart` + `BuiltinToolReturnPart`
- Agent loop does **not** execute these — only records in history
- Examples: `WebSearchTool`, `CodeExecutionTool`, `WebFetchTool`

**Toolsets** (`AbstractToolset`) — client-executed:

- Return `ToolDefinition` objects with JSON schemas
- Agent loop calls `toolset.call_tool()` to execute
- Model adapters convert to provider-specific function tool formats
- Examples: `FunctionToolset`, `CombinedToolset`

**The gap**: No mechanism for client-executed tools with provider-native API presentation.

### 2.4 PR #4393: Execution Environments

PR #4393 by @dmontagu introduces `ExecutionEnvironment` ABC with `LocalEnvironment`, `DockerEnvironment`, and `ExecutionEnvironmentToolset` (shell, read_file, write_file, replace_str). Currently open with changes requested. Provides execution backends but uses generic function tool presentation.

### 2.5 PR #4600: Current Draft (Absorbed)

PR #4600 by @mattbrandman introduces `ShellTool` as an `AbstractBuiltinTool` for remote shell/skills support. This plan absorbs and extends #4600, keeping its remote shell work and adding the local native toolset track. Key #4600 elements carried forward:

- `ShellTool` builtin with `skills` and `network_policy`
- `SkillReference` and `CodeExecutionNetworkPolicy` types
- `UploadedFile.target` field (`'message'` / `'container'` / `'both'`)
- `UploadedFile.part_kind` for `ModelResponsePart` union
- Anthropic: `code_execution_20250825` beta, `BetaBashCodeExecutionToolResultBlock`, `BetaTextEditorCodeExecutionToolResultBlock`, container management, `BetaContainerUploadBlockParam`
- OpenAI: `FunctionShellToolParam`, `_OpenAICodeExecutionContext`, shell tool synthesis, container auto/reference management, file mounting
- Mutual exclusion: `ShellTool` + `CodeExecutionTool` cannot coexist
- Profile: `supports_shell_network_policy`

---

## 3. Architecture: Capability-Driven Routing

### 3.1 Design Principle: Users Declare Capabilities, Not Implementation

The user should **never** need to choose between `ShellTool` (remote builtin) and `ShellToolset` (local toolset) directly. Instead, the user declares that their agent has a **shell capability**, and the framework determines the optimal execution path based on:

1. **What the user provided** — local execution context (paths, executor, environment) vs. none
2. **What the provider supports** — remote containers, native local tools, or neither
3. **Data privacy defaults** — if anything local is pointed at (directory path, skills on local filesystem), **default to local mode** to avoid accidental data leakage to providers

```python
# User declares capability — routing is automatic
from pydantic_ai import Agent
from pydantic_ai.capabilities import Shell

# Local execution (implied by local path — data stays local)
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[
    Shell(cwd='/workspace'),
])

# Remote execution (implied by provider-hosted config)
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[
    Shell(skills=[...], network_policy=...),
])

# Auto: use provider remote if available, no local config needed
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[
    Shell(),
])
```

This replaces the previous design where remote and local were independent tracks that a user manually selected.

### 3.2 Capability as the Integration Layer

A `Shell` capability wraps both implementation layers. We ship a minimal `Capability` protocol now (Phase 4) and migrate to the richer #4303 base class when it lands. The user-facing API is stable from day one.

```
Shell (Capability)
├── provides tools:
│   ├── ShellToolset (local, client-executed) — when local execution is configured
│   └── OR nothing — when using remote-only mode
├── provides builtin_tools:
│   ├── ShellTool (remote, provider-hosted) — when remote is configured or auto-detected
│   └── OR nothing — when using local-only mode
├── provides hooks:
│   ├── pre_tool_call: translate BuiltinToolCallPart → ToolCallPart for local execution
│   └── post_tool_call: translate ToolReturnPart → provider-native result format
├── provides model_settings:
│   └── container config, skills, network policy (when remote)
└── provides instructions:
    └── execution environment context for the model
```

**Key constraint**: The concept of the shell tool is **not hardcoded into the agent loop**. All bridging between built-in tool call parts and local tool execution happens within the capability's hooks. The agent loop continues to handle `ToolCallPart` and `BuiltinToolCallPart` through its existing code paths.

### 3.3 Routing Logic

The capability determines the execution path at `prepare()` time:

```
Shell(cwd='/workspace')  →  local mode
  reason: local path provided → data privacy default

Shell(executor=my_callback)  →  local mode
  reason: explicit local executor

Shell(skills=[SkillReference(...)], network_policy=...)  →  remote mode
  reason: provider-hosted config, no local context

Shell()  →  auto mode
  if provider supports remote containers → remote
  else if provider supports native local tools → local (function tool fallback)
  else → generic function tool fallback

Shell(cwd='/workspace', skills=[...])  →  local mode + skills
  reason: local path present → local-by-default for privacy
  skills uploaded to local execution context, NOT sent to provider
```

### 3.4 Implementation Layers (Building Blocks)

The capability delegates to two existing implementation layers:

**Remote layer** (`ShellTool` as `AbstractBuiltinTool`):
- Provider-hosted container execution
- Results come back as `BuiltinToolCallPart` + `BuiltinToolReturnPart`
- Skills, network policy, file mounting are container-level config
- Agent loop does NOT execute these — only records in history

**Local layer** (`ShellToolset` / `TextEditorToolset` / `ApplyPatchToolset`):
- Client-executed with provider-native presentation via `NativeToolDefinition`
- Adapters emit provider-specific format (bash_20250124, shell with local env)
- Model returns `ToolCallPart` (normalized by adapter)
- Agent loop executes via `toolset.call_tool()`

These layers already exist (Phases 1-3 complete). The capability is the new integration layer on top.

### 3.5 Cross-Provider Tool Multiplicity

One Pydantic AI concept may map to **different numbers of provider tools**:

| Pydantic AI | Anthropic | OpenAI |
|---|---|---|
| Shell capability (local) | `bash_20250124` (1 tool) | `shell` with `environment: local` (1 tool) |
| Shell capability (remote) | `code_execution_20260120` → 2 sub-tools (`bash_code_execution` + `text_editor_code_execution`) | `shell` with `container_auto` (1 tool) |
| Text editor | `text_editor_20250728` with `command` arg (1 tool, multiple commands) | N/A (function tool fallback) |
| File patching | N/A (use text editor) | `apply_patch` (1 tool) |

The Anthropic text editor is particularly notable: it bundles `view`, `str_replace`, `create`, `insert` as `command` values within a single tool. In Pydantic AI, the `TextEditorToolset` handles this as a single tool with command dispatch internally, matching Anthropic's native format. This breaks the 1:1 tool mapping assumption — the capability layer handles this abstraction.

### 3.6 Anti-Patterns to Avoid

**Do NOT follow the `MemoryTool` pattern.** The current Anthropic-specific memory tool implementation requires a manually registered companion function tool with a matching name. This is Anthropic-specific, not cross-provider, and creates hidden coupling. The capability approach replaces this pattern with explicit, typed, cross-provider routing.

**Do NOT hardcode shell concepts in the agent loop.** All shell-specific logic belongs in the capability, its hooks, or the model adapters. The agent loop should only know about generic `ToolCallPart`, `BuiltinToolCallPart`, and their return counterparts.

### 3.7 Relationship to Built-in Tool Fallback

There is an open issue about **automatic fallback** from a locally-defined tool to a provider built-in tool (with a `prefers_builtin` flag on `ToolDefinition`). The shell capability is a concrete instance of this pattern: a locally-defined shell tool should use the provider's native built-in tool format when available. The capability approach generalizes this — the web search capability will follow the same pattern (use provider's web search built-in when available, fall back to Tavily/etc. otherwise).

Similarly, there is in-flight work on **tool search** with a `defer_loading` flag on `ToolDefinition` that affects whether tool definitions are sent to the API or pre-filtered locally. The capability layer should be aware of this for large skill sets.

---

## 4. Implementation Layer: Remote Shell — `ShellTool` Builtin

This section describes the remote/hosted shell implementation layer, largely carried forward from #4600. In the capability-driven architecture (section 3), this is the backend that the `Shell` capability delegates to when remote execution is selected.

### 4.1 New Types in `builtin_tools.py`

```python
@dataclass(kw_only=True)
class SkillReference:
    """Reference to a provider-hosted skill."""
    skill_id: str
    version: str | int | None = None
    source: Literal['custom', 'provider'] = 'custom'

@dataclass(kw_only=True)
class CodeExecutionNetworkPolicy:
    """Network access policy for hosted execution environments."""
    mode: Literal['disabled', 'allowlist']
    allowed_domains: Sequence[str] = ()

@dataclass(kw_only=True)
class ShellTool(AbstractBuiltinTool):
    """Builtin tool for provider-hosted workspace execution.

    Provides bash, text editor, skills, and network policy in a hosted container.
    Mutually exclusive with CodeExecutionTool.

    Supported by:
    * Anthropic (code_execution_20250825 beta)
    * OpenAI Responses (FunctionShellToolParam with container_auto)
    """
    skills: Sequence[SkillReference] = ()
    network_policy: CodeExecutionNetworkPolicy | None = None
    kind: str = 'shell'
```

### 4.2 `UploadedFile` Changes

```python
UploadedFileTarget = Literal['message', 'container', 'both']

class UploadedFile:
    # ... existing fields ...
    target: UploadedFileTarget = 'message'
    part_kind: Literal['uploaded-file'] = 'uploaded-file'
```

- `'message'`: Sent as model-visible content (backward compatible default)
- `'container'`: Mounted into the execution container only
- `'both'`: Both message content and container mount
- `part_kind` enables `UploadedFile` in the `ModelResponsePart` discriminated union

### 4.3 Validation

In `models/__init__.py`:

- `ShellTool` + `CodeExecutionTool` = `UserError` (mutually exclusive — OpenAI's API errors if both `code_interpreter` and `shell` are present; Anthropic's `code_execution_20250825` supersedes `code_execution_20250522`)
- `ShellTool.network_policy` on unsupported provider = `UserError`
- `UploadedFile(target='container')` without `ShellTool` = `UserError`

### 4.4 New Fields on `ModelProfile`

```python
@dataclass(kw_only=True)
class ModelProfile:
    # ... existing fields ...

    supports_shell_network_policy: bool = False
    """Whether this model supports network policy on hosted shell containers.
    True for OpenAI Responses.
    """
```

### 4.5 File Uploads & Container Lifecycle

The two providers have fundamentally different models for file uploads and container management:

#### Anthropic: Files as Message Content

Anthropic file uploads are sent as **message content blocks** in the user message. The `UploadedFile.target` field controls where the file goes:

```python
# File visible to the model AND mounted in the container
UploadedFile(file_id='file-xxx', provider_name='anthropic', target='both')

# File only mounted in the container (not visible in conversation)
UploadedFile(file_id='file-xxx', provider_name='anthropic', target='container')

# File only visible in conversation (default, backward compatible)
UploadedFile(file_id='file-xxx', provider_name='anthropic', target='message')
```

Wire format:

- `target='message'` or `'both'` → `BetaImageBlockParam` / `BetaRequestDocumentBlockParam` (image/document content)
- `target='container'` or `'both'` → `BetaContainerUploadBlockParam(file_id=..., type='container_upload')`
- Using `target='container'` without `ShellTool` → `UserError`

Container reuse is **automatic via message history**: the adapter extracts `container_id` from `ModelResponse.provider_details` in the conversation history and passes it in the `container` param of the next request. Users can also explicitly control this via a new field on `AnthropicModelSettings`:

```python
class AnthropicModelSettings(TypedDict, total=False):
    # ... existing fields ...

    anthropic_container: BetaContainerParams | Literal[False] | None
    """Container configuration for Anthropic code execution.
    - None (default): auto-extract container_id from message history.
    - {'id': 'cntr_xxx'}: use a specific container.
    - False: force a fresh container, ignoring history.
    """
```

#### OpenAI: Files at Container Creation Time

OpenAI file mounting happens at the **environment level**, not in message content. Files must be associated with the container when it's configured:

```python
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

# Mount files into the container via typed model settings
settings = OpenAIResponsesModelSettings(
    openai_shell_uploaded_files=[
        UploadedFile(file_id='file-xxx', provider_name='openai', target='container'),
    ],
)
agent.run('Analyze these files', model_settings=settings)

# Or use a pre-created container with files already mounted
settings = OpenAIResponsesModelSettings(openai_shell_container='cntr_xxx')
agent.run('Continue analysis', model_settings=settings)
```

Wire format:

- `UploadedFile(target='container')` → collected by `_collect_container_file_ids()` from messages and merged with `openai_shell_uploaded_files` into `environment.file_ids` on the `container_auto` shell tool (first turn). On subsequent turns (when `container_reference` is used), new container-targeted files are uploaded via `client.containers.files.create(container_id, file_id)` before the model request.
- `openai_shell_container='cntr_xxx'` → `container_reference` environment type (cannot combine with skills, network policy, or file uploads — those require `container_auto`)
- `openai_shell_container=False` → force fresh `container_auto`, ignore history

Container reuse across turns:

- **Automatic**: `_build_shell_tool_param` scans message history via `ShellTool.get_container_id(messages)` for an existing `container_id` (from `BuiltinToolCallPart.args['container_id']`). If found, uses `container_reference` instead of `container_auto`. This is the default behavior — no user configuration needed.
- **Explicit**: `openai_shell_container='cntr_xxx'` for containers created via `client.containers.create()`
- **Force fresh**: `openai_shell_container=False` forces `container_auto` AND strips `container_reference` from message history round-trip items (line ~2269), ensuring OpenAI doesn't infer the old container from conversation context.

**Important**: `container_auto` always creates a new container. Each turn that uses `container_auto` gets a fresh container and loses all prior state. This is why auto-reuse via `container_reference` is the default — without it, multi-turn shell sessions would be broken. Empirically verified: `container_auto` with `file_ids` on turn 2 does NOT add files to an existing container; it creates a new one.

**Creation-time vs. reuse-time settings**: `skills`, `network_policy`, and `openai_shell_uploaded_files` are creation-time concerns on `container_auto`. Once auto-reuse switches to `container_reference`, these settings no longer apply. Users who need to change them must use `openai_shell_container=False` to force a fresh container.

**Expired container auto-recovery**: Containers have short lifetimes. When a user resumes a chat days later, the auto-inferred `container_id` from history will be expired. The adapter handles this with a targeted retry in `_responses_create`:

- If the API returns 404 with `"Container with id '...' not found"` AND the container was **auto-inferred** from history (not explicitly set via `openai_shell_container`), the adapter retries the request with `openai_shell_container=False`, which creates a fresh `container_auto` with `file_ids` from `openai_shell_uploaded_files` re-mounted.
- If the container was **explicitly set** by the user (`openai_shell_container='cntr_xxx'`), the 404 propagates as `ModelHTTPError` — the user chose that specific container, so they should know it's gone.
- Detection uses `_is_container_not_found_error(e)` helper (follows the existing `_check_azure_content_filter` pattern for error-body inspection).
- The retry is self-healing: the fresh response contains a new `container_id`, which becomes the most recent in history, so subsequent turns auto-reuse the new container without hitting the expired one again.

#### Comparison


| Aspect                      | Anthropic                                         | OpenAI                                                                                          |
| --------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| File upload mechanism       | Message content blocks                            | Environment `file_ids`                                                                          |
| When files are sent         | Per-message (can change each turn)                | At container creation / per-request                                                             |
| Container reuse             | Auto from history + `anthropic_container` setting | Auto from history via `ShellTool.get_container_id()` + `openai_shell_container` setting         |
| Pre-created containers      | Via `anthropic_container: {'id': '...'}`          | Via `openai_shell_container: 'cntr_xxx'` using `container_reference`                            |
| Container + skills conflict | No conflict (skills in `container.skills`)        | `container_reference` cannot combine with skills/network_policy/file_ids (use `container_auto`) |


### 4.6 Anthropic Adapter: Remote Shell

- `ShellTool` → `BetaCodeExecutionTool20250825Param(type='code_execution_20250825')` or `BetaCodeExecutionTool20260120Param(type='code_execution_20260120')`
- **SDK note**: The `20260120` version is GA at the API level (no beta HTTP header required), but the Python SDK still places it under the `beta` module (`BetaCodeExecutionTool20260120Param`). Implementation should use `client.beta.messages.create()` with the `20260120` param type. No `anthropic-beta` header is needed in the HTTP request — the SDK's `beta` module path is a naming artifact, not a feature gate.
- Beta features: `skills-2025-10-02` (if skills present)
- Skills → `BetaSkillParams` in `container.skills`
- Container ID persistence: extract from `response.container.id` → store in `provider_details['container_id']` → extract from history for next request
- New response block types mapped to `BuiltinToolCallPart` / `BuiltinToolReturnPart`:
  - `BetaBashCodeExecutionToolResultBlock` → tool name `bash_code_execution`
  - `BetaTextEditorCodeExecutionToolResultBlock` → tool name `text_editor_code_execution`
  - File outputs → `UploadedFile` parts extracted from result blocks
- `UploadedFile(target='container')` → `BetaContainerUploadBlockParam`
- Round-trip: `BetaBashCodeExecutionToolResultBlockParam`, `BetaTextEditorCodeExecutionToolResultBlockParam` for history replay

### 4.7 OpenAI Adapter: Remote Shell

- `ShellTool` → `shell` tool with `container_auto` environment
- Skills: Attached to containers via `client.containers.create(skills=[...])`, then referenced via `container_reference`
- Network policy → `network_policy` on `container_auto` environment
- Domain secrets → `domain_secrets` in network policy for auth headers
- File mounting → `file_ids` in environment (from `UploadedFile(target='container')`)
- Container reuse: **automatic** via `ShellTool.get_container_id(messages)` history scan → `container_reference`; explicit via `openai_shell_container='cntr_xxx'`; force fresh via `openai_shell_container=False` (strips container refs from both tool param and message history round-trip)
- `shell_call` / `shell_call_output` output items mapped to `BuiltinToolCallPart` / `BuiltinToolReturnPart`
- Raw `shell` tools in `openai_builtin_tools` auto-normalized
- New fields on `OpenAIResponsesModelSettings` TypedDict:

```python
class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
    # ... existing fields ...

    openai_shell_uploaded_files: Sequence[UploadedFile]
    """Files to mount into an OpenAI hosted shell container.
    Requires a shell tool in the request (from ShellTool or openai_builtin_tools).
    """

    openai_shell_container: str | Literal[False] | None
    """Container configuration for the OpenAI shell tool.
    - None (default): auto-reuse container from message history via ShellTool.get_container_id();
      falls back to container_auto on first turn.
    - str (e.g. 'cntr_xxx'): reuse a specific pre-created container via container_reference.
    - False: force fresh container_auto, strip container references from history round-trip.
    Cannot combine container_reference with skills, network_policy, or file uploads.
    """
```

**Important**: OpenAI uses the **same `shell` tool type** for both hosted and local execution. The difference is the `environment` type: `container_auto`/`container_reference` for hosted, `local` for client-executed. Both use `shell_call`/`shell_call_output` output items. This means `ShellTool` (builtin, remote) and `ShellToolset` (toolset, local) will both emit an OpenAI `shell` tool — differentiated by environment type.

---

## 5. Implementation Layer: Local Shell — Native Toolsets

This section describes the local/client-executed tool implementation layer. In the capability-driven architecture (section 3), these toolsets are the backends that the `Shell`, `TextEditor`, and `FilePatching` capabilities delegate to when local execution is selected.

### 5.1 Core Concept: `NativeToolDefinition`

A typed discriminated union on `ToolDefinition` that tells model adapters to use provider-native format. Each native tool kind has its own typed config — no `dict[str, Any]`:

```python
@dataclass(kw_only=True)
class ShellNativeDefinition:
    """Native shell tool (Anthropic bash_20250124, OpenAI shell with local env)."""
    kind: Literal['shell'] = 'shell'

@dataclass(kw_only=True)
class TextEditorNativeDefinition:
    """Native text editor tool (Anthropic text_editor_20250728)."""
    kind: Literal['text_editor'] = 'text_editor'
    max_characters: int | None = None

@dataclass(kw_only=True)
class ApplyPatchNativeDefinition:
    """Native apply_patch tool (OpenAI apply_patch)."""
    kind: Literal['apply_patch'] = 'apply_patch'

NativeToolDefinition = ShellNativeDefinition | TextEditorNativeDefinition | ApplyPatchNativeDefinition
```

Added to `ToolDefinition`:

```python
native_definition: NativeToolDefinition | None = None
"""Optional provider-native tool format hints.

When set, model adapters that support the native kind emit the
provider-specific tool format. Unsupported adapters fall back to
parameters_json_schema. Adapters use isinstance() to extract typed
config (e.g., TextEditorNativeDefinition.max_characters).
"""
```

**Design rationale**: `ToolDefinition.metadata` (which already exists as `dict[str, Any] | None`) was considered but rejected — native tool format is a first-class concern about tool presentation, not arbitrary metadata. A new typed field is more explicit and avoids `Any`. The discriminated union satisfies the project guideline "use `TypedDict` or dataclass instead of `dict[str, Any]` when structure is known."

### 5.2 `ShellToolset`

New file: `pydantic_ai_slim/pydantic_ai/toolsets/shell.py`

```python
@dataclass(kw_only=True)
class ShellOutput:
    """Result of a shell command execution."""
    output: str
    exit_code: int

class ShellExecutor(Protocol):
    """Protocol for shell command execution."""
    async def execute(self, command: str) -> ShellOutput: ...
    async def restart(self) -> ShellOutput: ...

@dataclass
class ShellToolset(AbstractToolset[AgentDepsT]):
    """Toolset for local shell command execution.

    Uses provider-native format when supported (Anthropic bash, OpenAI
    shell with local env), falls back to function tool otherwise.
    """
    executor: ShellExecutor
    """The shell executor. Use ShellToolset.local() for a default subprocess-based executor."""

    tool_name: str = 'shell'
    description: str = 'Execute a shell command and return the output.'
    max_output_chars: int = 200_000
    """Maximum output characters. Exceeding output is truncated with a marker."""
    timeout: float | None = None
    """Timeout in seconds per command. Raises ModelRetry on timeout."""

    @classmethod
    def local(
        cls,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ShellToolset[AgentDepsT]:
        """Create a ShellToolset with a default subprocess-based executor."""
        return cls(executor=_LocalShellExecutor(cwd=cwd, env=env), **kwargs)
```

**Session model**: `_LocalShellExecutor` maintains a **persistent shell session** — `cd`, `export`, and other state-modifying commands persist across consecutive `execute()` calls. This matches the behavior models expect from Anthropic's `bash` tool (documented as "persistent bash session that maintains state"). `restart()` terminates the session and starts a fresh one. The `ToolDefinition` sets `sequential=True` to ensure concurrent calls are serialized, since the underlying subprocess is single-threaded.

`**call_tool()` handles:**

- `command` → `executor.execute(command)`
- `restart=True` → `executor.restart()`
- Timeout: wraps call with `anyio.fail_after(self.timeout)` → raises `ModelRetry(f'Command timed out after {self.timeout}s.')`
- Truncation: if `len(output) > max_output_chars`, truncates and appends `'[output truncated — {shown} chars of {total} total]'`

**Hello world:**

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import ShellToolset

agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[ShellToolset.local(cwd='/workspace')])
result = agent.run_sync('List all Python files')
```

`**get_tools()` emits:**

```python
ToolDefinition(
    name='shell',
    description='Execute a shell command and return the output.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'command': {'type': 'string', 'description': 'The shell command to execute'},
            'restart': {'type': 'boolean', 'description': 'Restart the shell session', 'default': False},
        },
        'required': ['command'],
    },
    native_definition=ShellNativeDefinition(),
)
```

**Execution flow:**

1. Adapter sees `native_definition` is `ShellNativeDefinition`
2. Anthropic → `{"type": "bash_20250124", "name": "bash"}`
3. OpenAI Responses → `{"type": "shell", "environment": {"type": "local"}}`
4. Others → regular function tool using `parameters_json_schema`
5. Model returns tool call; adapter normalizes to `ToolCallPart(tool_name='shell', ...)`
6. Agent loop calls `ShellToolset.call_tool('shell', {'command': '...'})`
7. Toolset invokes `executor.execute(command)` → returns `ShellOutput`

### 5.3 `TextEditorToolset`

New file: `pydantic_ai_slim/pydantic_ai/toolsets/text_editor.py`

```python
# Discriminated union of per-command types for type safety
class ViewCommand(TypedDict):
    command: Literal['view']
    path: str
    view_range: NotRequired[list[int]]  # [start, end], 1-indexed

class StrReplaceCommand(TypedDict):
    command: Literal['str_replace']
    path: str
    old_str: str
    new_str: str

class CreateCommand(TypedDict):
    command: Literal['create']
    path: str
    file_text: str

class InsertCommand(TypedDict):
    command: Literal['insert']
    path: str
    insert_line: int
    insert_text: str

TextEditorCommand = ViewCommand | StrReplaceCommand | CreateCommand | InsertCommand

@dataclass(kw_only=True)
class TextEditorOutput:
    output: str
    success: bool = True

TextEditorExecuteFunc = Callable[[TextEditorCommand], Awaitable[TextEditorOutput]]

@dataclass
class TextEditorToolset(AbstractToolset[AgentDepsT]):
    """Toolset for local text editor operations.

    Uses provider-native format on Anthropic (text_editor_20250728),
    falls back to function tool otherwise.
    """
    execute: TextEditorExecuteFunc
    max_characters: int | None = None
    tool_name: str = 'str_replace_based_edit_tool'
```

### 5.4 `ApplyPatchToolset`

New file: `pydantic_ai_slim/pydantic_ai/toolsets/apply_patch.py`

```python
@dataclass(kw_only=True)
class ApplyPatchOperation:
    """A file patch operation from the model."""
    operation_type: Literal['create_file', 'update_file', 'delete_file']
    path: str
    diff: str | None = None  # V4A format diff (for update_file)
    content: str | None = None  # Full content (for create_file)

@dataclass(kw_only=True)
class ApplyPatchOutput:
    status: Literal['completed', 'failed']
    output: str | None = None

ApplyPatchExecuteFunc = Callable[[ApplyPatchOperation], Awaitable[ApplyPatchOutput]]

@dataclass
class ApplyPatchToolset(AbstractToolset[AgentDepsT]):
    """Toolset for OpenAI's apply_patch tool (V4A diffs).

    Uses native apply_patch on OpenAI, falls back to function tool otherwise.
    """
    execute: ApplyPatchExecuteFunc
```

### 5.5 Fallback Warning

When a `ToolDefinition` with `native_definition` is processed by an adapter that doesn't support the native kind, the adapter falls back to a standard function tool using `parameters_json_schema`. To make this visible, adapters emit a `warnings.warn()` once per native tool kind:

```
ShellToolset: falling back to function tool format on google — model performance may be degraded.
```

This is non-breaking, opt-out via standard Python warning filters, and helps users discover when they're not getting native format benefits.

### 5.6 Profile Flags for Native Tool Support

```python
@dataclass(kw_only=True)
class ModelProfile:
    # ... existing ...
    supports_native_shell_tool: bool = False
    supports_native_text_editor_tool: bool = False
    supports_native_apply_patch_tool: bool = False
```

### 5.7 Adapter Name Mapping

Model adapters resolve native tool names using the `ToolDefinition` objects themselves as the source of truth — no separate mutable mapping dict is needed.

- **Outgoing**: When building the provider request, the adapter iterates `ToolDefinition` objects. If one has `native_definition`, the adapter emits the provider-native format and records the mapping implicitly (the `ToolDefinition` already carries both its `name` and its `native_definition.kind`).
- **Incoming**: When a response contains a provider-native tool name (e.g., `'bash'`), the adapter scans the `ToolDefinition` list for one with a matching `native_definition` (e.g., `ShellNativeDefinition`) and uses its `name` field. This works correctly even under `PrefixedToolset` or `RenamedToolset` because the `ToolDefinition.name` already reflects the prefix/rename (e.g., `'my_shell'`).

This approach avoids hidden mutable state and is composability-safe — the tool definitions are the single source of truth for both directions.

**Implementation note for streaming**: The `AnthropicStreamedResponse` and `OpenAIResponsesStreamedResponse` classes are constructed separately from their parent model classes. They currently do not receive tool definitions. For native tool name resolution during streaming, the tool definitions (or a pre-built native-name-to-toolset-name lookup) must be passed to the streamed response constructor. This is a small addition to each constructor signature (e.g., `_native_tool_names: dict[str, str]`) built once during `_responses_create` / `_messages_create` from the `ModelRequestParameters.function_tools` list.

### 5.8 Anthropic Adapter: Local Tools

When `ToolDefinition` has `native_definition.kind == 'shell'`:

- Emit `BetaToolBash20250124Param(type='bash_20250124', name='bash')`
- Register name mapping: `'bash'` ↔ toolset's `tool_name`

When `native_definition.kind == 'text_editor'`:

- Emit `BetaToolTextEditor20250728Param(type='text_editor_20250728', name='str_replace_based_edit_tool', max_characters=...)`
- Register name mapping

Response: `BetaToolUseBlock` with native name → normalize to `ToolCallPart` with toolset name.
Next request: `ToolReturnPart` → `BetaToolResultBlockParam` with native name.

### 5.9 OpenAI Adapter: Local Tools

When `native_definition.kind == 'shell'`:

- Emit `{"type": "shell", "environment": {"type": "local"}}` — uses the same `shell` tool type as hosted mode, just with local environment
- Same `shell_call` / `shell_call_output` output items as hosted mode
- Client executes commands from `shell_call` and returns `shell_call_output`

When `native_definition.kind == 'apply_patch'`:

- Emit `{"type": "apply_patch"}` for GPT-5.1+
- Fall back to function tool for older models

Response: `shell_call` / `apply_patch_call` → `ToolCallPart` with toolset name.
Result: `shell_call_output` / `apply_patch_call_output` with appropriate format.

**Architectural elegance**: Because OpenAI uses the same `shell` tool type for both hosted and local, the adapter code for handling `shell_call`/`shell_call_output` is shared. The only difference is whether the environment is `container_auto` (hosted, builtin) or `local` (client-executed, toolset).

**Streaming note**: OpenAI `shell_call` / `shell_call_output` items are delivered as **atomic/complete items**, not incrementally streamed (unlike `function_call` which streams argument deltas). The OpenAI Python SDK (v2.26.0) has no streaming event types for shell calls (`ResponseStreamEvent` union excludes them). This means both streaming and non-streaming adapter paths handle shell calls identically — as complete output items. No incremental delta assembly is needed.

---

## 6. Provider Adapter Changes (Summary)

### 6.1 Anthropic (`models/anthropic.py`)


| Change                                                                  | Track  | Source |
| ----------------------------------------------------------------------- | ------ | ------ |
| `ShellTool` → `BetaCodeExecutionTool20250825Param`                      | Remote | #4600  |
| Skills → `BetaSkillParams` in container                                 | Remote | #4600  |
| Container ID persistence                                                | Remote | #4600  |
| `BetaBashCodeExecutionToolResultBlock` mapping                          | Remote | #4600  |
| `BetaTextEditorCodeExecutionToolResultBlock` mapping                    | Remote | #4600  |
| `BetaContainerUploadBlockParam` from `UploadedFile(target='container')` | Remote | #4600  |
| `native_definition.kind='shell'` → `bash_20250124`                      | Local  | New    |
| `native_definition.kind='text_editor'` → `text_editor_20250728`         | Local  | New    |
| Name mapping (bash ↔ shell, str_replace_based_edit_tool ↔ toolset name) | Local  | New    |


### 6.2 OpenAI Responses (`models/openai.py`)


| Change                                                                         | Track  | Source |
| ------------------------------------------------------------------------------ | ------ | ------ |
| `ShellTool` → `FunctionShellToolParam` (container_auto)                        | Remote | #4600  |
| Skills → `SkillReferenceParam` in environment                                  | Remote | #4600  |
| Network policy → `OpenAIContainerAutoNetworkPolicy`                            | Remote | #4600  |
| File mounting → `file_ids` in environment                                      | Remote | #4600  |
| `_OpenAICodeExecutionContext` for transport tracking                           | Remote | #4600  |
| `ResponseFunctionShellToolCall` / Output mapping                               | Remote | #4600  |
| `openai_shell_uploaded_files`, `openai_shell_container` settings               | Remote | #4600  |
| `native_definition.kind='shell'` → `shell` with `environment: {type: "local"}` | Local  | New    |
| `native_definition.kind='apply_patch'` → `apply_patch` (GPT-5.1+)              | Local  | New    |
| Name mapping for local tools                                                   | Local  | New    |


### 6.3 OpenAI Chat (`models/openai.py`)

- `ShellTool` excluded from `supported_builtin_tools` (Chat API doesn't support it)
- Native toolsets fall back to function tools (no native format in Chat Completions)

### 6.4 Other Adapters

- No changes for remote (they don't support `ShellTool`)
- Native toolsets automatically fall back to function tools via `parameters_json_schema`

---

## 7. Agent Loop & Message Changes

### 7.1 `messages.py`

From #4600:

- `UploadedFileTarget` type alias
- `UploadedFile.target` field (default `'message'`)
- `UploadedFile.part_kind` field
- `UploadedFile` added to `ModelResponsePart` union
- `PartStartEvent` / `PartEndEvent` `previous_part_kind` / `next_part_kind` updated for `'uploaded-file'`

### 7.2 `_agent_graph.py`

- Handle `UploadedFile` in response part iteration (skip — file references are for history/streams only)
- No changes needed for native toolsets (they use standard `ToolCallPart` / `ToolReturnPart`)

### 7.2.1 OTel/Logfire Observability

No new instrumentation is required:

- **Track 2 (local)**: Native toolsets produce standard `ToolCallPart` / `ToolReturnPart`, which are already instrumented in `_otel_messages.py`
- **Track 1 (remote)**: `BuiltinToolCallPart` / `BuiltinToolReturnPart` are already handled by existing instrumentation
- Container IDs in `provider_details` will appear naturally in span attributes via the existing `provider_metadata` instrumentation

### 7.3 No `pause_turn` in This PR

`pause_turn` (Anthropic's long-running operation stop reason) is **not** in scope. It requires careful design for the agent loop and may warrant its own plan. Container state persistence via `container.id` round-tripping is sufficient for now. If the API returns `pause_turn`, the adapter should treat it as `end_turn` and surface the stop reason in `provider_details` so users can handle it in application code.

### 7.4 Multi-Provider Message History

When a conversation switches providers mid-stream (e.g., Anthropic → OpenAI), message history may contain:

- `BuiltinToolCallPart` / `BuiltinToolReturnPart` from the previous provider's remote shell
- `ToolCallPart` / `ToolReturnPart` from the previous provider's native local tools
- `UploadedFile` references with a `provider_name` that doesn't match the new provider

**Handling:**

- `UploadedFile` with mismatched `provider_name` → **skip silently** during message mapping (the file ID is provider-specific and unusable). The existing pattern in the Anthropic adapter already validates `item.provider_name != self.system`.
- `BuiltinToolCallPart` / `BuiltinToolReturnPart` with mismatched `provider_name` → already handled (adapters check `response_part.provider_name == self.system` before processing)
- `ToolCallPart` from native local tools → normalized to the toolset's `tool_name` (e.g., `'shell'`), so they round-trip as regular tool calls. The new adapter will map them to its own native format if the tool is still present, or as-is otherwise.

---

## 8. Error Handling

Following established patterns in the codebase (`ModelRetry` for recoverable tool errors, `UserError` for configuration mistakes):

### 8.1 Local Shell Errors (Track 2)


| Scenario                                | Handling                                                                                      | Pattern                                                     |
| --------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Command timeout                         | `ModelRetry(f'Command timed out after {timeout}s.')`                                          | Same as `FunctionToolset` timeout in `toolsets/function.py` |
| Non-zero exit code                      | Return `ShellOutput(output=stderr+stdout, exit_code=N)` — let the model decide how to proceed | Not an error — exit codes are informational                 |
| Output exceeds `max_output_chars`       | Truncate and append `'[output truncated — {shown} chars of {total} total]'`                   | Toolset-level truncation                                    |
| `execute()` raises unexpected exception | Propagate as-is (not `ModelRetry`) — framework-level exception                                | Follows `FunctionToolset` pattern                           |


### 8.2 Remote Shell Errors (Track 1)


| Scenario                                           | Handling                                                                                               | Pattern                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| Container expired / not found                      | Provider returns API error → adapter raises `ModelHTTPError` or similar                                | Standard API error handling                  |
| Stale `container_id` in history                    | Adapter catches container-not-found, clears `container_id` from settings, retries with fresh container | Graceful degradation                         |
| Invalid `file_id` in `UploadedFile`                | Provider returns API error → surfaces as request failure                                               | Standard API error handling                  |
| Skills not available on model                      | `UserError` at configuration time (checked via `supported_builtin_tools` in `prepare_request`)         | Same as existing builtin tool validation     |
| `ShellTool.network_policy` on unsupported provider | `UserError` at configuration time                                                                      | Same pattern as existing profile flag checks |


### 8.3 Concurrency

- **Remote containers**: Concurrent `agent.run()` calls sharing a container are the provider's responsibility. Both Anthropic and OpenAI handle concurrent requests to the same container safely.
- **Local shell (`_LocalShellExecutor`)**: The persistent subprocess is inherently sequential. The `ToolDefinition` sets `sequential=True`, which causes the agent loop to serialize tool calls to this tool rather than running them concurrently.
- **Multiple agent instances**: If two `Agent` instances share the same `ShellToolset` instance, their calls are serialized via `sequential=True`. For true parallelism, use separate `ShellToolset` instances.

### 8.4 Configuration Errors


| Scenario                                                  | Handling                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ShellTool` + `CodeExecutionTool`                         | `UserError('ShellTool and CodeExecutionTool are mutually exclusive...')`                          |
| `UploadedFile(target='container')` without `ShellTool`    | `UserError('UploadedFile with target including container requires ShellTool...')`                 |
| `openai_shell_container` + skills/network_policy/file_ids | `UserError('container_reference cannot combine with skills, network_policy, or file uploads...')` |
| `ShellTool` on unsupported provider                       | `UserError` via existing `supported_builtin_tools` validation                                     |


---

## 9. Security Considerations

Local shell execution (Track 2) gives an LLM the ability to run arbitrary commands on the developer's machine. This is inherently dangerous and must be documented clearly.

### 9.1 Documentation Requirements

All shell-related docs pages must include a prominent warning:

> Running shell commands from LLM output can be dangerous. Always sandbox execution, restrict file access, and review tool activity. Consider using `ApprovalRequiredToolset` to require human approval for sensitive commands.

### 9.2 Built-in Safety Mechanisms

These are provided by the framework without requiring hooks or callbacks:

- **Output truncation**: `ShellToolset.max_output_chars` prevents token budget exhaustion from large outputs
- **Timeout**: `ShellToolset.timeout` prevents runaway commands
- `**ModelRetry` on timeout**: Model can adjust its approach rather than hanging
- `**ApprovalRequiredToolset` composition**: Users can wrap `ShellToolset` in `ApprovalRequiredToolset` for human-in-the-loop approval:

```python
from pydantic_ai.toolsets import ShellToolset, ApprovalRequiredToolset

shell = ShellToolset.local(cwd='/workspace')
approved_shell = ApprovalRequiredToolset(wrapped=shell)
agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[approved_shell])
```

- `**FilteredToolset` composition**: Users can restrict which commands are available
- **Execution callback ownership**: The `ShellExecutor` protocol means the user controls exactly how commands are executed — they can add allowlists, denylists, sandboxing, logging, or any other safety mechanism in their implementation

### 9.3 What We Explicitly Do NOT Provide

- Command sanitization — this is the executor's responsibility
- Built-in allowlists/denylists — use `FilteredToolset` or custom executor
- Hooks or callbacks for command interception — out of scope for this PR, may come with Capabilities (#4303)
- Path traversal prevention for `TextEditorToolset` — the executor implementation controls file access

### 9.4 Remote Shell Security (Track 1)

Provider-hosted containers are sandboxed by the provider. Security considerations:

- Network policy controls what the container can access (OpenAI `allowlist`)
- Domain secrets prevent credential leakage (OpenAI `domain_secrets`)
- Containers have limited resources (Anthropic: 5GiB RAM, 5GiB disk, 1 CPU, no internet by default)
- Container expiry (OpenAI: 20 min inactivity) limits exposure window

---

## 10. Relationship to PR #4393

**Complementary, not overlapping.** This plan and #4393 address different concerns:


| Concern          | This Plan                                                                     | PR #4393                                               |
| ---------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------ |
| Where code runs  | Provider containers (remote) OR local via callback (local)                    | Framework-managed environments (local, Docker, memory) |
| How tools appear | Provider-native formats + function fallback                                   | Generic function tools only                            |
| Key files        | `builtin_tools.py`, `tools.py`, `models/`, `toolsets/shell.py`, `messages.py` | `environments/`, `toolsets/execution_environment.py`   |


**Future integration** after both land:

```python
from pydantic_ai.environments import LocalEnvironment
from pydantic_ai.toolsets import ShellToolset, TextEditorToolset

env = LocalEnvironment('/workspace')
agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[
    ShellToolset(execute=lambda cmd: env.shell(cmd)),
    TextEditorToolset(execute=lambda cmd: _env_editor(env, cmd)),
])
```

Or via a convenience wrapper:

```python
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[NativeExecutionEnvironmentToolset(LocalEnvironment('/workspace'))],
)
```

---

## 11. Relationship to Other PRs & In-Flight Work

### 11.1 PR #4505 (ShellTool PLAN.md by @bsherifi)

Our `NativeToolDefinition` generalizes their `native_kind: Literal['shell']` concept, adding `config` for tools that need extra parameters (text editor's `max_characters`). This plan supersedes #4505 while acknowledging the contribution.

### 11.2 PR #4513 (TextEditorTool by @st4r0)

Their approach follows the `MemoryTool` pattern (builtin + companion function tool). The capability-driven toolset approach is architecturally cleaner for client-executed tools — and avoids the anti-pattern identified in section 3.6. Their test cassettes and Anthropic adapter changes are valuable references.

### 11.3 #4303 (Capabilities Abstraction) — ENHANCES, NOT BLOCKS

The capabilities PR (Douwe, in-flight) introduces a richer `Capability` base class with hooks, instructions, and model settings. We ship Phase 4 with a minimal `Capability` protocol now and migrate to #4303's base class when it lands — gaining hooks for built-in ↔ tool call bridging, system prompt instructions, and model settings integration. Our implementation layers (`ShellTool`, `ShellToolset`, etc.) become backends that the `Shell` capability delegates to:

```
Shell (Capability)
├── ShellTool (remote, provider-hosted) when provider supports containers
├── ShellToolset (local, client-executed) when user provides local execution
├── ExecutionEnvironmentToolset (local, from #4393) for environment-backed execution
├── Hooks: built-in ↔ tool call part bridging for local native tools
└── Instructions, model settings, guardrails
```

The capability's hooks handle the translation between `BuiltinToolCallPart` and `ToolCallPart` without hardcoding shell concepts into the agent loop.

### 11.4 Built-in Tool Fallback Issue

Open issue about automatic fallback from a locally-defined tool to a provider built-in tool, using a `prefers_builtin` or similar flag on `ToolDefinition`. The shell capability is a concrete instance of this pattern — a locally-defined shell tool should prefer the provider's native built-in tool format when available. The `NativeToolDefinition` field on `ToolDefinition` is our current implementation of this concept. The capability layer will generalize it (web search capability will follow the same fallback pattern: provider web search → Tavily/etc.).

### 11.5 Tool Search / Deferred Loading

In-flight work on a `defer_loading` flag on `ToolDefinition` that allows tool definitions to be filtered on the server side (provider tool search) or pre-filtered locally. Relevant for large skill sets where sending all tool definitions would be inefficient. The capability layer should be aware of this when exposing many tools from skills.

### 11.6 Unified Built-in Tool Call/Return Parts PR

Open PR that unifies the `content` field of `BuiltinToolCallPart` and `BuiltinToolReturnPart` across all providers (currently raw JSON from each API). This plan should not conflict with that work. The unified content format will make it easier for capability hooks to translate between built-in and regular tool call parts, since the content will have a consistent shape regardless of provider.

### 11.7 David Sanchez's Cartesian Test Patterns

David's work on the multimodal tool return parts PR and tool choice PR established patterns for cross-provider Cartesian product testing. This plan's test strategy (section 12) should follow those patterns for comprehensive coverage across provider × mode × streaming combinations.

### 11.8 `MemoryTool` — Anti-Pattern to Replace

The current `MemoryTool` implementation on Anthropic requires a manually registered companion function tool with a matching name. This is Anthropic-specific and creates hidden coupling. Once the capabilities abstraction lands, `MemoryTool` should be refactored to use the same capability-driven pattern as Shell. This plan does NOT implement that refactor but establishes the pattern it should follow.

---

## 12. Test Strategy

### 12.0 Running Tests Locally

API keys for Anthropic, OpenAI, and other providers are managed via [Doppler](https://www.doppler.com/). To run integration tests locally with the required keys injected:

```bash
doppler run -- make test
doppler run -- pytest tests/models/test_anthropic.py -k test_shell
doppler run -- pytest tests/models/test_openai_responses.py -k test_shell
```

### 12.0.1 Recording VCR Cassettes

All model/API integration tests use `pytest-recording` with VCR cassettes. Cassettes must be **recorded locally and committed** — CI runs with `record_mode=none` (playback only).

**To record new cassettes:**

```bash
# Record all cassettes (rewrites existing ones)
doppler run -- make update-vcr-tests

# Record cassettes for specific tests only
doppler run -- uv run -m pytest --record-mode=rewrite tests/models/test_anthropic.py -k test_shell
doppler run -- uv run -m pytest --record-mode=rewrite tests/models/test_openai_responses.py -k test_shell
```

**Record modes:**

- `rewrite` — record and overwrite existing cassettes (use when updating tests)
- `once` — record only if cassette doesn't exist (use for new tests)
- `none` — playback only, fail if cassette missing (CI default)

**Cassette location:** `tests/models/cassettes/{test_module_name}/` — auto-discovered from test function name. Example: `test_anthropic_shell_tool()` → `tests/models/cassettes/test_anthropic/test_anthropic_shell_tool.yaml`

**Cassette hygiene:**

- Sensitive headers (`authorization`, `x-api-key`) are automatically filtered by the custom serializer in `tests/json_body_serializer.py`
- JSON bodies are normalized and prettified
- Access tokens and client secrets are scrubbed
- Review cassette diffs before committing — ensure no secrets leaked

### 12.0.2 Code Coverage Requirements

**100% code coverage is mandatory** — enforced in CI via `fail_under = 100` in `pyproject.toml`.

```bash
# Run tests with coverage locally
make testcov

# Run tests without coverage (faster for development)
make test
```

**CI coverage pipeline:**

1. Tests run on Python 3.10–3.14 across 3 variants (slim, standard, all-extras)
2. Each run writes `.coverage/.coverage.{python}-{variant}`
3. `uv run coverage combine` merges all coverage data
4. `uv run coverage report` enforces 100% threshold
5. `uv run strict-no-cover` verifies `# pragma: no cover` markers are correct (fails if marked code is actually covered)
6. HTML report uploaded as CI artifact for review

**Coverage markers:**

- `# pragma: no cover` — strict: asserts code is NOT covered. CI fails if it gets covered. Use only for genuinely untestable code.
- `# pragma: lax no cover` — lenient: allows partial coverage without failing. Use for platform-specific or CI-only code paths.

**Already excluded from coverage** (no markers needed): `raise NotImplementedError`, `if TYPE_CHECKING:`, `@overload`, `@abstractmethod`, `assert_never()`, `except ImportError`.

**For this PR:** Every new code path must be covered by tests. VCR cassettes provide integration coverage; unit tests cover validation logic and edge cases. The fallback path (native → function tool on unsupported provider) needs explicit test coverage.

### 12.1 Integration Tests (VCR Cassettes)

Per `tests/AGENTS.md`:

**Remote (Track 1):**

- Anthropic: `code_execution_20250825` with skills
- Anthropic: `code_execution_20250825` with container file uploads
- OpenAI: `FunctionShellToolParam` with `container_auto`
- OpenAI: hosted shell with skills and network policy
- Both: streaming variants

**Local (Track 2):**

- Anthropic: `bash_20250124` via `ShellToolset`
- Anthropic: `text_editor_20250728` via `TextEditorToolset`
- OpenAI: `shell` (local mode) via `ShellToolset`
- OpenAI: `apply_patch` via `ApplyPatchToolset`

### 12.2 Unit Tests

- `ShellTool` + `CodeExecutionTool` mutual exclusion validation
- `ShellToolset.get_tools()` returns correct `ToolDefinition` with `native_definition`
- `TextEditorToolset.get_tools()` returns correct definitions
- Adapter name mapping (bidirectional)
- Fallback to function tool when native unsupported
- `SkillReference` serialization/deserialization
- `UploadedFile.target` handling in Anthropic and OpenAI adapters
- **Native tool fallback**: verify `ShellToolset` with a provider that doesn't support native shell (e.g., Google, Groq) emits a standard function tool and logs a `warnings.warn()`

### 12.3 Cross-Provider Cartesian Matrix

Following David Sanchez's patterns from the multimodal tool return parts and tool choice PRs, define a Cartesian matrix covering all combinations:

| Dimension | Values |
|---|---|
| Provider | Anthropic, OpenAI Responses, OpenAI Chat (fallback), Google (fallback), Groq (fallback) |
| Mode | Remote (hosted), Local (native), Local (function fallback) |
| Streaming | Yes, No |
| Tool type | Shell, Text Editor, Apply Patch |
| Capability routing | Local path → local, Remote config → remote, Auto → provider-dependent |

Not all combinations are valid (e.g., OpenAI Chat doesn't support remote shell). Invalid combinations should be tested for graceful error handling or silent fallback. The matrix ensures we've thought about all interaction gaps.

### 12.4 Snapshot Tests

- `result.all_messages()` for full tool call/return flow
- Provider-specific request payloads (verify native format emission)
- Capability routing decisions (verify local-by-default for data privacy)

---

## 13. Documentation

### 13.1 Updated: `docs/builtin-tools.md`

- Add `ShellTool` section with skills, network policy, file mounting
- Document `SkillReference` and `CodeExecutionNetworkPolicy`
- Document `UploadedFile.target` for container uploads
- Mutual exclusion with `CodeExecutionTool`

### 13.2 New: `docs/native-tools.md`

- **Capability API first**: Show `Shell(cwd='/workspace')` as the recommended approach
- **Security warning**: Prominent admonition about LLM-driven shell execution risks
- Explain native vs function tools and why it matters (models fine-tuned on provider schemas)
- **Data privacy note**: Explain that local paths default to local execution to avoid data leakage
- **Advanced section**: Direct `ShellToolset`, `TextEditorToolset`, `ApplyPatchToolset` usage for users who need low-level control
- Provider-specific behavior and fallback
- Safety patterns: `ApprovalRequiredToolset` composition, custom executors with restrictions

### 13.3 Updated: `docs/toolsets.md`

- Add section on native toolsets, link to `docs/native-tools.md`

### 13.4 Updated: Provider Docs

- `docs/models/anthropic.md`: `code_execution_20250825`/`20260120`, skills, container uploads
- `docs/models/openai.md`: `FunctionShellToolParam`, hosted shell, skills, local shell mode
- **Costs subsection**: document container billing ($0.05/hr Anthropic, OpenAI TBD) and native tool token overhead (Anthropic: +245 tokens for bash, +700 for text editor per request)

### 13.5 Migration Note

Users currently passing raw shell tool dicts via `openai_builtin_tools=[{"type": "shell", ...}]` can continue to do so — this is **not deprecated**. `ShellTool` is the recommended typed alternative with better type safety, validation, and cross-provider support. No migration is required; both paths coexist.

### 13.6 API Reference

- `docs/api/toolsets/shell.md`
- `docs/api/toolsets/text_editor.md`
- `docs/api/toolsets/apply_patch.md`

---

## 14. Rollout & Phasing

### Per-Phase Checklist (All Phases)

Every phase must complete the following before merge:

- `__all__` exports in new/modified modules; public types re-exported from `pydantic_ai/__init__.py` (api-design rule: export commonly-used types from top-level package)
- `_: KW_ONLY` marker before optional fields in all new dataclasses (api-design rule: prevents breakage when adding parameters)
- Private implementation details prefixed with `_` and excluded from `__all__` (e.g., `_LocalShellExecutor`)
- `make format` and `make typecheck` pass
- `make testcov` passes with 100% coverage
- New docs pages registered in `mkdocs.yml` nav; links verified with `make docs-serve`
- Docstrings on all public types and methods, with backtick-wrapped identifiers
- PR uses the [PR template](.github/pull_request_template.md) with issue refs (#3365, #3963, #3794), AI-generated code checkbox checked
- VCR cassettes recorded and committed (no secrets in diffs)

### Beta Module Question

The [version policy](docs/version-policy.md) says minor releases may introduce features under a `beta` module when the API is not yet stable. `NativeToolDefinition` and the native toolsets (`ShellToolset`, `TextEditorToolset`, `ApplyPatchToolset`) introduce a new abstraction pattern. **Question for maintainers**: should the native toolsets ship under `pydantic_ai.beta.toolsets` initially, or is the design stable enough for the main module? The remote `ShellTool` follows established `AbstractBuiltinTool` patterns and does not need beta. The capability wrapper (Phase 4) is more likely to need beta since it depends on the capabilities abstraction landing.

### Phase 1: Remote Shell + Skills (from #4600) — COMPLETE

This is the existing #4600 work, refined:

1. `ShellTool` builtin with `skills`, `network_policy`
2. `SkillReference`, `CodeExecutionNetworkPolicy` types
3. `UploadedFile.target`, `UploadedFile.part_kind`
4. Anthropic: `code_execution_20250825`/`20260120`, skills beta, container management, result block mapping
5. OpenAI: `FunctionShellToolParam`, container_auto, skills, network policy, file mounting
6. Validation: mutual exclusion, network policy support check
7. `supports_shell_network_policy` profile flag
8. Tests (VCR cassettes for both streaming and non-streaming), docs, docstrings

### Phase 2: Local Shell (`NativeToolDefinition` + `ShellToolset`) — COMPLETE

1. `ShellNativeDefinition`, `TextEditorNativeDefinition`, `ApplyPatchNativeDefinition` types, `NativeToolDefinition` union, `native_definition` field on `ToolDefinition`
2. `ShellToolset`, `ShellExecutor`, `ShellOutput`, `_LocalShellExecutor` in `toolsets/shell.py`
3. Anthropic adapter: `bash_20250124` emission, name resolution via `ToolDefinition`, response mapping
4. OpenAI adapter: `shell` with `environment: {type: "local"}` emission and response mapping
5. Profile flags: `supports_native_shell_tool`
6. Fallback warning: `warnings.warn()` on unsupported providers
7. Tests (VCR cassettes, fallback path unit test), docs (`native-tools.md` with decision tree + security warning)

### Phase 3: Local Text Editor + Apply Patch — COMPLETE

1. `TextEditorToolset` with discriminated union `TextEditorCommand` in `toolsets/text_editor.py`
2. `ApplyPatchToolset` with typed `ApplyPatchOperation` in `toolsets/apply_patch.py`
3. Anthropic adapter: `text_editor_20250728`
4. OpenAI adapter: `apply_patch`
5. Profile flags: `supports_native_text_editor_tool`, `supports_native_apply_patch_tool`
6. Tests, docs

### Phase 4: Capability Wrapper (NEW — does NOT block on #4303)

This phase builds a minimal `Capability` protocol now and implements the `Shell` capability on top of it. When Douwe's richer `Capability` base class lands (#4303), we swap the base class and gain hooks, instructions, and model settings integration. The user-facing API (`capabilities=[Shell(...)]`) stays the same.

#### 4a. Minimal `Capability` Protocol

New file: `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py`

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.toolsets.abstract import AbstractToolset


class Capability(ABC):
    """A high-level agent capability that provides toolsets and/or builtin tools.

    Capabilities are the recommended way to give agents execution abilities.
    They encapsulate routing logic (local vs remote), data privacy defaults,
    and provider-specific configuration.

    This is a minimal protocol that will be extended when #4303 lands to
    also support hooks, instructions, and model settings.
    """

    @abstractmethod
    def toolsets(self) -> list[AbstractToolset]:
        """Return toolsets this capability provides (client-executed)."""
        ...

    @abstractmethod
    def builtin_tools(self) -> list[AbstractBuiltinTool]:
        """Return builtin tools this capability provides (provider-executed)."""
        ...
```

#### 4b. `Agent` Integration

Add `capabilities` parameter to `Agent.__init__`:

```python
class Agent:
    def __init__(
        self,
        model=...,
        *,
        toolsets=(),
        builtin_tools=(),
        capabilities=(),   # NEW
        ...
    ):
        # Decompose capabilities into toolsets + builtin_tools
        for cap in capabilities:
            self._toolsets.extend(cap.toolsets())
            self._builtin_tools.extend(cap.builtin_tools())
        # ... rest of existing init
```

This is a minimal addition — capabilities just decompose into the existing toolset/builtin_tool paths. When #4303 lands, this integration point expands to also extract hooks, instructions, and model settings from capabilities.

#### 4c. `Shell` Capability

New file: `pydantic_ai_slim/pydantic_ai/capabilities/shell.py`

```python
@dataclass(kw_only=True)
class Shell(Capability):
    """Shell execution capability with automatic routing.

    Routing logic:
    - If `cwd` or `executor` is provided → local mode (data stays local)
    - If `skills` or `network_policy` is provided without local context → remote mode
    - If nothing is provided → auto mode (remote if provider supports it, else local fallback)

    Data privacy: local paths always default to local mode to avoid
    accidental data leakage to providers.
    """

    # Local execution config
    cwd: str | Path | None = None
    executor: ShellExecutor | None = None
    env: dict[str, str] | None = None

    # Remote execution config
    skills: Sequence[SkillReference] = ()
    network_policy: CodeExecutionNetworkPolicy | None = None

    # Shared config
    timeout: float | None = None
    max_output_chars: int = 200_000

    def _resolve_mode(self) -> Literal['local', 'remote', 'auto']:
        has_local = self.cwd is not None or self.executor is not None
        has_remote = bool(self.skills) or self.network_policy is not None
        if has_local:
            return 'local'  # data privacy default
        if has_remote and not has_local:
            return 'remote'
        return 'auto'

    def toolsets(self) -> list[AbstractToolset]:
        mode = self._resolve_mode()
        if mode == 'remote':
            return []  # remote-only, no local toolset
        # local or auto: provide ShellToolset
        executor = self.executor or _LocalShellExecutor(cwd=self.cwd, env=self.env)
        return [ShellToolset(
            executor=executor,
            timeout=self.timeout,
            max_output_chars=self.max_output_chars,
        )]

    def builtin_tools(self) -> list[AbstractBuiltinTool]:
        mode = self._resolve_mode()
        if mode == 'local':
            return []  # local-only, no builtin
        # remote or auto: provide ShellTool
        return [ShellTool(
            skills=self.skills,
            network_policy=self.network_policy,
        )]
```

#### 4d. `TextEditor` and `FilePatching` Capabilities

```python
@dataclass(kw_only=True)
class TextEditor(Capability):
    """Text editor capability (Anthropic text_editor_20250728, function fallback on others)."""
    execute: TextEditorExecuteFunc
    max_characters: int | None = None

    def toolsets(self) -> list[AbstractToolset]:
        return [TextEditorToolset(execute=self.execute, max_characters=self.max_characters)]

    def builtin_tools(self) -> list[AbstractBuiltinTool]:
        return []


@dataclass(kw_only=True)
class FilePatching(Capability):
    """File patching capability (OpenAI apply_patch, function fallback on others)."""
    execute: ApplyPatchExecuteFunc

    def toolsets(self) -> list[AbstractToolset]:
        return [ApplyPatchToolset(execute=self.execute)]

    def builtin_tools(self) -> list[AbstractBuiltinTool]:
        return []
```

#### 4e. `CodingEnvironment` Convenience Capability

```python
@dataclass(kw_only=True)
class CodingEnvironment(Capability):
    """Combined shell + text editor + file patching for a local coding environment.

    Provides all execution tools backed by a local directory.
    """
    cwd: str | Path = '.'
    env: dict[str, str] | None = None
    timeout: float | None = None
    max_output_chars: int = 200_000

    def toolsets(self) -> list[AbstractToolset]:
        return [
            ShellToolset.local(cwd=self.cwd, env=self.env,
                               timeout=self.timeout, max_output_chars=self.max_output_chars),
            TextEditorToolset(execute=_local_text_editor(self.cwd)),
            ApplyPatchToolset(execute=_local_apply_patch(self.cwd)),
        ]

    def builtin_tools(self) -> list[AbstractBuiltinTool]:
        return []  # all local
```

#### 4f. Tests

- Routing logic: `Shell(cwd=...)` → local, `Shell(skills=...)` → remote, `Shell()` → auto
- Data privacy: `Shell(cwd='/workspace', skills=[...])` → local (path wins)
- Decomposition: verify `Agent(capabilities=[Shell(...)])` produces correct toolsets + builtin_tools
- Cross-provider Cartesian matrix (section 12.3) for Shell capability
- VCR cassettes for capability-level integration tests

#### 4g. Docs

- Update `docs/native-tools.md` to use capability API as recommended approach
- Demote direct `ShellToolset`/`ShellTool` usage to "Advanced / Low-Level" section
- Add `docs/capabilities.md` with overview of the capability pattern

#### 4h. Migration Path to #4303

When the capabilities PR lands:

1. `Capability` base class gains: `hooks()`, `instructions()`, `model_settings()` methods
2. `Agent` integration expands to extract those from capabilities
3. `Shell` adds hook for built-in ↔ tool call part bridging (local native tools)
4. `Shell` adds instructions (execution environment context for the model)
5. No change to user-facing API: `capabilities=[Shell(cwd='/workspace')]` stays the same

**User-facing API (available NOW, before #4303):**

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Shell, TextEditor, CodingEnvironment

# Simple: local shell
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Shell(cwd='/workspace')])

# Full coding environment (shell + text editor + file patching)
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[
    CodingEnvironment(cwd='/workspace'),
])

# Remote with skills
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[
    Shell(skills=[SkillReference(skill_id='my-skill')]),
])

# Auto: framework picks the best option for the provider
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Shell()])
```

### Phase 4→#4303 Migration (After Capabilities PR lands)

When Douwe's capabilities PR merges:

1. Swap `Capability` base class from our minimal protocol to the richer #4303 base
2. Add `hooks()` to `Shell` for built-in ↔ tool call part bridging
3. Add `instructions()` for execution environment system prompt context
4. Add `model_settings()` for container config pass-through
5. Agent integration automatically picks up hooks/instructions/settings
6. User-facing API unchanged: `capabilities=[Shell(cwd='/workspace')]`

### Future Work (Out of Scope)

These are not phases of this plan but natural follow-ups:

- **WebSearch Capability**: Same pattern as Shell — use provider's web search built-in when available, fall back to Tavily/etc. otherwise. Validates the capability abstraction is generalizable.
- **CodeMode Capability (#4050)**: Programmatic tool calling backed by Monty or provider code execution. Uses the same capability hooks for routing.
- **MemoryTool Refactor**: Replace the Anthropic-specific `MemoryTool` pattern with a capability-driven `Memory` capability. Follows the same pattern established here.
- **Tool Search Integration**: Deferred loading flag on `ToolDefinition` for large skill sets, capability-aware filtering.
- **Execution Environment Integration (Post-#4393)**: `CodingEnvironment` backed by `ExecutionEnvironment` ABC from PR #4393 (Docker, E2B, etc.).

---

## 15. Open Questions

### 15.1 Naming: Capability vs Implementation Layer

With the capability-driven architecture, users primarily interact with `Shell` (capability), not `ShellTool`/`ShellToolset` directly. The `Tool`/`Toolset` distinction becomes an implementation detail.

**Recommendation**: Keep `ShellTool` and `ShellToolset` as internal implementation layers. The `Shell` capability is the public API. Advanced users who need direct access can still use the toolsets, but docs should present the capability as the primary interface.

**Resolved**: The previous decision tree ("Which shell tool do I use?") is replaced by the capability's automatic routing logic.

### 15.2 Anthropic Tool Version Selection

`bash_20250124` vs `bash_20241022`; `text_editor_20250728` vs `text_editor_20250124`.

**Recommendation**: Model profile determines version. Add version fields to Anthropic-specific profile.

### 15.3 Remote + Local Coexistence

Can a single `Shell` capability use both remote and local simultaneously? Anthropic supports this (hosted `bash_code_execution` is distinct from local `bash_20250124`). OpenAI uses the same `shell` tool type for both (differentiated by environment).

**Recommendation**: Support it as an advanced option on the capability (e.g., `Shell(cwd='/workspace', also_remote=True)`). Default behavior should pick one mode. On OpenAI, both emit a `shell` tool but with different environment types, so the adapter can distinguish them.

### 15.4 Container ID for Remote Shell

**Resolved**: Container ID is stored in two provider-specific locations:

- **Anthropic**: `ModelResponse.provider_details['container_id']`
- **OpenAI**: `BuiltinToolCallPart.args['container_id']`

A unified helper `ShellTool.get_container_id(messages)` scans message history in reverse, checking both locations. This helper is used by:

1. **OpenAI adapter** (`_build_shell_tool_param`): auto-switches from `container_auto` to `container_reference` on subsequent turns. The Anthropic adapter already had equivalent logic in `_get_container()`.
2. **User tools** (non-Temporal): `ShellTool.get_container_id(ctx.messages)` provides direct access.
3. **Temporal activities**: `ShellToolTemporalRunContext` (in `durable_exec/temporal/_builtin_tools.py`) overrides `serialize_run_context` to include `container_id` via the same helper. Users opt in with `TemporalAgent(agent, run_context_type=ShellToolTemporalRunContext)`.

**Design rationale**: Container ID is NOT added to `RunContext` (which only has generic agent-loop fields) or the default `TemporalRunContext` serializer (which has zero tool-specific knowledge). The Temporal subclass lives in `durable_exec/temporal/` (not `builtin_tools.py`) to keep the dependency direction correct.

**Force-fresh behavior**: `openai_shell_container=False` strips container references from BOTH the shell tool param (`container_auto` instead of `container_reference`) AND the message history round-trip items (omits `environment` on old `shell_call` input items). Without both, OpenAI infers the old container from conversation context.

**Expired container recovery**: Auto-inferred containers that no longer exist (404 from OpenAI) are automatically retried with `container_auto` + files from `openai_shell_uploaded_files`. Explicitly set containers (`openai_shell_container='cntr_xxx'`) propagate the 404 as `ModelHTTPError`. See section 4.5 for details.

### 15.5 `UploadedFile.target` Shape

`Literal['message', 'container', 'both']` vs `set[str]` for extensibility.

**Recommendation**: Keep `Literal` from #4600 — it's simpler and the three values cover all known use cases.

### 15.6 Mutual Exclusion: `ShellTool` + `CodeExecutionTool`

**Resolved**: Mutual exclusion is correct and required. OpenAI's API errors if both `code_interpreter` and `shell` are present in the same request. Anthropic's `code_execution_20250825` supersedes `code_execution_20250522`.

### 15.7 `SkillReference.source`

Anthropic uses `'custom'` vs `'provider'` (maps to `type: 'custom'` vs `type: 'anthropic'`). OpenAI ignores it.

**Recommendation**: Keep cross-provider as-is from #4600. Ignored gracefully on providers that don't support it.

### 15.8 OpenAI `shell_call` Handling Across Phases

Phase 1 (remote) and Phase 2 (local) both produce OpenAI `shell_call` / `shell_call_output` items. Phase 1 must NOT hardcode `shell_call` → `BuiltinToolCallPart` without a discrimination mechanism, or Phase 2 will break.

**Recommendation**: Discriminate by environment type in the response. The `shell_call` response includes an `environment` field indicating whether it came from a hosted or local environment. Phase 1 maps hosted → `BuiltinToolCallPart`; Phase 2 maps local → `ToolCallPart`.

---

## Appendix A: Provider Tool Format Reference

### Anthropic Remote Shell (code_execution_20260120 GA / code_execution_20250825 beta)

**Request (GA — no beta header needed):**

```json
{
  "tools": [{"type": "code_execution_20260120", "name": "code_execution"}],
  "container": {
    "skills": [{"skill_id": "my_skill", "type": "custom", "version": "1"}]
  }
}
```

**Request (beta):**

```json
{
  "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
  "betas": ["code-execution-2025-08-25", "skills-2025-10-02"],
  "container": {
    "skills": [{"skill_id": "my_skill", "type": "custom", "version": "1"}]
  }
}
```

**Response contains (server-executed — client does nothing):**

- `server_tool_use` blocks (IDs prefixed `srvtoolu_`) for `bash_code_execution` and `text_editor_code_execution`
- `bash_code_execution_tool_result` — output of shell commands run in container
- `text_editor_code_execution_tool_result` — output of file operations in container
- `BetaCodeExecutionResultBlock` — file outputs with `file_id`s
- `container.id` for session persistence

### Anthropic Local Bash (bash_20250124)

**Request:**

```json
{"type": "bash_20250124", "name": "bash"}
```

**Model output:**

```json
{"type": "tool_use", "id": "toolu_xxx", "name": "bash", "input": {"command": "ls -la"}}
```

**Client result:**

```json
{"type": "tool_result", "tool_use_id": "toolu_xxx", "content": "total 42\n..."}
```

### Anthropic Local Text Editor (text_editor_20250728)

**Request:**

```json
{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10000}
```

**Model output:**

```json
{
  "type": "tool_use", "id": "toolu_xxx", "name": "str_replace_based_edit_tool",
  "input": {"command": "str_replace", "path": "main.py", "old_str": "foo", "new_str": "bar"}
}
```

### OpenAI Remote Shell (FunctionShellToolParam)

**Request:**

```json
{
  "type": "shell",
  "environment": {
    "type": "container_auto",
    "skills": [{"skill_id": "..."}],
    "network_policy": {"type": "allowlist", "allowed_domains": ["example.com"]},
    "file_ids": ["file-xxx"]
  }
}
```

**Response:** `ResponseFunctionShellToolCall` / `ResponseFunctionShellToolCallOutput`

### OpenAI Local Shell

**Request:** `{"type": "shell", "environment": {"type": "local"}}`
**Response:** `shell_call` → client executes → `shell_call_output` (same item types as hosted mode)

### OpenAI Apply Patch

**Request:** `{"type": "apply_patch"}`
**Response:** `apply_patch_call` with V4A diffs → client applies → `apply_patch_call_output`

---

## Appendix B: Capability Matrix


| Feature                | Anthropic                                                          | OpenAI Responses                                                                                     | OpenAI Chat | Google | Groq | Other |
| ---------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------- | ------ | ---- | ----- |
| Remote shell (hosted)  | `code_execution_20260120` (GA) / `code_execution_20250825` (beta)  | `shell` with `container_auto`/`container_reference`                                                  | No          | No     | No   | No    |
| Remote sub-tools       | `bash_code_execution` + `text_editor_code_execution` (server-side) | Shell commands via `shell_call`/`shell_call_output` (server-side)                                    | No          | No     | No   | No    |
| Skills                 | `skills-2025-10-02` beta                                           | Via containers (`client.containers.create(skills=[...])`)                                            | No          | No     | No   | No    |
| Local bash             | `bash_20250124` (client-executed, separate tool type)              | No                                                                                                   | No          | No     | No   | No    |
| Local shell            | No                                                                 | `shell` with `environment: {type: "local"}` (client-executed, same `shell_call`/`shell_call_output`) | No          | No     | No   | No    |
| Local text editor      | `text_editor_20250728` (client-executed)                           | No                                                                                                   | No          | No     | No   | No    |
| Local apply_patch      | No                                                                 | `apply_patch` (GPT-5.1+, client-executed)                                                            | No          | No     | No   | No    |
| Function tool fallback | Yes                                                                | Yes                                                                                                  | Yes         | Yes    | Yes  | Yes   |


---

## Appendix C: Version Compatibility

### Anthropic


| Model                       | Bash Version    | Text Editor Version    | Text Editor Name              |
| --------------------------- | --------------- | ---------------------- | ----------------------------- |
| Claude 4.x                  | `bash_20250124` | `text_editor_20250728` | `str_replace_based_edit_tool` |
| Claude Sonnet 3.7           | `bash_20250124` | `text_editor_20250124` | `str_replace_editor`          |
| Claude Sonnet 3.5 (retired) | `bash_20241022` | `text_editor_20241022` | `str_replace_editor`          |


### OpenAI


| Model            | Shell (hosted + local) | Apply Patch | Notes                       |
| ---------------- | ---------------------- | ----------- | --------------------------- |
| GPT-5.4+         | Yes (both modes)       | Yes         | Full shell support          |
| GPT-5.1-5.3      | Yes (hosted)           | Yes         | Local mode availability TBD |
| GPT-4o and below | No                     | No          | Function tool fallback only |


---

## Appendix D: Files Changed

### Phase 1 (Remote Shell, from #4600)

- `pydantic_ai_slim/pydantic_ai/builtin_tools.py` — `ShellTool`, `SkillReference`, `CodeExecutionNetworkPolicy`
- `pydantic_ai_slim/pydantic_ai/__init__.py` — re-exports
- `pydantic_ai_slim/pydantic_ai/messages.py` — `UploadedFile.target`, `.part_kind`, `ModelResponsePart` union
- `pydantic_ai_slim/pydantic_ai/models/__init__.py` — validation (`_validate_builtin_tools`)
- `pydantic_ai_slim/pydantic_ai/models/anthropic.py` — `code_execution_20250825`, skills, container management, new result blocks
- `pydantic_ai_slim/pydantic_ai/models/openai.py` — `FunctionShellToolParam`, `_OpenAICodeExecutionContext`, shell tool synthesis, container management
- `pydantic_ai_slim/pydantic_ai/profiles/__init__.py` — `supports_shell_network_policy`
- `pydantic_ai_slim/pydantic_ai/_agent_graph.py` — handle `UploadedFile` in response parts
- `pydantic_ai_slim/pydantic_ai/ui/` — `UploadedFile` target-aware rendering
- `docs/builtin-tools.md`, `docs/input.md`, `docs/models/anthropic.md`, `docs/models/openai.md`
- `tests/` — anthropic, openai, openai_responses, builtin_tools, vercel AI, model_request_parameters

### Phase 2 (Local Shell, New)

- `pydantic_ai_slim/pydantic_ai/tools.py` — `ShellNativeDefinition`, `TextEditorNativeDefinition`, `ApplyPatchNativeDefinition`, `NativeToolDefinition` union, `native_definition` field
- `pydantic_ai_slim/pydantic_ai/toolsets/shell.py` — `ShellToolset`, `ShellOutput`, `ShellExecuteFunc`
- `pydantic_ai_slim/pydantic_ai/models/anthropic.py` — `bash_20250124` emission, name mapping
- `pydantic_ai_slim/pydantic_ai/models/openai.py` — `shell` (local mode) emission, name mapping
- `pydantic_ai_slim/pydantic_ai/profiles/` — `supports_native_shell_tool`
- `docs/native-tools.md`, `docs/toolsets.md`
- `tests/` — new cassettes, cross-provider parametrization

### Phase 3 (Text Editor + Apply Patch, New)

- `pydantic_ai_slim/pydantic_ai/toolsets/text_editor.py`
- `pydantic_ai_slim/pydantic_ai/toolsets/apply_patch.py`
- `pydantic_ai_slim/pydantic_ai/models/anthropic.py` — `text_editor_20250728`
- `pydantic_ai_slim/pydantic_ai/models/openai.py` — `apply_patch`
- `pydantic_ai_slim/pydantic_ai/profiles/` — additional flags
- `docs/`, `tests/`

### Phase 4 (Capability Wrapper, New)

- `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` — `Capability` protocol, re-exports
- `pydantic_ai_slim/pydantic_ai/capabilities/shell.py` — `Shell` capability with routing logic
- `pydantic_ai_slim/pydantic_ai/capabilities/text_editor.py` — `TextEditor` capability
- `pydantic_ai_slim/pydantic_ai/capabilities/file_patching.py` — `FilePatching` capability
- `pydantic_ai_slim/pydantic_ai/capabilities/coding_env.py` — `CodingEnvironment` convenience capability
- `pydantic_ai_slim/pydantic_ai/agent.py` — `capabilities` parameter on `Agent.__init__`
- `pydantic_ai_slim/pydantic_ai/__init__.py` — re-export `Shell`, `TextEditor`, `FilePatching`, `CodingEnvironment`
- `docs/capabilities.md` — capability pattern overview
- `docs/native-tools.md` — updated to recommend capability API
- `tests/test_capabilities.py` — routing logic, decomposition, data privacy defaults
- `tests/test_capabilities_integration.py` — VCR cassettes through capability API

