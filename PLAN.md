# PLAN: Provider-Native Shell Tools — Remote & Local Execution with Skills

> **Issue refs:** #3365 (Anthropic/OpenAI Skills), #3963 (Shell/Bash builtin), #3794 (Text Editor tool)
> **Related PRs:** #4393 (Execution Environments), #4505 (ShellTool PLAN), #4513 (TextEditorTool builtin), #4600 (Anthropic Skills draft — absorbed into this plan)
> **Upstream vision:** #4303 (Capabilities abstraction), #4050 (Code Mode)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background & Landscape](#2-background--landscape)
3. [Architectural Analysis: Two Tracks](#3-architectural-analysis-two-tracks)
4. [Track 1: Remote Shell — `ShellTool` Builtin](#4-track-1-remote-shell--shelltool-builtin)
5. [Track 2: Local Shell — Native Toolsets](#5-track-2-local-shell--native-toolsets)
6. [Provider Adapter Changes](#6-provider-adapter-changes)
7. [Agent Loop & Message Changes](#7-agent-loop--message-changes)
8. [Error Handling](#8-error-handling)
9. [Security Considerations](#9-security-considerations)
10. [Relationship to PR #4393](#10-relationship-to-pr-4393)
11. [Relationship to Other PRs](#11-relationship-to-other-prs)
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

| | Remote (Provider-Hosted) | Local (Client-Executed) |
|---|---|---|
| **Who runs it** | Provider's container infrastructure | Developer's local machine / Docker |
| **Pydantic AI pattern** | `AbstractBuiltinTool` (server-executed) | `AbstractToolset` (client-executed) |
| **Skills support** | Yes — skills run in the container | OpenAI: Yes (`LocalEnvironmentParam` has `skills` field); Anthropic: No |
| **Anthropic** | `code_execution_20260120` (GA) / `code_execution_20250825` (beta) — provides `bash_code_execution` + `text_editor_code_execution` server-side sub-tools, plus `skills-2025-10-02` beta | `bash_20250124`, `text_editor_20250728` (fully client-executed, separate from hosted container) |
| **OpenAI** | `shell` with `container_auto` / `container_reference` (server-executed) | `shell` with `environment: {type: "local"}` (client-executed, same tool type!), `apply_patch` |
| **Use case** | "Let the provider run code for me" | "I want to run shell commands locally with optimal model performance" |

**Important architectural note for Anthropic**: The hosted `bash_code_execution` (runs in Anthropic's container, returns `server_tool_use` blocks) is **completely separate** from the local `bash_20250124` (runs on client's machine, returns `tool_use` blocks). State is NOT shared between them. A model can use both simultaneously — they appear as two different execution environments.

### Why Tool Names and Parameters Matter

These models are **specifically fine-tuned** on their own tool harnesses:

| Provider | Local Shell | Local Text Editor | Local Patch | Remote Shell (Hosted) |
|----------|------------|------------------|-------------|----------------------|
| Anthropic | `bash` (`bash_20250124`) — client-executed, separate tool type | `str_replace_based_edit_tool` (`text_editor_20250728`) — client-executed | N/A | `code_execution_20260120` (GA) / `code_execution_20250825` (beta) — server-executed `bash_code_execution` + `text_editor_code_execution` sub-tools |
| OpenAI | `shell` with `environment: {type: "local"}` — client-executed (same `shell_call`/`shell_call_output` as hosted) | N/A | `apply_patch` (V4A diffs) — client-executed | `shell` with `container_auto`/`container_reference` — server-executed (same tool type as local!) |

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

| Version | Beta Header | Notes |
|---------|------------|-------|
| `code_execution_20260120` | None (GA) | Latest, recommended, no beta needed |
| `code_execution_20250825` | `code-execution-2025-08-25` | Original hosted shell beta |
| `code_execution_20250522` | `code-execution-2025-05-22` | Legacy Python-only sandbox |

The `20250825`+ versions provide two **server-side sub-tools**:
- `bash_code_execution` — runs shell commands in the container
- `text_editor_code_execution` — file operations (view, create, str_replace, insert)

Response blocks use `server_tool_use` type (IDs prefixed `srvtoolu_`), with results as `bash_code_execution_tool_result` and `text_editor_code_execution_tool_result`.

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

## 3. Architectural Analysis: Two Tracks

### 3.1 Track 1: Remote Shell → `ShellTool` (Builtin)

Provider-hosted container execution fits the existing `AbstractBuiltinTool` pattern perfectly:
- The provider runs code in their container — we don't execute anything
- Results come back as `BuiltinToolCallPart` + `BuiltinToolReturnPart`
- Skills, network policy, and file mounting are container-level configuration
- This is exactly what #4600 implements

**Verdict**: `ShellTool` as `AbstractBuiltinTool` is the right pattern for remote execution.

### 3.2 Track 2: Local Shell → Native Toolsets

Client-executed tools that need provider-native presentation don't fit either existing pattern:
- Can't be `AbstractBuiltinTool` — the agent loop doesn't execute builtins
- Can't be plain `AbstractToolset` — adapters convert toolsets to generic function tools

We need a bridge: **native toolsets** that are client-executed (toolset path) but presented in provider-native format when supported. This requires a `NativeToolDefinition` on `ToolDefinition` that model adapters can recognize and translate.

### 3.3 How the Two Tracks Coexist

```
User wants remote execution (provider container):
  → ShellTool(skills=[...], network_policy=...) [builtin tool]
  → Provider runs code, returns BuiltinToolCallPart/ReturnPart
  → Agent loop records in history, no execution

User wants local execution (their machine/Docker):
  → ShellToolset(execute=my_callback) [native toolset]
  → Adapter emits provider-native format (bash_20250124, shell with local env)
  → Model returns ToolCallPart (normalized by adapter)
  → Agent loop executes via toolset.call_tool()
```

These are independent — a user picks one or both based on their needs. They share no code paths besides the message format.

---

## 4. Track 1: Remote Shell — `ShellTool` Builtin

This section describes the remote/hosted shell support, largely carried forward from #4600.

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
- `UploadedFile(target='container')` → collected by `_OpenAICodeExecutionContext`, merged into `environment.file_ids` on the `container_auto` shell tool
- `openai_shell_container='cntr_xxx'` → `container_reference` environment type (cannot combine with skills, network policy, or file uploads — those require `container_auto`)
- `openai_shell_container=False` → force fresh `container_auto`, ignore history

Container reuse across turns:
- **Implicit**: message history round-tripping (OpenAI's `previous_response_id` handles this natively)
- **Explicit**: `openai_shell_container='cntr_xxx'` for containers created via `client.containers.create()`

#### Comparison

| Aspect | Anthropic | OpenAI |
|--------|-----------|--------|
| File upload mechanism | Message content blocks | Environment `file_ids` |
| When files are sent | Per-message (can change each turn) | At container creation / per-request |
| Container reuse | Auto from history + `anthropic_container` setting | Implicit via `previous_response_id` + `openai_shell_container` setting |
| Pre-created containers | Via `anthropic_container: {'id': '...'}` | Via `openai_shell_container: 'cntr_xxx'` using `container_reference` |
| Container + skills conflict | No conflict (skills in `container.skills`) | `container_reference` cannot combine with skills/network_policy/file_ids (use `container_auto`) |

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
- Container reuse: `openai_shell_container` setting → `container_reference` with explicit `container_id`, or `container_auto` for fresh
- `_OpenAICodeExecutionContext` tracks transport type and container-target uploaded files
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
    - None (default): fresh container_auto each time.
    - str (e.g. 'cntr_xxx'): reuse via container_reference.
    - False: force fresh container, ignore history.
    Cannot combine container_reference with skills, network_policy, or file uploads.
    """
```

**Important**: OpenAI uses the **same `shell` tool type** for both hosted and local execution. The difference is the `environment` type: `container_auto`/`container_reference` for hosted, `local` for client-executed. Both use `shell_call`/`shell_call_output` output items. This means `ShellTool` (builtin, remote) and `ShellToolset` (toolset, local) will both emit an OpenAI `shell` tool — differentiated by environment type.

---

## 5. Track 2: Local Shell — Native Toolsets

This section describes the local/client-executed tool support, which is new to this plan.

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

**`call_tool()` handles:**
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

**`get_tools()` emits:**
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

| Change | Track | Source |
|--------|-------|--------|
| `ShellTool` → `BetaCodeExecutionTool20250825Param` | Remote | #4600 |
| Skills → `BetaSkillParams` in container | Remote | #4600 |
| Container ID persistence | Remote | #4600 |
| `BetaBashCodeExecutionToolResultBlock` mapping | Remote | #4600 |
| `BetaTextEditorCodeExecutionToolResultBlock` mapping | Remote | #4600 |
| `BetaContainerUploadBlockParam` from `UploadedFile(target='container')` | Remote | #4600 |
| `native_definition.kind='shell'` → `bash_20250124` | Local | New |
| `native_definition.kind='text_editor'` → `text_editor_20250728` | Local | New |
| Name mapping (bash ↔ shell, str_replace_based_edit_tool ↔ toolset name) | Local | New |

### 6.2 OpenAI Responses (`models/openai.py`)

| Change | Track | Source |
|--------|-------|--------|
| `ShellTool` → `FunctionShellToolParam` (container_auto) | Remote | #4600 |
| Skills → `SkillReferenceParam` in environment | Remote | #4600 |
| Network policy → `OpenAIContainerAutoNetworkPolicy` | Remote | #4600 |
| File mounting → `file_ids` in environment | Remote | #4600 |
| `_OpenAICodeExecutionContext` for transport tracking | Remote | #4600 |
| `ResponseFunctionShellToolCall` / Output mapping | Remote | #4600 |
| `openai_shell_uploaded_files`, `openai_shell_container` settings | Remote | #4600 |
| `native_definition.kind='shell'` → `shell` with `environment: {type: "local"}` | Local | New |
| `native_definition.kind='apply_patch'` → `apply_patch` (GPT-5.1+) | Local | New |
| Name mapping for local tools | Local | New |

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

| Scenario | Handling | Pattern |
|----------|----------|---------|
| Command timeout | `ModelRetry(f'Command timed out after {timeout}s.')` | Same as `FunctionToolset` timeout in `toolsets/function.py` |
| Non-zero exit code | Return `ShellOutput(output=stderr+stdout, exit_code=N)` — let the model decide how to proceed | Not an error — exit codes are informational |
| Output exceeds `max_output_chars` | Truncate and append `'[output truncated — {shown} chars of {total} total]'` | Toolset-level truncation |
| `execute()` raises unexpected exception | Propagate as-is (not `ModelRetry`) — framework-level exception | Follows `FunctionToolset` pattern |

### 8.2 Remote Shell Errors (Track 1)

| Scenario | Handling | Pattern |
|----------|----------|---------|
| Container expired / not found | Provider returns API error → adapter raises `ModelHTTPError` or similar | Standard API error handling |
| Stale `container_id` in history | Adapter catches container-not-found, clears `container_id` from settings, retries with fresh container | Graceful degradation |
| Invalid `file_id` in `UploadedFile` | Provider returns API error → surfaces as request failure | Standard API error handling |
| Skills not available on model | `UserError` at configuration time (checked via `supported_builtin_tools` in `prepare_request`) | Same as existing builtin tool validation |
| `ShellTool.network_policy` on unsupported provider | `UserError` at configuration time | Same pattern as existing profile flag checks |

### 8.3 Concurrency

- **Remote containers**: Concurrent `agent.run()` calls sharing a container are the provider's responsibility. Both Anthropic and OpenAI handle concurrent requests to the same container safely.
- **Local shell (`_LocalShellExecutor`)**: The persistent subprocess is inherently sequential. The `ToolDefinition` sets `sequential=True`, which causes the agent loop to serialize tool calls to this tool rather than running them concurrently.
- **Multiple agent instances**: If two `Agent` instances share the same `ShellToolset` instance, their calls are serialized via `sequential=True`. For true parallelism, use separate `ShellToolset` instances.

### 8.4 Configuration Errors

| Scenario | Handling |
|----------|----------|
| `ShellTool` + `CodeExecutionTool` | `UserError('ShellTool and CodeExecutionTool are mutually exclusive...')` |
| `UploadedFile(target='container')` without `ShellTool` | `UserError('UploadedFile with target including container requires ShellTool...')` |
| `openai_shell_container` + skills/network_policy/file_ids | `UserError('container_reference cannot combine with skills, network_policy, or file uploads...')` |
| `ShellTool` on unsupported provider | `UserError` via existing `supported_builtin_tools` validation |

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
- **`ModelRetry` on timeout**: Model can adjust its approach rather than hanging
- **`ApprovalRequiredToolset` composition**: Users can wrap `ShellToolset` in `ApprovalRequiredToolset` for human-in-the-loop approval:

```python
from pydantic_ai.toolsets import ShellToolset, ApprovalRequiredToolset

shell = ShellToolset.local(cwd='/workspace')
approved_shell = ApprovalRequiredToolset(wrapped=shell)
agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[approved_shell])
```

- **`FilteredToolset` composition**: Users can restrict which commands are available
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

| Concern | This Plan | PR #4393 |
|---------|-----------|----------|
| Where code runs | Provider containers (remote) OR local via callback (local) | Framework-managed environments (local, Docker, memory) |
| How tools appear | Provider-native formats + function fallback | Generic function tools only |
| Key files | `builtin_tools.py`, `tools.py`, `models/`, `toolsets/shell.py`, `messages.py` | `environments/`, `toolsets/execution_environment.py` |

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

## 11. Relationship to Other PRs

### 11.1 PR #4505 (ShellTool PLAN.md by @bsherifi)

Our `NativeToolDefinition` generalizes their `native_kind: Literal['shell']` concept, adding `config` for tools that need extra parameters (text editor's `max_characters`). This plan supersedes #4505 while acknowledging the contribution.

### 11.2 PR #4513 (TextEditorTool by @st4r0)

Their approach follows the `MemoryTool` pattern (builtin + companion function tool). The toolset approach is architecturally cleaner for client-executed tools. Their test cassettes and Anthropic adapter changes are valuable references.

### 11.3 #4303 (Capabilities Abstraction)

This plan's `ShellTool` (remote) and `ShellToolset` (local) are the implementation layer that a future `Shell()` capability would delegate to:

```
Shell (Capability, v2)
├── ShellTool (remote, provider-hosted) when provider supports containers
├── ShellToolset (local, client-executed) when user provides local execution
├── ExecutionEnvironmentToolset (local, from #4393) for environment-backed execution
└── Adds instructions, hooks, guardrails, compaction on top
```

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

### 12.3 Cross-Provider Parametrization

- Shell: Anthropic, OpenAI Responses, generic fallback
- Text editor: Anthropic, generic fallback
- Apply patch: OpenAI, generic fallback

### 12.4 Snapshot Tests

- `result.all_messages()` for full tool call/return flow
- Provider-specific request payloads (verify native format emission)

---

## 13. Documentation

### 13.1 Updated: `docs/builtin-tools.md`

- Add `ShellTool` section with skills, network policy, file mounting
- Document `SkillReference` and `CodeExecutionNetworkPolicy`
- Document `UploadedFile.target` for container uploads
- Mutual exclusion with `CodeExecutionTool`

### 13.2 New: `docs/native-tools.md`

- **Decision tree at the top**: "Which shell tool do I use?" (remote vs local)
- **Security warning**: Prominent admonition about LLM-driven shell execution risks
- Explain native vs function tools and why it matters (models fine-tuned on provider schemas)
- `ShellToolset`, `TextEditorToolset`, `ApplyPatchToolset` usage with hello-world examples
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

- [ ] `__all__` exports in new/modified modules; public types re-exported from `pydantic_ai/__init__.py` (api-design rule: export commonly-used types from top-level package)
- [ ] `_: KW_ONLY` marker before optional fields in all new dataclasses (api-design rule: prevents breakage when adding parameters)
- [ ] Private implementation details prefixed with `_` and excluded from `__all__` (e.g., `_LocalShellExecutor`)
- [ ] `make format` and `make typecheck` pass
- [ ] `make testcov` passes with 100% coverage
- [ ] New docs pages registered in `mkdocs.yml` nav; links verified with `make docs-serve`
- [ ] Docstrings on all public types and methods, with backtick-wrapped identifiers
- [ ] PR uses the [PR template](.github/pull_request_template.md) with issue refs (#3365, #3963, #3794), AI-generated code checkbox checked
- [ ] VCR cassettes recorded and committed (no secrets in diffs)

### Beta Module Question

The [version policy](docs/version-policy.md) says minor releases may introduce features under a `beta` module when the API is not yet stable. `NativeToolDefinition` and the native toolsets (`ShellToolset`, `TextEditorToolset`, `ApplyPatchToolset`) introduce a new abstraction pattern. **Question for maintainers**: should Track 2 (native toolsets) ship under `pydantic_ai.beta.toolsets` initially, or is the design stable enough for the main module? Track 1 (remote `ShellTool`) follows established `AbstractBuiltinTool` patterns and does not need beta.

### Phase 1: Remote Shell + Skills (from #4600)

This is the existing #4600 work, refined:

1. `ShellTool` builtin with `skills`, `network_policy`
2. `SkillReference`, `CodeExecutionNetworkPolicy` types
3. `UploadedFile.target`, `UploadedFile.part_kind`
4. Anthropic: `code_execution_20250825`/`20260120`, skills beta, container management, result block mapping
5. OpenAI: `FunctionShellToolParam`, container_auto, skills, network policy, file mounting
6. Validation: mutual exclusion, network policy support check
7. `supports_shell_network_policy` profile flag
8. Tests (VCR cassettes for both streaming and non-streaming), docs, docstrings

### Phase 2: Local Shell (`NativeToolDefinition` + `ShellToolset`)

1. `ShellNativeDefinition`, `TextEditorNativeDefinition`, `ApplyPatchNativeDefinition` types, `NativeToolDefinition` union, `native_definition` field on `ToolDefinition`
2. `ShellToolset`, `ShellExecutor`, `ShellOutput`, `_LocalShellExecutor` in `toolsets/shell.py`
3. Anthropic adapter: `bash_20250124` emission, name resolution via `ToolDefinition`, response mapping
4. OpenAI adapter: `shell` with `environment: {type: "local"}` emission and response mapping
5. Profile flags: `supports_native_shell_tool`
6. Fallback warning: `warnings.warn()` on unsupported providers
7. Tests (VCR cassettes, fallback path unit test), docs (`native-tools.md` with decision tree + security warning)

### Phase 3: Local Text Editor + Apply Patch

1. `TextEditorToolset` with discriminated union `TextEditorCommand` in `toolsets/text_editor.py`
2. `ApplyPatchToolset` with typed `ApplyPatchOperation` in `toolsets/apply_patch.py`
3. Anthropic adapter: `text_editor_20250728`
4. OpenAI adapter: `apply_patch`
5. Profile flags: `supports_native_text_editor_tool`, `supports_native_apply_patch_tool`
6. Tests, docs

### Future Work (Out of Scope)

These are not phases of this plan but natural follow-ups:

- **Execution Environment Integration (Post-#4393)**: `NativeExecutionEnvironmentToolset` convenience wrapper that backs `ShellToolset` / `TextEditorToolset` with `ExecutionEnvironment` from PR #4393. Depends on #4393 landing.
- **Capabilities Layer (#4303)**: Native toolsets become backends for `Shell()`, `FileSystem()`, `CodeMode()` capabilities. Depends on the Capabilities abstraction design.

---

## 15. Open Questions

### 15.1 Naming: `ShellToolset` vs `ShellTool`

`ShellTool` (builtin, remote) follows the `*Tool` pattern (`WebSearchTool`, `CodeExecutionTool`). `ShellToolset` (toolset, local) follows the `*Toolset` pattern (`FunctionToolset`, `CombinedToolset`). The names differ by suffix, but the conceptual distinction (remote vs local) is non-obvious.

**Recommendation**: Keep the naming as-is — `ShellTool` and `ShellToolset`. The `Tool`/`Toolset` suffix distinction is consistent with the codebase. To address confusion, documentation must include a prominent decision tree:

> **Which shell tool do I use?**
> - Want the **provider to run code** in their hosted container? → `ShellTool` (builtin)
> - Want to **run commands on your machine** (or your Docker)? → `ShellToolset` (toolset)

### 15.2 Anthropic Tool Version Selection

`bash_20250124` vs `bash_20241022`; `text_editor_20250728` vs `text_editor_20250124`.

**Recommendation**: Model profile determines version. Add version fields to Anthropic-specific profile.

### 15.3 `ShellTool` + `ShellToolset` Coexistence

Can a user use both remote (`ShellTool`) and local (`ShellToolset`) simultaneously?

**Recommendation**: Allow it — they serve different purposes and produce different message types (`BuiltinToolCallPart` vs `ToolCallPart`). On OpenAI, both emit a `shell` tool but with different environment types (`container_auto` vs `local`), so the adapter can distinguish them. The model sees them as two separate tools.

### 15.4 Container ID for Remote Shell

**Resolved** (from #4600): Store in `ModelResponse.provider_details['container_id']`. Anthropic adapter extracts from history. OpenAI uses `container_reference` or implicit history round-tripping.

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

| Feature | Anthropic | OpenAI Responses | OpenAI Chat | Google | Groq | Other |
|---------|-----------|-----------------|-------------|--------|------|-------|
| Remote shell (hosted) | `code_execution_20260120` (GA) / `code_execution_20250825` (beta) | `shell` with `container_auto`/`container_reference` | No | No | No | No |
| Remote sub-tools | `bash_code_execution` + `text_editor_code_execution` (server-side) | Shell commands via `shell_call`/`shell_call_output` (server-side) | No | No | No | No |
| Skills | `skills-2025-10-02` beta | Via containers (`client.containers.create(skills=[...])`) | No | No | No | No |
| Local bash | `bash_20250124` (client-executed, separate tool type) | No | No | No | No | No |
| Local shell | No | `shell` with `environment: {type: "local"}` (client-executed, same `shell_call`/`shell_call_output`) | No | No | No | No |
| Local text editor | `text_editor_20250728` (client-executed) | No | No | No | No | No |
| Local apply_patch | No | `apply_patch` (GPT-5.1+, client-executed) | No | No | No | No |
| Function tool fallback | Yes | Yes | Yes | Yes | Yes | Yes |

---

## Appendix C: Version Compatibility

### Anthropic

| Model | Bash Version | Text Editor Version | Text Editor Name |
|-------|-------------|--------------------|--------------------|
| Claude 4.x | `bash_20250124` | `text_editor_20250728` | `str_replace_based_edit_tool` |
| Claude Sonnet 3.7 | `bash_20250124` | `text_editor_20250124` | `str_replace_editor` |
| Claude Sonnet 3.5 (retired) | `bash_20241022` | `text_editor_20241022` | `str_replace_editor` |

### OpenAI

| Model | Shell (hosted + local) | Apply Patch | Notes |
|-------|----------------------|-------------|-------|
| GPT-5.4+ | Yes (both modes) | Yes | Full shell support |
| GPT-5.1-5.3 | Yes (hosted) | Yes | Local mode availability TBD |
| GPT-4o and below | No | No | Function tool fallback only |

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
