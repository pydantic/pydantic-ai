# Shell Tool — Native Provider Support

## Problem

Anthropic (`bash_20250124`) and OpenAI (`local_shell`) ship schema-less shell tools that their models are trained to use. These are developer-executed — you run the command, not the provider. Today pydantic-ai has no way to use them; Anthropic's raises `NotImplementedError`, OpenAI's is silently dropped.

## Approach: Toolset with `native_kind`

Shell doesn't fit the builtin tool system — builtins are server-executed, shell is not. It fits the toolset system: local execution via `call_tool()`, results via `ToolReturnPart`. The missing piece is telling model adapters to use the native wire format instead of a regular function tool.

New field on `ToolDefinition`:

```python
native_kind: NativeToolKind | None = None  # NativeToolKind = Literal['shell']
```

When an adapter sees `native_kind='shell'`, it emits the native tool definition (`BetaToolBash20250124Param` / `LocalShell`) and remaps names in responses. When an adapter doesn't recognize the `native_kind`, it falls back to a regular function tool with the JSON schema. The tool works everywhere; native formatting is a progressive enhancement.

This is orthogonal to both `kind` (execution semantics) and `prefer_builtin` (#4342, fallback direction). Extensible to `'text_editor'` etc. later.

## User API

```python
from pydantic_ai.toolsets import ShellTool, ShellOutput

# Callback
async def my_shell(*, command: str, restart: bool = False) -> ShellOutput:
    proc = await asyncio.create_subprocess_shell(command, ...)
    stdout, stderr = await proc.communicate()
    return ShellOutput(output=stdout.decode() + stderr.decode(), exit_code=proc.returncode or 0)

agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[ShellTool(execute=my_shell)])

# Or subclass
class DockerShell(ShellTool):
    async def execute(self, *, command: str, restart: bool = False) -> ShellOutput:
        ...
```

`ShellOutput` is `output: str` + `exit_code: int`. Both providers accept a flat string for results, so keeping it simple. Non-zero exit codes map to `is_error=True` on the tool result.

## Provider parameter normalization

The callback always gets `command: str` + `restart: bool`. Adapters normalize:

- **Anthropic**: pass through (already `{command: str, restart: bool}`)
- **OpenAI**: `action.command` is `List[str]` (argv tokens) — join via `shlex.join()`. Drop `env`, `working_directory`, `timeout_ms`, `user` (execution-context concerns the developer controls, not the model)
- **Fallback**: matches the JSON schema directly

`restart` is Anthropic-specific but harmless in the fallback schema — other models just won't use it.

## Adapter changes

### Anthropic

The bash tool response comes back as a regular `BetaToolUseBlock` (not `BetaServerToolUseBlock` — that's `bash_code_execution`, a different server-side feature). So it already flows through the `ToolCallPart` path. Changes:

- **Tool defs**: emit `BetaToolBash20250124Param` for `native_kind='shell'`, maintain a `"bash" ↔ "shell"` name map
- **Responses**: apply name map in existing `BetaToolUseBlock` handling (both streaming and non-streaming)
- **History replay**: reverse map `"shell"` → `"bash"` in `_map_message()`. Results use standard `tool_result` format — no change needed

Don't touch `_map_server_tool_use_block` or the `bash_code_execution` `NotImplementedError`.

### OpenAI Responses

- **Tool defs**: emit `{'type': 'local_shell'}` for `native_kind='shell'`
- **Responses**: replace the `LocalShellCall` no-op with `ToolCallPart(tool_name='shell', args={'command': shlex.join(...)})`. No streaming-specific handling needed — `LocalShellCall` arrives as a complete item
- **History replay**: `ToolReturnPart` for shell → `local_shell_call_output` (not `function_call_output`). Needs the item `id` from the corresponding `ToolCallPart`, not the `call_id`

### Other adapters

No changes. `native_kind` is ignored, tool is sent as a regular function tool.

## Scope

**Changes**: `ToolDefinition` (one field), new `toolsets/shell.py`, Anthropic adapter, OpenAI Responses adapter.

**No changes**: `AbstractBuiltinTool`, agent loop, `ToolManager`, `ModelRequestParameters`, other adapters.

## Testing

- ShellTool unit tests (callback, subclass, error handling)
- Anthropic + OpenAI integration tests with VCR (streaming + non-streaming)
- Fallback: non-native provider receives function tool, execution works
- Name mapping round-trips across conversation turns
- Edge cases: non-zero exit, `restart=True`, empty output

## Deferred

- OpenAI hosted shell (`container_auto`) — server-executed, separate concern
- `native_kind='text_editor'`
- `ExecutionEnvironmentToolset` (#4153) adopting `native_kind='shell'`
