Implement Phase 2 of PLAN.md — Local Shell (NativeToolDefinition + ShellToolset).

Read PLAN.md §5 (through §5.8) and §14 Phase 2 for full spec. Read PROGRESS.md
for Phase 1 context. Read the coding guidelines in agent_docs/index.md and the
directory-specific AGENTS.md files.

Summary of what Phase 2 adds:

1. NativeToolDefinition types in a new module (ShellNativeDefinition,
   TextEditorNativeDefinition, ApplyPatchNativeDefinition) + union type +
   `native_definition` field on ToolDefinition (in tools.py)

2. ShellToolset in new file toolsets/shell.py:
   - ShellExecutor protocol, ShellOutput dataclass
   - _LocalShellExecutor (persistent subprocess session, sequential=True)
   - ShellToolset.local() classmethod factory
   - call_tool handles command/restart, timeout via anyio, output truncation

3. Anthropic adapter: when ToolDefinition has ShellNativeDefinition, emit
   BetaToolBash20250124Param instead of function tool. Map 'bash' response
   name back to toolset name via ToolDefinition lookup. Handle both streaming
   and non-streaming. Pass native-name lookup dict to stream constructor.

4. OpenAI adapter: when ToolDefinition has ShellNativeDefinition, emit
   shell with environment {type: "local"}. Map shell_call/shell_call_output
   back to toolset name. Handle both streaming and non-streaming.

5. Profile flag: supports_native_shell_tool on Anthropic and OpenAI profiles

6. Fallback: warnings.warn() once per native tool kind on unsupported providers

7. Tests (VCR cassettes via `doppler run --`), docs (native-tools.md)

Key architectural point: Phase 1 ShellTool is a builtin (remote, server-executed).
Phase 2 ShellToolset is a toolset (local, client-executed). OpenAI uses the same
shell tool type for both — differentiated by environment type (container_auto vs local).

We use jj not git. Use doppler for API keys when recording cassettes.
