# Native Tools

Native tools let your agent execute commands on your local machine using provider-optimized tool formats. Models trained on their provider's native tool schemas perform significantly better than with generic function-tool equivalents.

!!! warning "Security"
    Running shell commands from LLM output can be dangerous. Always sandbox execution, restrict file access, and review tool activity. Consider using [`ApprovalRequiredToolset`][pydantic_ai.toolsets.ApprovalRequiredToolset] to require human approval for sensitive commands.

## Which Shell Tool Do I Use?

| Goal | Tool | Type |
|------|------|------|
| Provider runs code in **their hosted container** | [`ShellTool`][pydantic_ai.builtin_tools.ShellTool] | [Built-in tool](builtin-tools.md) |
| Run commands on **your machine** (or your Docker) | [`ShellToolset`][pydantic_ai.toolsets.ShellToolset] | [Native toolset](toolsets.md) |

## ShellToolset

[`ShellToolset`][pydantic_ai.toolsets.ShellToolset] executes shell commands locally on your machine. When supported by the model (Anthropic, OpenAI), it uses the provider-native shell format for best performance. On unsupported providers, it falls back to a standard function tool.

### Basic Usage

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import ShellToolset

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[ShellToolset.local(cwd='/workspace')],
)
result = agent.run_sync('List all Python files')
print(result.output)
```

The [`ShellToolset.local()`][pydantic_ai.toolsets.ShellToolset.local] factory creates a toolset with a subprocess-based executor that maintains a persistent shell session — `cd`, `export`, and other state-modifying commands persist across calls.

### Configuration

```python {test="skip"}
from pydantic_ai.toolsets import ShellToolset

shell = ShellToolset.local(
    cwd='/workspace',               # Working directory
    env={'PATH': '/usr/bin'},        # Environment variables
    tool_name='bash',                # Custom tool name (default: 'shell')
    timeout=30.0,                    # Per-command timeout in seconds
    max_output_chars=100_000,        # Output truncation limit
)
```

### Provider Support

| Provider | Format | Notes |
|----------|--------|-------|
| Anthropic | `bash_20250124` | Native bash tool format |
| OpenAI Responses | `shell` with `local` env | Same shell tool type as hosted mode |
| Others | Function tool fallback | Emits a warning; model performance may be degraded |

### Safety Patterns

Wrap in [`ApprovalRequiredToolset`][pydantic_ai.toolsets.ApprovalRequiredToolset] for human-in-the-loop approval:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import ShellToolset

shell = ShellToolset.local(cwd='/workspace')
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[shell.approval_required()],
)
```

### Custom Executors

Implement the [`ShellExecutor`][pydantic_ai.toolsets.shell.ShellExecutor] protocol for custom execution backends (Docker, SSH, sandboxed environments):

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets.shell import ShellOutput, ShellToolset


class DockerExecutor:
    """Execute commands in a Docker container."""

    async def execute(self, command: str) -> ShellOutput:
        # Your Docker execution logic here
        return ShellOutput(output='...', exit_code=0)

    async def restart(self) -> ShellOutput:
        return ShellOutput(output='Container restarted.', exit_code=0)

    async def close(self) -> None:
        pass  # Clean up container


agent = Agent(
    'openai:gpt-5.4',
    toolsets=[ShellToolset(executor=DockerExecutor())],
)
```

## How Native Tools Work

When a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] has a [`native_definition`][pydantic_ai.tools.ToolDefinition.native_definition] set (e.g., [`ShellNativeDefinition`][pydantic_ai.tools.ShellNativeDefinition]), model adapters check the model profile's `supports_native_shell_tool` flag:

- **Supported**: The adapter emits the provider-native format (e.g., Anthropic `bash_20250124`, OpenAI `shell` with `local` environment).
- **Unsupported**: The adapter falls back to a standard function tool using `parameters_json_schema` and emits a one-time warning.

The tool call flows through the standard agent loop — the model returns a tool call, the agent executes it via `ShellToolset.call_tool()`, and the result is sent back to the model.
