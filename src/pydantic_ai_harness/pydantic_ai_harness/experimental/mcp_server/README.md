# MCP toolset server (experimental)

Serve existing Pydantic AI toolsets as an MCP server, so MCP clients such as Claude Code can call them
without the tools being rewritten as FastMCP tools. This is the reverse of
[`MCPToolset`][pydantic_ai.mcp.MCPToolset], which consumes MCP servers as toolsets.

Prototype for [pydantic-ai#9191](https://github.com/pydantic/pydantic-ai/issues/9191); the API may change
without a deprecation period.

## Install

```bash
uv add "pydantic-ai-harness[mcp-server]"
```

## Usage

```python
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.experimental.mcp_server import run_mcp_toolset_server

toolset = FunctionToolset()


@toolset.tool
def double(ctx: RunContext[dict[str, str]], x: int) -> str:
    return ctx.deps['prefix'] + str(x * 2)


run_mcp_toolset_server(toolsets=[toolset], name='math', deps={'prefix': 'result: '})
```

Run the script and point an MCP client at its stdio transport; the client sees one tool, `double`,
with the JSON schema derived from the type hints, and calls return the tool's result.

`mcp_server_from_toolsets` builds the [`FastMCP`](https://gofastmcp.com) server without running it, for
custom transports or embedding.

## RunContext contract

Each tool call executes against one synthetic RunContext built at server construction:

| Field | Value |
| --- | --- |
| `deps` | the `deps` argument passed to the server |
| `model` | the `model` argument, or a placeholder whose metadata is benign and whose requests fail clearly |
| `usage` | a fresh `RunUsage` shared by the server |
| everything else | the default; there is no agent run, so `messages`, `retry` and friends carry no run state |

## Out of scope for the prototype

- Tools that require approval, external execution or deferred loading are skipped rather than served.
- Dynamic toolsets resolve once at construction, not per session or per call.
- Per-client session deps need the session-contract work tracked in the issue.
