# Client

Pydantic AI can act as an [MCP client](https://modelcontextprotocol.io/quickstart/client), connecting to MCP servers to use their tools as part of an agent run. The [`MCPToolset`][pydantic_ai.mcp.MCPToolset] [toolset](../toolsets.md) wraps the [FastMCP Client](https://gofastmcp.com/clients/) and works with both local (stdio) and remote (Streamable HTTP, SSE) MCP servers.

!!! tip "Recommended: the `MCP` capability"
    For most use cases, use the [`MCP` capability](../capabilities.md#mcp) — it takes a URL (or any `MCPToolset` input via `local=`) and additionally lets you opt into the model provider's [native MCP support](../native-tools.md#mcp-server-tool) with a single `native=True` flag. Reach for `MCPToolset` directly when you need to manage the client lifecycle yourself, attach the same MCP server to multiple agents, or pass advanced transport / client configuration that doesn't fit the capability shape.

## Install

You need to either install [`pydantic-ai`](../install.md), or [`pydantic-ai-slim`](../install.md#slim-install) with the `mcp` optional group:

```bash
pip/uv-add "pydantic-ai-slim[mcp]"
```

## Usage

An [`MCPToolset`][pydantic_ai.mcp.MCPToolset] accepts any of the following as its first positional argument:

- A URL string (Streamable HTTP, or SSE if the path ends in `/sse`)
- A path to a local Python or Node.js script (run via stdio)
- A [FastMCP transport](https://gofastmcp.com/clients/transports) like [`StdioTransport`][fastmcp.client.transports.StdioTransport], [`StreamableHttpTransport`][fastmcp.client.transports.StreamableHttpTransport], or [`SSETransport`][fastmcp.client.transports.SSETransport]
- A pre-built [`fastmcp.Client`][fastmcp.Client] (for advanced FastMCP-specific configuration like [OAuth](https://gofastmcp.com/clients/auth/oauth) or [tool transformation](https://gofastmcp.com/patterns/tool-transformation))
- An in-process [FastMCP server](https://gofastmcp.com/servers/) (for testing or single-process deployments — no network round trip)

Each `MCPToolset` instance is a [toolset](../toolsets.md) and can be registered with an [`Agent`][pydantic_ai.Agent] via the `toolsets` argument.

You can use [`async with agent`][pydantic_ai.agent.Agent.__aenter__] to open and close connections to all registered MCP toolsets (and in the case of stdio servers, start and stop the subprocesses) around the context where they'll be used in agent runs. You can also use `async with toolset` to manage the lifecycle of a specific toolset directly, for example if you'd like to share it across multiple agents. If you don't explicitly enter one of these context managers, the toolset will be opened and closed automatically as needed.

Note that a shared `MCPToolset` instance connects to the server as a single identity; if your users have their own credentials for the MCP server, see [per-user authentication](#per-user-authentication).

### Streamable HTTP

The [Streamable HTTP](https://modelcontextprotocol.io/introduction#streamable-http) transport is the recommended way to connect to a remote MCP server.

!!! note
    A Streamable HTTP `MCPToolset` requires an MCP server to be running and accepting HTTP connections before running the agent. Running the server is not managed by Pydantic AI.

Before creating the toolset, we need to run a server that supports the Streamable HTTP transport.

```python {title="streamable_http_server.py" dunder_name="not_main"}
from mcp.server.fastmcp import FastMCP

app = FastMCP()

@app.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    app.run(transport='streamable-http')
```

Then we can create the toolset:

```python {title="mcp_streamable_http_client.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset('http://localhost:8000/mcp')  # (1)!
agent = Agent('openai:gpt-5.2', toolsets=[toolset])  # (2)!

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.
```

1. Define the MCP toolset with the URL used to connect.
2. Create an agent with the MCP toolset attached.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

**What's happening here?**

- The model receives the prompt "What is 7 plus 5?"
- The model decides "Oh, I've got this `add` tool, that will be a good way to answer this question"
- The model returns a tool call
- Pydantic AI sends the tool call to the MCP server using the Streamable HTTP transport
- The model is called again with the return value of running the `add` tool (12)
- The model returns the final answer

You can visualise this clearly, and even see the tool call, by adding three lines of code to instrument the example with [logfire](https://logfire.pydantic.dev/docs):

```python {title="mcp_streamable_http_client_logfire.py" test="skip"}
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

### SSE

The [HTTP + Server-Sent Events](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse) transport is also supported. URLs ending in `/sse` are auto-detected as SSE; for any other path, pass an explicit [`SSETransport`][fastmcp.client.transports.SSETransport].

!!! note
    The SSE transport in MCP is deprecated. You should prefer Streamable HTTP for new deployments.

```python {title="mcp_sse_client.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset('http://localhost:3001/sse')
agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

### Stdio

MCP also offers the [stdio transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio), where the server is run as a subprocess and communicates with the client over `stdin` and `stdout`. Pass a path to a Python or Node.js script, or build a [`StdioTransport`][fastmcp.client.transports.StdioTransport] for full control over the command, arguments, and environment.

```python {title="mcp_stdio_client.py" test="skip"}
from fastmcp.client.transports import StdioTransport

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(StdioTransport(command='python', args=['mcp_server.py']))
agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

### In-process FastMCP server

If you already have a [FastMCP server](https://gofastmcp.com/servers/) in the same Python process as your agent, you can hand it directly to `MCPToolset` and save the network round trip:

```python {title="mcp_in_process_server.py"}
from fastmcp import FastMCP

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

fastmcp_server = FastMCP('my_server')

@fastmcp_server.tool()
async def add(a: int, b: int) -> int:
    return a + b

toolset = MCPToolset(fastmcp_server)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Loading MCP toolsets from configuration

Instead of constructing `MCPToolset` instances individually, you can load multiple toolsets from a JSON configuration file using [`load_mcp_toolsets()`][pydantic_ai.mcp.load_mcp_toolsets].

This is particularly useful when you need to manage multiple MCP servers or want to configure servers externally without modifying code.

### Configuration format

The configuration file should be a JSON file with an `mcpServers` object containing server definitions. Each server is identified by a unique key and contains the configuration for that server:

```json {title="mcp_config.json"}
{
  "mcpServers": {
    "python-runner": {
        "command": "uv",
        "args": ["run", "mcp-run-python", "stdio"]
    },
    "weather": {
      "command": "python",
      "args": ["mcp_server.py"]
    },
    "weather-api": {
      "url": "http://localhost:3001/sse"
    },
    "calculator": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

!!! note
    The MCP server is only inferred to be an SSE server because of the `/sse` suffix. Any other server with the `url` field is treated as a Streamable HTTP server. We made this decision given that the SSE transport is deprecated.

### Environment variables

The configuration file supports environment variable expansion using the `${VAR}` and `${VAR:-default}` syntax, [like Claude Code](https://code.claude.com/docs/en/mcp#environment-variable-expansion-in-mcp-json). This is useful for keeping sensitive information like API keys or host names out of your configuration files:

```json {title="mcp_config_with_env.json"}
{
  "mcpServers": {
    "python-runner": {
      "command": "${PYTHON_CMD:-python3}",
      "args": ["run", "${MCP_MODULE}", "stdio"],
      "env": {
        "API_KEY": "${MY_API_KEY}"
      }
    },
    "weather-api": {
      "url": "https://${SERVER_HOST:-localhost}:${SERVER_PORT:-8080}/sse"
    }
  }
}
```

When loading this configuration with [`load_mcp_toolsets()`][pydantic_ai.mcp.load_mcp_toolsets]:

- `${VAR}` references are replaced with the corresponding environment variable values.
- `${VAR:-default}` references use the environment variable value if set, otherwise the default value.

!!! warning
    If a referenced environment variable using `${VAR}` syntax is not defined, a `ValueError` will be raised. Use the `${VAR:-default}` syntax to provide a fallback value.

!!! warning "Treat configuration files as trusted input"
    A configuration file specifies executables and arguments to spawn as subprocesses, so anyone who can write it can run arbitrary commands. `${VAR}` references are expanded against the full process environment without an allowlist, so a config file can also read any environment variable. Only load configuration files you control; never load them from untrusted sources.

### Usage

```python {title="mcp_config_loader.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_toolsets

# Load all toolsets from the configuration file
toolsets = load_mcp_toolsets('mcp_config.json')

# Create an agent with all loaded toolsets
agent = Agent('openai:gpt-5.2', toolsets=toolsets)

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
```

## Tool call customization

`MCPToolset` accepts a `process_tool_call` callback that lets you customize tool call requests and their responses. A common use case is to inject metadata that the server-side handler needs to read:

```python {title="mcp_process_tool_call.py" requires="mcp_server.py" test="skip"}
from typing import Any

from fastmcp.client.transports import StdioTransport

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult
from pydantic_ai.models.test import TestModel


async def process_tool_call(
    ctx: RunContext[int],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """A tool call processor that passes along the deps."""
    return await call_tool(name, tool_args, {'deps': ctx.deps})


toolset = MCPToolset(
    StdioTransport(command='python', args=['mcp_server.py']),
    process_tool_call=process_tool_call,
)
agent = Agent(
    model=TestModel(call_tools=['echo_deps']),
    deps_type=int,
    toolsets=[toolset],
)


async def main():
    result = await agent.run('Echo with deps set to 42', deps=42)
    print(result.output)
    #> {"echo_deps":{"echo":"This is an echo message","deps":42}}
```

How the server reads the injected metadata is MCP server SDK specific. For example, with the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) it's accessible via the [`ctx: Context`](https://github.com/modelcontextprotocol/python-sdk#context) argument on tool handlers:

```python {title="mcp_server.py"}
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP('Pydantic AI MCP Server')


@mcp.tool()
async def echo_deps(ctx: Context[ServerSession, None]) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """
    await ctx.info('This is an info message')

    deps: Any = getattr(ctx.request_context.meta, 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}


if __name__ == '__main__':
    mcp.run()
```

## Tool prefixes to avoid naming conflicts

When connecting to multiple MCP servers that might provide tools with the same name, wrap each `MCPToolset` with [`.prefixed(...)`][pydantic_ai.toolsets.AbstractToolset.prefixed] to prepend a prefix to its tool names:

```python {title="mcp_tool_prefix.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

weather = MCPToolset('http://localhost:3001/sse').prefixed('weather')   # `weather_*`
calculator = MCPToolset('http://localhost:3002/sse').prefixed('calc')   # `calc_*`

# Both servers may expose a `get_data` tool, but they're disambiguated as
# `weather_get_data` and `calc_get_data`.
agent = Agent('openai:gpt-5.2', toolsets=[weather, calculator])
```

## Server instructions

MCP servers can provide instructions during initialization that give context about how to best interact with the server's tools. These are accessible via [`MCPToolset.instructions`][pydantic_ai.mcp.MCPToolset.instructions] after the connection is established, and can be automatically injected into the agent's instructions by setting `include_instructions=True`:

```python {title="mcp_server_include_instructions.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset('http://localhost:8000/mcp', include_instructions=True)
agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

## Tool metadata

MCP tools can include metadata that provides additional information about the tool's characteristics, which can be useful when [filtering tools][pydantic_ai.toolsets.FilteredToolset]. The `meta` and `annotations` fields can be found on the `metadata` dict on the [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] object that's passed to filter functions, and the tool's output schema (if any) is available as the `return_schema` field.

[`MCPToolset`][pydantic_ai.mcp.MCPToolset] additionally exposes a `task: bool` flag indicating whether the server declares support for [task-augmented execution](#background-tasks) on the tool.

## Background tasks

[`MCPToolset`][pydantic_ai.mcp.MCPToolset] supports MCP [task-augmented execution](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks) (SEP-1686). Servers can declare per-tool task support via `execution.taskSupport`, and `MCPToolset` routes calls accordingly:

| `execution.taskSupport` | Behavior |
| --- | --- |
| `"required"` | Always calls with `task=True`. The server creates a task and the client awaits the final result via `tasks/result`. |
| `"optional"` | Always calls with `task=True` to opt in to durability, cancellation, and progress notifications. |
| `"forbidden"` or absent | Calls normally. |

For [FastMCP](https://gofastmcp.com/) servers, declare task support per tool with `task=TaskConfig(mode=...)`:

```python {title="background_task_server.py" dunder_name="not_main"}
from fastmcp import FastMCP
from fastmcp.server.tasks import TaskConfig

mcp = FastMCP('long_running_server')


@mcp.tool(task=TaskConfig(mode='required'))
async def deep_research(topic: str) -> str:
    import asyncio
    await asyncio.sleep(0)
    return f'Researched {topic}'


if __name__ == '__main__':
    mcp.run(transport='streamable-http')
```

The client side needs no extra configuration — `MCPToolset` sends `task=True` automatically based on the server's declaration:

```python {title="background_task_client.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset('http://localhost:8000/mcp')
agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

## Resources

MCP servers can provide [resources](https://modelcontextprotocol.io/docs/concepts/resources) — files, data, or content that can be accessed by the client. Resources in MCP are application-driven, with host applications determining how to incorporate context manually based on their needs. They are _not_ exposed to the LLM automatically (unless a tool returns a `ResourceLink` or `EmbeddedResource`).

`MCPToolset` exposes methods to discover and read resources:

- [`list_resources()`][pydantic_ai.mcp.MCPToolset.list_resources] — list all available resources on the server
- [`list_resource_templates()`][pydantic_ai.mcp.MCPToolset.list_resource_templates] — list resource templates with parameter placeholders
- [`read_resource(uri)`][pydantic_ai.mcp.MCPToolset.read_resource] — read the contents of a specific resource by URI

Text content is returned as `str`, and binary content as [`BinaryContent`][pydantic_ai.messages.BinaryContent].

Before consuming resources, we need to run a server that exposes some:

```python {title="mcp_resource_server.py"}
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Pydantic AI MCP Server')


@mcp.resource('resource://user_name.txt', mime_type='text/plain')
async def user_name_resource() -> str:
    return 'Alice'


if __name__ == '__main__':
    mcp.run()
```

Then we can read them from the client:

```python {title="mcp_resources.py" requires="mcp_resource_server.py" test="skip"}
import asyncio

from fastmcp.client.transports import StdioTransport

from pydantic_ai.mcp import MCPToolset


async def main():
    toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'mcp_resource_server']))

    async with toolset:
        # List all available resources
        resources = await toolset.list_resources()
        for resource in resources:
            print(f' - {resource.name}: {resource.uri} ({resource.mime_type})')
            #>  - user_name_resource: resource://user_name.txt (text/plain)

        # Read a text resource
        user_name = await toolset.read_resource('resource://user_name.txt')
        print(f'Text content: {user_name}')
        #> Text content: Alice


if __name__ == '__main__':
    asyncio.run(main())
```

_(This example is complete, it can be run "as is")_

## HTTP authentication

For HTTP transports, `MCPToolset` accepts an `auth` argument: a bearer token string, any [`httpx.Auth`](https://www.python-httpx.org/advanced/authentication/), or the literal string `'oauth'` to enable [FastMCP's OAuth flow](https://gofastmcp.com/clients/auth/oauth). Static headers like API keys can be passed via the `headers` argument instead.

### Per-user authentication

In a multi-user or multi-tenant application, each user typically has their own credentials for the MCP server, such as a tenant-scoped bearer token.

!!! warning "A shared `MCPToolset` instance is a single identity"
    An `MCPToolset` instance maintains one MCP session that's shared by all concurrent agent runs using it: the connection is established (and authentication resolved) by whichever run needs it first, and only torn down once the last one finishes. Deriving credentials per-request from task-local state like a [`ContextVar`][contextvars.ContextVar] inside an `httpx.Auth` does not work on a shared instance: overlapping runs will silently send their requests with the credentials of whichever run opened the session.

To make requests with the credentials of the user in question, each concurrent run needs its own `MCPToolset` instance so that it establishes its own authenticated session. The recommended way to do this is to build the toolset [dynamically](../toolsets.md#dynamically-building-a-toolset) using the [`@agent.toolset`][pydantic_ai.agent.Agent.toolset] decorator: the decorated function is passed the [run context][pydantic_ai.tools.RunContext] and can read the user's credentials from the run's [dependencies](../dependencies.md):

```python {title="mcp_per_user_auth.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPToolset


@dataclass
class UserDeps:
    mcp_token: str


agent = Agent('openai:gpt-5.2', deps_type=UserDeps)


@agent.toolset(per_run_step=False)  # (1)!
def user_mcp_server(ctx: RunContext[UserDeps]) -> MCPToolset:
    return MCPToolset('http://localhost:8000/mcp', auth=ctx.deps.mcp_token)


async def main():
    result = await agent.run('What is 7 plus 5?', deps=UserDeps(mcp_token='<token>'))
    print(result.output)
    #> The answer is 12.
```

1. `per_run_step=False` builds the toolset once per run instead of ahead of each run step, so the whole run shares a single MCP session.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Because the per-run toolset's session is established inside the run itself, credentials held in a `ContextVar` also resolve correctly with this pattern — but passing them through deps is more explicit and doesn't depend on task-local state.

As an alternative to a dynamic toolset, you can construct a new `MCPToolset` yourself for each request and pass it to the [`toolsets` argument](../toolsets.md) of the agent run methods.

## Custom TLS / SSL configuration

In some environments you need to tweak how HTTPS connections are established — for example to trust an internal Certificate Authority, present a client certificate for **mTLS**, or (during local development only!) disable certificate verification altogether. `MCPToolset` exposes an `http_client` parameter so you can pass your own pre-configured [`httpx.AsyncClient`](https://www.python-httpx.org/async/):

```python {title="mcp_custom_tls_client.py" test="skip"}
import ssl

import httpx

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# Trust an internal / self-signed CA
ssl_ctx = ssl.create_default_context(cafile='/etc/ssl/private/my_company_ca.pem')

# Optional: load a client certificate for mutual TLS
ssl_ctx.load_cert_chain(certfile='/etc/ssl/certs/client.crt', keyfile='/etc/ssl/private/client.key')

http_client = httpx.AsyncClient(verify=ssl_ctx, timeout=httpx.Timeout(10.0))

toolset = MCPToolset('http://localhost:3001/sse', http_client=http_client)  # (1)!
agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

1. When you supply `http_client`, Pydantic AI reuses this client for every request. Anything supported by **httpx** (`verify`, `cert`, custom proxies, timeouts, etc.) therefore applies to all MCP traffic.

## Client identification

When connecting to an MCP server, you can optionally specify an [Implementation](https://modelcontextprotocol.io/specification/2025-11-25/schema#implementation) object as client information that will be sent to the server during initialization. This is useful for:

- Identifying your application in server logs
- Allowing servers to provide custom behavior based on the client
- Debugging and monitoring MCP connections
- Version-specific feature negotiation

```python {title="mcp_client_with_name.py" test="skip"}
from mcp import types as mcp_types

from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(
    'http://localhost:3001/sse',
    client_info=mcp_types.Implementation(
        name='MyApplication',
        version='2.1.0',
    ),
)
```

## MCP sampling

!!! info "What is MCP sampling?"
    In MCP, [sampling](https://modelcontextprotocol.io/docs/concepts/sampling) is a system by which an MCP server can make LLM calls via the MCP client — effectively proxying requests to an LLM via the client over whatever transport is being used.

    Sampling is extremely useful when MCP servers need to use Gen AI but you don't want to provision them each with their own LLM credentials, or when a public MCP server would like the connecting client to pay for LLM calls.

    Confusingly, it has nothing to do with the concept of "sampling" in observability, or frankly the concept of "sampling" in any other domain.

    ??? info "Sampling diagram"
        Here's a mermaid diagram that may or may not make the data flow clearer:

        ```mermaid
        sequenceDiagram
            participant LLM
            participant MCP_Client as MCP client
            participant MCP_Server as MCP server

            MCP_Client->>LLM: LLM call
            LLM->>MCP_Client: LLM tool call response

            MCP_Client->>MCP_Server: tool call
            MCP_Server->>MCP_Client: sampling "create message"

            MCP_Client->>LLM: LLM call
            LLM->>MCP_Client: LLM text response

            MCP_Client->>MCP_Server: sampling response
            MCP_Server->>MCP_Client: tool call response
        ```

Pydantic AI supports sampling as both a client and server. See the [server](./server.md#mcp-sampling) documentation for details on how to use sampling within a server.

To use sampling as a client, an `MCPToolset` needs to have a [`sampling_model`][pydantic_ai.mcp.MCPToolset.sampling_model] set. This can be done either directly on the toolset using the `sampling_model=` constructor keyword argument, or by using [`agent.set_mcp_sampling_model()`][pydantic_ai.agent.Agent.set_mcp_sampling_model] to use the agent's model (or one specified as an argument) as the sampling model on all `MCPToolset`s registered with the agent.

Let's say we have an MCP server that wants to use sampling (in this case to generate an SVG as per the tool arguments):

??? example "Sampling MCP server"

    ```python {title="generate_svg.py"}
    import re
    from pathlib import Path

    from mcp import SamplingMessage
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.types import TextContent

    app = FastMCP()


    @app.tool()
    async def image_generator(ctx: Context, subject: str, style: str) -> str:
        prompt = f'{subject=} {style=}'
        # `ctx.session.create_message` is the sampling call
        result = await ctx.session.create_message(
            [SamplingMessage(role='user', content=TextContent(type='text', text=prompt))],
            max_tokens=1_024,
            system_prompt='Generate an SVG image as per the user input',
        )
        assert isinstance(result.content, TextContent)

        path = Path(f'{subject}_{style}.svg')
        # remove triple backticks if the svg was returned within markdown
        if m := re.search(r'^```\w*$(.+?)```$', result.content.text, re.S | re.M):
            path.write_text(m.group(1), encoding='utf-8')
        else:
            path.write_text(result.content.text, encoding='utf-8')
        return f'See {path}'


    if __name__ == '__main__':
        # run the server via stdio
        app.run()
    ```

Using this server with an `Agent` will automatically allow sampling:

```python {title="sampling_mcp_client.py" requires="generate_svg.py" test="skip"}
from fastmcp.client.transports import StdioTransport

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(StdioTransport(command='python', args=['generate_svg.py']))
agent = Agent('openai:gpt-5.2', toolsets=[toolset])


async def main():
    agent.set_mcp_sampling_model()
    result = await agent.run('Create an image of a robot in a punk style.')
    print(result.output)
    #> Image file written to robot_punk.svg.
```

_(This example is complete, it can be run "as is")_

## Elicitation

In MCP, [elicitation](https://modelcontextprotocol.io/docs/concepts/elicitation) allows a server to request [structured input](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation#supported-schema-types) from the client for missing or additional context during a session.

Elicitation lets models essentially say "Hold on — I need to know X before I can continue", rather than requiring everything upfront or taking a shot in the dark.

### How elicitation works

Elicitation introduces a protocol message type called [`ElicitRequest`](https://modelcontextprotocol.io/specification/2025-06-18/schema#elicitrequest), which is sent from the server to the client when it needs additional information. The client can then respond with an [`ElicitResult`](https://modelcontextprotocol.io/specification/2025-06-18/schema#elicitresult) or an `ErrorData` message.

A typical interaction looks like this:

- User makes a request to the MCP server (e.g. "Book a table at that Italian place")
- The server identifies that it needs more information (e.g. "Which Italian place?", "What date and time?")
- The server sends an `ElicitRequest` to the client asking for the missing information.
- The client receives the request, presents it to the user (e.g. via a terminal prompt, GUI dialog, or web interface).
- User provides the requested information, declines, or cancels.
- The client sends an `ElicitResult` back to the server with the user's response.
- With the structured data, the server can continue processing the original request.

This allows for a more interactive and user-friendly experience, especially for multi-stage workflows. Instead of requiring all information upfront, the server can ask for it as needed.

### Setting up elicitation

To enable elicitation, provide an `elicitation_handler` when creating your `MCPToolset`:

```python {title="restaurant_server.py"}
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP(name='Restaurant Booking')


class BookingDetails(BaseModel):
    """Schema for restaurant booking information."""

    restaurant: str = Field(description='Choose a restaurant')
    party_size: int = Field(description='Number of people', ge=1, le=8)
    date: str = Field(description='Reservation date (DD-MM-YYYY)')


@mcp.tool()
async def book_table(ctx: Context) -> str:
    """Book a restaurant table with user input."""
    # Ask user for booking details using Pydantic schema
    result = await ctx.elicit(message='Please provide your booking details:', schema=BookingDetails)

    if result.action == 'accept' and result.data:
        booking = result.data
        return f'✅ Booked table for {booking.party_size} at {booking.restaurant} on {booking.date}'
    elif result.action == 'decline':
        return 'No problem! Maybe another time.'
    else:  # cancel
        return 'Booking cancelled.'


if __name__ == '__main__':
    mcp.run(transport='stdio')
```

This server demonstrates elicitation by requesting structured booking details from the client when the `book_table` tool is called. Here's how to wire up the matching client:

```python {title="client_example.py" requires="restaurant_server.py" test="skip"}
import asyncio

from fastmcp.client.transports import StdioTransport
from mcp.types import ElicitRequestParams, ElicitResult

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


async def handle_elicitation(context, params: ElicitRequestParams) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f'\n{params.message}')

    if not params.requestedSchema:
        response = input('Response: ')
        return ElicitResult(action='accept', content={'response': response})

    # Collect data for each field
    properties = params.requestedSchema['properties']
    data = {}

    for field, info in properties.items():
        description = info.get('description', field)

        value = input(f'{description}: ')

        # Convert to proper type based on JSON schema
        if info.get('type') == 'integer':
            data[field] = int(value)
        else:
            data[field] = value

    # Confirm
    confirm = input('\nConfirm booking? (y/n/c): ').lower()

    if confirm == 'y':
        print('Booking details:', data)
        return ElicitResult(action='accept', content=data)
    elif confirm == 'n':
        return ElicitResult(action='decline')
    else:
        return ElicitResult(action='cancel')


toolset = MCPToolset(
    StdioTransport(command='python', args=['restaurant_server.py']),
    elicitation_handler=handle_elicitation,
)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])


async def main():
    """Run the agent to book a restaurant table."""
    result = await agent.run('Book me a table')
    print(f'\nResult: {result.output}')


if __name__ == '__main__':
    asyncio.run(main())
```

### Supported schema types

MCP elicitation supports string, number, boolean, and enum types with flat object structures only. These limitations ensure reliable cross-client compatibility. See [supported schema types](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation#supported-schema-types) for details.

### Security

MCP elicitation requires careful handling — servers must not request sensitive information, and clients must implement user approval controls with clear explanations. See [security considerations](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation#security-considerations) for details.
