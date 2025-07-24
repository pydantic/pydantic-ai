# Client

Pydantic AI can act as an [MCP client](https://modelcontextprotocol.io/quickstart/client), connecting to MCP servers
to use their tools.

## Install

You need to either install [`pydantic-ai`](../install.md), or[`pydantic-ai-slim`](../install.md#slim-install) with the `mcp` optional group:

```bash
pip/uv-add "pydantic-ai-slim[mcp]"
```

!!! note
    MCP integration requires Python 3.10 or higher.

## Usage

Pydantic AI comes with two ways to connect to MCP servers:

- [`MCPServerStreamableHTTP`][pydantic_ai.mcp.MCPServerStreamableHTTP] which connects to an MCP server using the [Streamable HTTP](https://modelcontextprotocol.io/introduction#streamable-http) transport
- [`MCPServerSSE`][pydantic_ai.mcp.MCPServerSSE] which connects to an MCP server using the [HTTP SSE](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse) transport
- [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio] which runs the server as a subprocess and connects to it using the [stdio](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) transport

Examples of all three are shown below; [mcp-run-python](run-python.md) is used as the MCP server in all examples.

Each MCP server instance is a [toolset](../toolsets.md) and can be registered with an [`Agent`][pydantic_ai.Agent] using the `toolsets` argument.

You can use the [`async with agent`][pydantic_ai.Agent.__aenter__] context manager to open and close connections to all registered servers (and in the case of stdio servers, start and stop the subprocesses) around the context where they'll be used in agent runs. You can also use [`async with server`][pydantic_ai.mcp.MCPServer.__aenter__] to manage the connection or subprocess of a specific server, for example if you'd like to use it with multiple agents. If you don't explicitly enter one of these context managers to set up the server, this will be done automatically when it's needed (e.g. to list the available tools or call a specific tool), but it's more efficient to do so around the entire context where you expect the servers to be used.

### Streamable HTTP Client

[`MCPServerStreamableHTTP`][pydantic_ai.mcp.MCPServerStreamableHTTP] connects over HTTP using the
[Streamable HTTP](https://modelcontextprotocol.io/introduction#streamable-http) transport to a server.

!!! note
    [`MCPServerStreamableHTTP`][pydantic_ai.mcp.MCPServerStreamableHTTP] requires an MCP server to be running and accepting HTTP connections before running the agent. Running the server is not managed by Pydantic AI.

Before creating the Streamable HTTP client, we need to run a server that supports the Streamable HTTP transport.

```python {title="streamable_http_server.py" py="3.10" dunder_name="not_main"}
from mcp.server.fastmcp import FastMCP

app = FastMCP()

@app.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    app.run(transport='streamable-http')
```

Then we can create the client:

```python {title="mcp_streamable_http_client.py" py="3.10"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')  # (1)!
agent = Agent('openai:gpt-4o', toolsets=[server])  # (2)!

async def main():
    async with agent:  # (3)!
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.
```

1. Define the MCP server with the URL used to connect.
2. Create an agent with the MCP server attached.
3. Create a client session to connect to the server.

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)_

**What's happening here?**

- The model is receiving the prompt "how many days between 2000-01-01 and 2025-03-18?"
- The model decides "Oh, I've got this `run_python_code` tool, that will be a good way to answer this question", and writes some python code to calculate the answer.
- The model returns a tool call
- Pydantic AI sends the tool call to the MCP server using the SSE transport
- The model is called again with the return value of running the code
- The model returns the final answer

You can visualise this clearly, and even see the code that's run by adding three lines of code to instrument the example with [logfire](https://logfire.pydantic.dev/docs):

```python {title="mcp_sse_client_logfire.py" test="skip"}
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

Will display as follows:

![Logfire run python code](../img/logfire-run-python-code.png)

### SSE Client

[`MCPServerSSE`][pydantic_ai.mcp.MCPServerSSE] connects over HTTP using the [HTTP + Server Sent Events transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse) to a server.

!!! note
    [`MCPServerSSE`][pydantic_ai.mcp.MCPServerSSE] requires an MCP server to be running and accepting HTTP connections before running the agent. Running the server is not managed by Pydantic AI.

The name "HTTP" is used since this implementation will be adapted in future to use the new
[Streamable HTTP](https://github.com/modelcontextprotocol/specification/pull/206) currently in development.

Before creating the SSE client, we need to run the server (docs [here](run-python.md)):

```bash {title="terminal (run sse server)"}
deno run \
  -N -R=node_modules -W=node_modules --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python sse
```

```python {title="mcp_sse_client.py" py="3.10"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE(url='http://localhost:3001/sse')  # (1)!
agent = Agent('openai:gpt-4o', toolsets=[server])  # (2)!


async def main():
    async with agent:  # (3)!
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.
```

1. Define the MCP server with the URL used to connect.
2. Create an agent with the MCP server attached.
3. Create a client session to connect to the server.

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)_

### MCP "stdio" Server

The other transport offered by MCP is the [stdio transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) where the server is run as a subprocess and communicates with the client over `stdin` and `stdout`. In this case, you'd use the [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio] class.

```python {title="mcp_stdio_client.py" py="3.10"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(  # (1)!
    'deno',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ]
)
agent = Agent('openai:gpt-4o', toolsets=[server])


async def main():
    async with agent:
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.
```

1. See [MCP Run Python](run-python.md) for more information.

## Tool call customisation

The MCP servers provide the ability to set a `process_tool_call` which allows
the customisation of tool call requests and their responses.

A common use case for this is to inject metadata to the requests which the server
call needs.

```python {title="mcp_process_tool_call.py" py="3.10"}
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.mcp import CallToolFunc, MCPServerStdio, ToolResult
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext


async def process_tool_call(
    ctx: RunContext[int],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """A tool call processor that passes along the deps."""
    return await call_tool(name, tool_args, {'deps': ctx.deps})


server = MCPServerStdio('python', ['mcp_server.py'], process_tool_call=process_tool_call)
agent = Agent(
    model=TestModel(call_tools=['echo_deps']),
    deps_type=int,
    toolsets=[server]
)


async def main():
    async with agent:
        result = await agent.run('Echo with deps set to 42', deps=42)
    print(result.output)
    #> {"echo_deps":{"echo":"This is an echo message","deps":42}}
```

## Using Tool Prefixes to Avoid Naming Conflicts

When connecting to multiple MCP servers that might provide tools with the same name, you can use the `tool_prefix` parameter to avoid naming conflicts. This parameter adds a prefix to all tool names from a specific server.

This allows you to use multiple servers that might have overlapping tool names without conflicts:

```python {title="mcp_tool_prefix_http_client.py" py="3.10"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Create two servers with different prefixes
weather_server = MCPServerSSE(
    url='http://localhost:3001/sse',
    tool_prefix='weather'  # Tools will be prefixed with 'weather_'
)

calculator_server = MCPServerSSE(
    url='http://localhost:3002/sse',
    tool_prefix='calc'  # Tools will be prefixed with 'calc_'
)

# Both servers might have a tool named 'get_data', but they'll be exposed as:
# - 'weather_get_data'
# - 'calc_get_data'
agent = Agent('openai:gpt-4o', toolsets=[weather_server, calculator_server])
```

## Custom TLS / SSL configuration

In some environments you need to tweak how HTTPS connections are established –
for example to trust an internal Certificate Authority, present a client
certificate for **mTLS**, or (during local development only!) disable
certificate verification altogether.
All HTTP-based MCP client classes
([`MCPServerStreamableHTTP`][pydantic_ai.mcp.MCPServerStreamableHTTP] and
[`MCPServerSSE`][pydantic_ai.mcp.MCPServerSSE]) expose an `http_client`
parameter that lets you pass your own pre-configured
[`httpx.AsyncClient`](https://www.python-httpx.org/async/).

```python {title="mcp_custom_tls_client.py" py="3.10"}
import httpx
import ssl

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE


# Trust an internal / self-signed CA
ssl_ctx = ssl.create_default_context(cafile="/etc/ssl/private/my_company_ca.pem")

# OPTIONAL: if the server requires **mutual TLS** load your client certificate
ssl_ctx.load_cert_chain(certfile="/etc/ssl/certs/client.crt", keyfile="/etc/ssl/private/client.key",)

http_client = httpx.AsyncClient(
    verify=ssl_ctx,
    timeout=httpx.Timeout(10.0),
)

server = MCPServerSSE(
    url="http://localhost:3001/sse",
    http_client=http_client,  # (1)!
)
agent = Agent("openai:gpt-4o", toolsets=[server])

async def main():
    async with agent:
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.
```

1. When you supply `http_client`, Pydantic AI re-uses this client for every
   request.  Anything supported by **httpx** (`verify`, `cert`, custom
   proxies, timeouts, etc.) therefore applies to all MCP traffic.

## MCP Sampling

!!! info "What is MCP Sampling?"
    In MCP [sampling](https://modelcontextprotocol.io/docs/concepts/sampling) is a system by which an MCP server can make LLM calls via the MCP client - effectively proxying requests to an LLM via the client over whatever transport is being used.

    Sampling is extremely useful when MCP servers need to use Gen AI but you don't want to provision them each with their own LLM credentials or when a public MCP server would like the connecting client to pay for LLM calls.

    Confusingly it has nothing to do with the concept of "sampling" in observability, or frankly the concept of "sampling" in any other domain.

    ??? info "Sampling Diagram"
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

Sampling is automatically supported by Pydantic AI agents when they act as a client.

To be able to use sampling, an MCP server instance needs to have a [`sampling_model`][pydantic_ai.mcp.MCPServerStdio.sampling_model] set. This can be done either directly on the server using the constructor keyword argument or the property, or by using [`agent.set_mcp_sampling_model()`][pydantic_ai.Agent.set_mcp_sampling_model] to set the agent's model or one specified as an argument as the sampling model on all MCP servers registered with that agent.

Let's say we have an MCP server that wants to use sampling (in this case to generate an SVG as per the tool arguments).

??? example "Sampling MCP Server"

    ```python {title="generate_svg.py" py="3.10"}
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
            path.write_text(m.group(1))
        else:
            path.write_text(result.content.text)
        return f'See {path}'


    if __name__ == '__main__':
        # run the server via stdio
        app.run()
    ```

Using this server with an `Agent` will automatically allow sampling:

```python {title="sampling_mcp_client.py" py="3.10" requires="generate_svg.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(command='python', args=['generate_svg.py'])
agent = Agent('openai:gpt-4o', toolsets=[server])


async def main():
    async with agent:
        agent.set_mcp_sampling_model()
        result = await agent.run('Create an image of a robot in a punk style.')
    print(result.output)
    #> Image file written to robot_punk.svg.
```

_(This example is complete, it can be run "as is" with Python 3.10+)_

You can disallow sampling by setting [`allow_sampling=False`][pydantic_ai.mcp.MCPServerStdio.allow_sampling] when creating the server reference, e.g.:

```python {title="sampling_disallowed.py" hl_lines="6" py="3.10"}
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    command='python',
    args=['generate_svg.py'],
    allow_sampling=False,
)
```
