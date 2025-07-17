# Server

Pydantic AI models can also be used within MCP Servers.

## MCP Server

Here's a simple example of a [Python MCP server](https://github.com/modelcontextprotocol/python-sdk) using Pydantic AI within a tool call:

```py {title="mcp_server.py" py="3.10"}
from mcp.server.fastmcp import FastMCP

from pydantic_ai import Agent

server = FastMCP('Pydantic AI Server')
server_agent = Agent(
    'anthropic:claude-3-5-haiku-latest', system_prompt='always reply in rhyme'
)


@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}')
    return r.output


if __name__ == '__main__':
    server.run()
```

## Simple client

This server can be queried with any MCP client. Here is an example using the Python SDK directly:

```py {title="mcp_client.py" py="3.10" requires="mcp_server.py" dunder_name="not_main"}
import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def client():
    server_params = StdioServerParameters(
        command='python', args=['mcp_server.py'], env=os.environ
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
            """
            Oh, socks, those garments soft and sweet,
            That nestle softly 'round our feet,
            From cotton, wool, or blended thread,
            They keep our toes from feeling dread.
            """


if __name__ == '__main__':
    asyncio.run(client())
```

## MCP Sampling

!!! info "What is MCP Sampling?"
    See the [MCP client docs](./client.md#mcp-sampling) for details of what MCP sampling is, and how you can support it when using Pydantic AI as an MCP client.

When Pydantic AI agents are used within MCP servers, they can use sampling via [`MCPSamplingModel`][pydantic_ai.models.mcp_sampling.MCPSamplingModel].

We can extend the above example to use sampling so instead of connecting directly to the LLM, the agent calls back through the MCP client to make LLM calls.

```py {title="mcp_server_sampling.py" py="3.10"}
from mcp.server.fastmcp import Context, FastMCP

from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

server = FastMCP('Pydantic AI Server with sampling')
server_agent = Agent(system_prompt='always reply in rhyme')


@server.tool()
async def poet(ctx: Context, theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}', model=MCPSamplingModel(session=ctx.session))
    return r.output


if __name__ == '__main__':
    server.run()  # run the server over stdio
```

The [above](#simple-client) client does not support sampling, so if you tried to use it with this server you'd get an error.

The simplest way to support sampling in an MCP client is to [use](./client.md#mcp-sampling) a Pydantic AI agent as the client, but if you wanted to support sampling with the vanilla MCP SDK, you could do so like this:

```py {title="mcp_client_sampling.py" py="3.10" requires="mcp_server_sampling.py"}
import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult, ErrorData, TextContent


async def sampling_callback(
    context: RequestContext[ClientSession, Any], params: CreateMessageRequestParams
) -> CreateMessageResult | ErrorData:
    print('sampling system prompt:', params.systemPrompt)
    #> sampling system prompt: always reply in rhyme
    print('sampling messages:', params.messages)
    """
    sampling messages:
    [
        SamplingMessage(
            role='user',
            content=TextContent(
                type='text',
                text='write a poem about socks',
                annotations=None,
                meta=None,
            ),
        )
    ]
    """

    # TODO get the response content by calling an LLM...
    response_content = 'Socks for a fox.'

    return CreateMessageResult(
        role='assistant',
        content=TextContent(type='text', text=response_content),
        model='fictional-llm',
    )


async def client():
    server_params = StdioServerParameters(command='python', args=['mcp_server_sampling.py'])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
            #> Socks for a fox.


if __name__ == '__main__':
    asyncio.run(client())
```

_(This example is complete, it can be run "as is" with Python 3.10+)_
