# MCP Run Python

[Model Context Protocol](https://modelcontextprotocol.io/) server to run Python code in a sandbox.

The code is executed using [pyodide](https://pyodide.org) in [deno](https://deno.com/) and is therefore
isolated from the rest of the operating system.

The server can be run with [deno](https://deno.com/) installed using:

```bash
deno run \
  -N -R=node_modules -W=node_modules \
  --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python \
  [stdio|sse|warmup]
```

where:

- `-N -R=node_modules -W=node_modules` (alias of
  `--allow-net --allow-read=node_modules --allow-write=node_modules`) allows
  network access and read+write access to `./node_modules`. These are required
  so pyodide can download and cache the Python standard library and packages
- `--node-modules-dir=auto` tells deno to use a local `node_modules` directory
- `stdio` runs the server with the
  [Stdio MCP transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio)
  — suitable for running the process as a subprocess locally
- `sse` runs the server with the
  [SSE MCP transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse)
  — running the server as an HTTP server to connect locally or remotely
- `warmup` will run a minimal Python script to download and cache the Python
  standard library. This is also useful to check the server is running
  correctly.

To use `mcp-run-python` with the Python MCP client over `stdio`, use:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

code = """
import numpy
a = numpy.array([1, 2, 3])
print(a)
a
"""
server_params = StdioServerParameters(
    command='npx',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(tools)
            result = await session.call_tool('run_python_code', {'python_code': code})
            print(result)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

Usage of `@pydantic/mcp-run-python` with PydanticAI is described in the [client documentation](https://ai.pydantic.dev/mcp/client#mcp-stdio-server).
