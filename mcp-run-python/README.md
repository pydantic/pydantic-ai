# MCP Run Python

[Model Context Protocol](https://modelcontextprotocol.io/) server to run Python code in a sandbox.

The code is executed using [pyodide](https://pyodide.org) in node and is therefore isolated from
the rest of the operating system.

The server can be run with just npx thus:

```bash
npx @pydantic/mcp-run-python [stdio|sse]
```

where:

- `stdio` runs the server with the [Stdio MCP transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) — suitable for running the process as a subprocess locally
- and `sse` runs the server with the [SSE MCP transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse) — running the server as an HTTP server to connect locally or remotely
