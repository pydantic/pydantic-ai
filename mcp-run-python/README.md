# MCP Run Python

[Model Context Protocol](https://modelcontextprotocol.io/) server to run Python
code in a sandbox.

The code is executed using [pyodide](https://pyodide.org) in
[deno](https://deno.com/) and is therefore isolated from the rest of the
operating system.

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
