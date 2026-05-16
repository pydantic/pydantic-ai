Example of connecting a Pydantic AI agent to a remote MCP server for geospatial verification.

Demonstrates:

- [MCP toolsets](../mcp/client.md)
- Using [`MCPToolset`][pydantic_ai.mcp.MCPToolset] with a public Streamable HTTP MCP server
- Agent [system prompts](../agent.md#system-prompts)

In this example, the agent connects to the [emem](https://emem.dev) MCP server — a remote server for signed geospatial facts and place-based verification. The agent auto-discovers emem's tools (locate, recall, compare, verify, etc.) and uses them to answer geospatial questions with verifiable, signed evidence.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.emem_geospatial_agent
```

No API key is required for emem (reads are anonymous). You can override the default MCP URL with the `EMEM_MCP_URL` environment variable.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/emem_geospatial_agent.py"}```

## How it works

1. The [`MCPToolset`][pydantic_ai.mcp.MCPToolset] connects to `https://emem.dev/mcp` via the Streamable HTTP transport.
2. The agent auto-discovers all available emem tools.
3. The agent receives a geospatial verification question about Helsinki Airport.
4. The model decides which emem tools to call (e.g., recall elevation, land cover, surface water).
5. The agent synthesises the results, citing signed `fact_cid`s from the receipts when available.
