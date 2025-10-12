"""Simple example showing DynamicToolset with TemporalAgent.

NOTE: This example currently FAILS with the error:
    "Toolsets that are 'leaves' need to have a unique `id` in order to be used with Temporal"

Why it fails:
- The @agent.toolset decorator automatically wraps the function in a DynamicToolset
- DynamicToolset.id returns None (by design - it's not known until the function runs)
- At TemporalAgent.__init__ time, DynamicToolset._toolset is None (the function hasn't been called yet)
- When visit_and_replace is called, DynamicToolset behaves as a "leaf" because _toolset is None
- The visitor function requires all leaf toolsets to have an ID, but DynamicToolset.id returns None

Solution (not yet implemented):
- Need to implement TemporalDynamicToolset wrapper (similar to TemporalMCPServer)
- It will provide a fixed ID and static activities (get_tools, call_tool)
- Inside the activities, it will call the user's dynamic function where I/O is allowed
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset


class MCPConnection(BaseModel):
    """Configuration for an MCP server connection."""
    command: str  # e.g., "uvx", "npx", "python", etc.
    args: list[str]  # e.g., ["mcp-server-time"]
    id: str  # unique identifier for the server


class AgentDeps(BaseModel):
    """Dependencies for the agent - must be serializable for Temporal."""
    mcp_connections: list[MCPConnection] = Field(default_factory=lambda: [])
    user_id: str = "default"


# Create agent with dynamic toolset
agent = Agent("openai:gpt-4.1", deps_type=AgentDeps)

# Register dynamic toolset via decorator
@agent.toolset
def dynamic_tools(ctx: RunContext[AgentDeps]):
    """Returns toolset based on MCP connections - can do I/O!"""
    if not ctx.deps.mcp_connections:
        return None

    # Create MCP server for each connection config
    toolsets: list[AbstractToolset[AgentDeps]] = []
    for conn in ctx.deps.mcp_connections:
        # This creates MCP connection - I/O operation allowed in Temporal activity
        mcp_server = MCPServerStdio(conn.command, conn.args, id=conn.id)
        toolsets.append(mcp_server)

    # Combine all MCP servers into one toolset
    if len(toolsets) == 1:
        return toolsets[0]
    return CombinedToolset(toolsets)

# Wrap for Temporal - automatically handles DynamicToolset!
temporal_agent = TemporalAgent(agent, name="agent")


async def main():
    """Main function demonstrating usage in a Temporal workflow."""
    deps = AgentDeps(
        mcp_connections=[
            MCPConnection(command="uvx", args=["mcp-server-time"], id="time-server"),
            MCPConnection(command="npx", args=["@modelcontextprotocol/server-fetch"], id="fetch-server"),
        ],
        user_id="alice",
    )
    result = await temporal_agent.run("What time is it?", deps=deps)
    print(result.output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
