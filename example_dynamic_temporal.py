"""Complete Temporal workflow example with DynamicToolset and MCP servers.

This example demonstrates:
- Setting up a local Temporal environment
- Defining a workflow
- Creating a worker with activities
- Executing the workflow
- Using DynamicToolset with MCP servers inside a Temporal workflow

To run this example:
1. Make sure you have Temporal dependencies installed: `pip install temporalio`
2. Run this file: `python example_dynamic_temporal_full.py`
"""

import asyncio
from datetime import timedelta

from pydantic import BaseModel, Field
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import TemporalAgent, AgentPlugin, PydanticAIPlugin
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset


# Configuration
TASK_QUEUE = "dynamic-toolset-demo"
TEMPORAL_PORT = 7233  # Default Temporal port


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
agent = Agent("openai:gpt-4o-mini", deps_type=AgentDeps)


# Register dynamic toolset via decorator with ID for Temporal
@agent.toolset(id="dynamic_mcp_tools")
def dynamic_tools(ctx: RunContext[AgentDeps]):
    """Returns toolset based on MCP connections - can do I/O!"""
    toolsets: list[AbstractToolset[AgentDeps]] = []

    # Add MCP servers
    if ctx.deps.mcp_connections:
        for conn in ctx.deps.mcp_connections:
            # This creates MCP connection - I/O operation allowed in Temporal activity
            mcp_server = MCPServerStdio(conn.command, conn.args, id=conn.id)
            toolsets.append(mcp_server)

    # Create a FunctionToolset for custom tools
    custom_tools = FunctionToolset(id="custom_tools")

    @custom_tools.tool
    def get_weather(ctx: RunContext[AgentDeps]) -> str:
        """Get the current weather information."""
        return "The weather is +35 degrees Celsius, but it's raining heavily."

    toolsets.append(custom_tools)

    # Combine all toolsets
    if len(toolsets) == 0:
        return None
    elif len(toolsets) == 1:
        return toolsets[0]
    else:
        return CombinedToolset(toolsets)


# Wrap for Temporal - this MUST be done before defining the workflow
temporal_agent = TemporalAgent(agent, name="dynamic_toolset_agent")


# Define the workflow
@workflow.defn
class DynamicToolsetWorkflow:
    """Temporal workflow that uses the agent with dynamic MCP toolset."""

    @workflow.run
    async def run(self, prompt: str, deps: AgentDeps) -> str:
        """Run the agent inside the workflow."""
        result = await temporal_agent.run(prompt, deps=deps)
        return result.output


async def main():
    """Main function to run the Temporal workflow."""
    print("Starting Temporal workflow example...")

    # For this example, we'll connect to a local Temporal server
    # In production, you'd connect to your Temporal cluster
    try:
        client = await Client.connect(
            f"localhost:{TEMPORAL_PORT}",
            plugins=[PydanticAIPlugin()],  # Required for pydantic-ai serialization
        )
        print(f"✅ Connected to Temporal server at localhost:{TEMPORAL_PORT}")
    except Exception as e:
        print(f"❌ Failed to connect to Temporal server: {e}")
        print("\nTo run this example, you need a Temporal server running.")
        print("Options:")
        print("1. Install and start Temporal CLI: https://docs.temporal.io/cli")
        print("2. Use Docker: docker run -p {TEMPORAL_PORT}:{TEMPORAL_PORT} temporalio/auto-setup:latest")
        print("3. Use temporal.io cloud")
        return

    # Create deps with MCP connections
    deps = AgentDeps(
        mcp_connections=[
            MCPConnection(command="uvx", args=["mcp-server-time"], id="time-server"),
        ],
        user_id="alice",
    )

    # Start a worker with the workflow and agent activities
    print(f"\n🔧 Starting worker on task queue: {TASK_QUEUE}")
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicToolsetWorkflow],
        plugins=[AgentPlugin(temporal_agent)],
    ):
        print("✅ Worker started")

        # Execute the workflow
        print("\n🚀 Executing workflow...")
        workflow_id = "dynamic-toolset-example-" + str(asyncio.get_event_loop().time())

        result = await client.execute_workflow(
            DynamicToolsetWorkflow.run,
            args=[
                "What time is it in Bucharest? Also, tell me the weather.",
                deps,
            ],
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )

        print("\n✅ Workflow completed!")
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
