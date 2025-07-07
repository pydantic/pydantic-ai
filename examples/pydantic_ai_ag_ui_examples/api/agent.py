"""Create a Pydantic AI agent and AG-UI adapter."""

from __future__ import annotations

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.ag_ui import FastAGUI
from pydantic_ai.tools import AgentDepsT


def agent(
    model: str = 'openai:gpt-4o-mini',
    deps: AgentDepsT = None,
    instructions: str | None = None,
) -> FastAGUI[AgentDepsT, str]:
    """Create a Pydantic AI agent with AG-UI adapter.

    Args:
        model: The model to use for the agent.
        deps: Optional dependencies for the agent.
        instructions: Optional instructions for the agent.

    Returns:
        An instance of FastAGUI with the agent and adapter.
    """
    # Ensure environment variables are loaded.
    load_dotenv()

    return Agent(
        model,
        output_type=str,
        instructions=instructions,
        deps_type=type(deps),
    ).to_ag_ui(deps=deps)
