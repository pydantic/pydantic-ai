"""Runnable agent instance for the `Coder` harness."""

from pydantic_ai import Agent

from pydantic_ai_harness.coder._capability import Coder

coder_agent = Agent(
    name='coder',
    instructions='You are a coding agent built on Pydantic AI.',
    capabilities=[Coder()],
)
"""Model-less coding agent for CLIs that load `module:variable` targets."""
