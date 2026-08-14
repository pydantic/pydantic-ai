"""Runnable agent instance for the `Researcher` harness."""

from pydantic_ai import Agent

from pydantic_ai_harness.researcher._capability import Researcher

researcher_agent = Agent(name='researcher', capabilities=[Researcher()])
"""Model-less research agent for CLIs that load `module:variable` targets."""
