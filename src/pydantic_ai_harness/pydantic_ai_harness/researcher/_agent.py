"""Runnable agent instance for the `Researcher` harness."""

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.researcher._capability import Researcher

# The working directory holds spilled tool output, under `.pydantic-ai-harness/` (git-ignored).
researcher_agent = Agent[object](name='researcher', capabilities=[LocalWorkspace[object]('.'), Researcher[object]()])
"""Model-less research agent for CLIs that load `module:variable` targets."""
