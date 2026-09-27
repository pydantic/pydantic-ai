"""Runnable agent instance for the `Coder` harness."""

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder._capability import Coder

coder_agent = Agent[object](
    name='coder',
    capabilities=[
        LocalWorkspace[object]('.'),
        Coder[object](),
    ],
)
"""Model-less coding agent for CLIs that load `module:variable` targets."""
