"""Runnable agent instance for the `Coder` harness."""

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai_harness.coder._capability import Coder

try:
    _workspace: list[AbstractCapability[object]] = [LocalWorkspace[object]('.')]
except NotImplementedError:
    # `LocalWorkspace` needs POSIX. Elsewhere the agent still imports, and a run without
    # `workspace=` fails at its start naming what to attach.
    _workspace = []

coder_agent = Agent[object](name='coder', capabilities=[*_workspace, Coder[object]()])
"""Model-less coding agent for CLIs that load `module:variable` targets."""
