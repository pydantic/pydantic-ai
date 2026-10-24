"""Runnable agent instance for the `Researcher` harness."""

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai_harness.researcher._capability import Researcher
from pydantic_ai_harness.tool_output_limits import LocalFileStore

try:
    # The working directory holds spilled tool output, under `.pydantic-ai-harness/` (git-ignored).
    _capabilities: list[AbstractCapability[object]] = [LocalWorkspace[object]('.'), Researcher[object]()]
except NotImplementedError:
    # `LocalWorkspace` needs POSIX. Research is web-only, so elsewhere spills go to a host temp directory.
    _capabilities = [Researcher[object](store=LocalFileStore())]

researcher_agent = Agent[object](name='researcher', capabilities=_capabilities)
"""Model-less research agent for CLIs that load `module:variable` targets."""
