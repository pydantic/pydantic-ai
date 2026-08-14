"""Complete research-agent harness assembled from regular capabilities."""

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability, WebFetch, WebSearch
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_RESEARCHER_INSTRUCTIONS = """\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.
"""
"""Default instructions for `Researcher`."""


def _researcher() -> SubAgent[AgentDepsT]:
    agent = Agent[AgentDepsT](  # pyright: ignore[reportCallIssue, reportArgumentType]
        name='researcher',
        description='Research a focused sub-question on the web and report back with findings and source links',
        capabilities=[
            WebSearch[AgentDepsT](local=True),
            WebFetch[AgentDepsT](local=True),
            ToolOutputLimits[AgentDepsT](),
        ],
    )
    return SubAgent(agent)


class Researcher(CombinedCapability[AgentDepsT]):
    """A complete research-agent harness built as a regular combined capability.

    It comes with concise default instructions. Pass `instructions=` to
    replace them, or `instructions=None` to run with no default instructions.
    """

    def __init__(
        self,
        *,
        instructions: str | None = DEFAULT_RESEARCHER_INSTRUCTIONS,
        subagents: Sequence[SubAgent[AgentDepsT]] | None = None,
    ) -> None:
        delegates = [_researcher()] if subagents is None else subagents
        capabilities: list[AbstractCapability[AgentDepsT]] = []
        if instructions is not None:
            capabilities.append(Capability[AgentDepsT](instructions=instructions))
        capabilities.extend(
            [
                WebSearch[AgentDepsT](local=True),
                WebFetch[AgentDepsT](local=True),
            ]
        )
        if delegates:
            capabilities.append(SubAgents[AgentDepsT](agents=delegates, agent_folders=None))
        capabilities.append(ToolOutputLimits[AgentDepsT]())
        super().__init__(capabilities)
