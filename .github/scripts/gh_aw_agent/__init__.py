"""Runtime policy for the repository's Pydantic AI agentic workflows.

The public `pydantic-ai` engine definition loads
`gh_aw_agent:agent` from this package through `PAI_AGENT`. `build_agent()`
composes the harness `Coder` with the repository's workflow policies: safe
output completion, request and wall-clock budgets, CI Review deadlines, history
compaction and telemetry.

The engine owns the model, the AWF sandbox, gateway MCP tools and session
recording. This package only decides how the agent behaves inside that run.
"""

from ._agent import agent, build_agent

__all__ = ['agent', 'build_agent']
