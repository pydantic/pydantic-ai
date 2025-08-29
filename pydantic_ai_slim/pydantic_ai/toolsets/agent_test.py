from pydantic import BaseModel

from pydantic_ai.agent import Agent


class FakeClient(BaseModel):
    """A fake client for the inner agent."""

    pass


class InnerAgentDeps(BaseModel):
    """The dependencies for the inner agent."""

    client: FakeClient
    extra: str


class OuterAgentDeps(BaseModel):
    """The dependencies for the outer agent."""

    client: FakeClient


inner_agent: Agent[InnerAgentDeps, str] = Agent[InnerAgentDeps, str]('inner', deps_type=InnerAgentDeps)

other_outer_agent: Agent[OuterAgentDeps, str] = Agent[OuterAgentDeps, str]('other_outer', deps_type=OuterAgentDeps)

outer_agent: Agent[OuterAgentDeps, str] = Agent[OuterAgentDeps, str]('outer', deps_type=OuterAgentDeps)


async def outer_to_inner_deps_func(deps: OuterAgentDeps, input: str) -> InnerAgentDeps:
    """Transform outer deps to inner deps."""
    return InnerAgentDeps(client=deps.client, extra=input)


outer_agent.handoff(agent=inner_agent, deps_func=outer_to_inner_deps_func)
outer_agent.handoff(agent=other_outer_agent)
