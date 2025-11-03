from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class DepsA:
    a: int


@dataclass
class DepsB:
    b: str


@dataclass
class AgentDeps(DepsA, DepsB):
    pass


agent = Agent(
    instructions='...',
    model='...',
    deps_type=AgentDeps,
)


@agent.tool
def tool_1(ctx: RunContext[DepsA]) -> int:
    return ctx.deps.a


@agent.tool
def tool_2(ctx: RunContext[DepsB]) -> str:
    return ctx.deps.b


# Ensure that you can use tools with deps that are supertypes of the agent's deps
agent.run_sync('...', deps=AgentDeps(a=0, b='test'))
