from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import assert_type

from pydantic_ai import Agent, RunContext, Tool, ToolDefinition


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
def tool_func_1(ctx: RunContext[DepsA]) -> int:
    return ctx.deps.a


@agent.tool
def tool_func_2(ctx: RunContext[DepsB]) -> str:
    return ctx.deps.b


# Ensure that you can use tools with deps that are supertypes of the agent's deps
agent.run_sync('...', deps=AgentDeps(a=0, b='test'))


def my_plain_tool() -> str:
    return 'abc'


def my_context_tool(ctx: RunContext[int]) -> str:
    return str(ctx.deps)


async def my_prepare_none(ctx: RunContext, tool_defn: ToolDefinition) -> None:
    pass


async def my_prepare_object(ctx: RunContext, tool_defn: ToolDefinition) -> None:
    pass


async def my_prepare_any(ctx: RunContext[Any], tool_defn: ToolDefinition) -> None:
    pass


tool_1 = Tool(my_plain_tool)
assert_type(tool_1, Tool)

tool_2 = Tool(my_plain_tool, prepare=my_prepare_none)
assert_type(tool_2, Tool)  # due to default parameter of RunContext being None and inferring from prepare

tool_3 = Tool(my_plain_tool, prepare=my_prepare_object)
assert_type(tool_3, Tool)

tool_4 = Tool(my_plain_tool, prepare=my_prepare_any)
assert_type(tool_4, Tool[Any])

tool_5 = Tool(my_context_tool)
assert_type(tool_5, Tool[int])

tool_6 = Tool(my_context_tool, prepare=my_prepare_object)
assert_type(tool_6, Tool[int])

# Note: The following is not ideal behavior, but the workaround is to just not use Any as the argument to your prepare
# function, as shown in the example immediately above
tool_7 = Tool(my_context_tool, prepare=my_prepare_any)
assert_type(tool_7, Tool[Any])


# A type form such as `Literal[...]` or a union binds the deps type it spells; a class binds as before
assert_type(Agent('test', deps_type=Literal['human', 'machine']), Agent[Literal['human', 'machine'], str])
assert_type(Agent('test', deps_type=DepsA | None), Agent[DepsA | None, str])
assert_type(Agent('test', deps_type=AgentDeps), Agent[AgentDeps, str])
assert_type(Agent('test', deps_type=int, output_type=int), Agent[int, int])
Agent('test', deps_type=5)  # pyright: ignore[reportArgumentType,reportCallIssue]


def literal_deps_tool(ctx: RunContext[Literal['human', 'machine']]) -> str:
    return ctx.deps


Agent('test', deps_type=Literal['human', 'machine'], tools=[literal_deps_tool])
