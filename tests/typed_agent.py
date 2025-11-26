"""This file is used to test static typing, it's analyzed with pyright and mypy."""
# pyright: reportUnnecessaryTypeIgnoreComment=false

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeAlias

from typing_extensions import assert_type

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.output import StructuredDict, TextOutput, ToolOutput
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition

# Define here so we can check `if MYPY` below. This will not be executed, MYPY will always set it to True
MYPY = False


@dataclass
class MyDeps:
    foo: int
    bar: int


typed_agent = Agent(deps_type=MyDeps, output_type=str)
assert_type(typed_agent, Agent[MyDeps, str])


@typed_agent.system_prompt
async def system_prompt_ok1(ctx: RunContext[MyDeps]) -> str:
    return f'{ctx.deps}'


@typed_agent.system_prompt
def system_prompt_ok2() -> str:
    return 'foobar'


# we have overloads for every possible signature of system_prompt, so the type of decorated functions is correct
assert_type(system_prompt_ok1, Callable[[RunContext[MyDeps]], Awaitable[str]])
assert_type(system_prompt_ok2, Callable[[], str])


@typed_agent.tool
async def ok_tool(ctx: RunContext[MyDeps], x: str) -> str:
    assert_type(ctx.deps, MyDeps)
    total = ctx.deps.foo + ctx.deps.bar
    return f'{x} {total}'


# we can't add overloads for every possible signature of tool, so the type of ok_tool is obscured
assert_type(ok_tool, Callable[[RunContext[MyDeps], str], str])  # type: ignore[assert-type]


async def prep_ok(ctx: RunContext[MyDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    if ctx.deps.foo == 42:
        return None
    else:
        return tool_def


@typed_agent.tool(prepare=prep_ok)
def ok_tool_prepare(ctx: RunContext[MyDeps], x: int, y: str) -> str:
    return f'{ctx.deps.foo} {x} {y}'


async def prep_wrong_type(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition | None:
    if ctx.deps == 42:
        return None
    else:
        return tool_def


@typed_agent.tool(prepare=prep_wrong_type)  # type: ignore[arg-type]
def wrong_tool_prepare(ctx: RunContext[MyDeps], x: int, y: str) -> str:
    return f'{ctx.deps.foo} {x} {y}'


@typed_agent.tool_plain
def ok_tool_plain(x: str) -> dict[str, str]:
    return {'x': x}


@typed_agent.tool_plain
async def ok_json_list(x: str) -> list[str | int]:
    return [x, 1]


@typed_agent.tool
async def ok_ctx(ctx: RunContext[MyDeps], x: str) -> list[int | str]:
    return [ctx.deps.foo, ctx.deps.bar, x]


@typed_agent.tool
async def bad_tool1(ctx: RunContext[MyDeps], x: str) -> str:
    total = ctx.deps.foo + ctx.deps.spam  # type: ignore[attr-defined]
    return f'{x} {total}'


@typed_agent.tool  # type: ignore[arg-type]
async def bad_tool2(ctx: RunContext[int], x: str) -> str:
    return f'{x} {ctx.deps}'


@typed_agent.output_validator
def ok_validator_simple(data: str) -> str:
    return data


@typed_agent.output_validator
async def ok_validator_ctx(ctx: RunContext[MyDeps], data: str) -> str:
    if ctx.deps.foo == 1:
        raise ModelRetry('foo is 1')
    return data


# we have overloads for every possible signature of output_validator, so the type of decorated functions is correct
assert_type(ok_validator_simple, Callable[[str], str])
assert_type(ok_validator_ctx, Callable[[RunContext[MyDeps], str], Awaitable[str]])


@typed_agent.output_validator  # type: ignore[arg-type]
async def output_validator_wrong(ctx: RunContext[int], result: str) -> str:
    return result


def run_sync() -> None:
    result = typed_agent.run_sync('testing', deps=MyDeps(foo=1, bar=2))
    assert_type(result, AgentRunResult[str])
    assert_type(result.output, str)


async def run_stream() -> None:
    async with typed_agent.run_stream('testing', deps=MyDeps(foo=1, bar=2)) as streamed_result:
        result_items = [chunk async for chunk in streamed_result.stream_output()]
        assert_type(result_items, list[str])


def run_with_override() -> None:
    with typed_agent.override(deps=MyDeps(1, 2)):
        typed_agent.run_sync('testing', deps=MyDeps(3, 4))

    # invalid deps
    with typed_agent.override(deps=123):  # type: ignore[arg-type]
        typed_agent.run_sync('testing', deps=MyDeps(3, 4))


@dataclass
class Foo:
    a: int


@dataclass
class Bar:
    b: str


union_agent: Agent[None, Foo | Bar] = Agent(output_type=Foo | Bar)  # type: ignore[call-overload]
assert_type(union_agent, Agent[None, Foo | Bar])


def run_sync3() -> None:
    result = union_agent.run_sync('testing')
    assert_type(result, AgentRunResult[Foo | Bar])
    assert_type(result.output, Foo | Bar)


MyUnion: TypeAlias = 'Foo | Bar'
union_agent2: Agent[None, MyUnion] = Agent(output_type=MyUnion)  # type: ignore[call-overload]
assert_type(union_agent2, Agent[None, MyUnion])

structured_dict = StructuredDict(
    {
        'type': 'object',
        'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
        'required': ['name', 'age'],
    }
)
structured_dict_agent = Agent(output_type=structured_dict)
assert_type(structured_dict_agent, Agent[None, dict[str, Any]])


def foobar_ctx(ctx: RunContext[int], x: str, y: int) -> Decimal:
    return Decimal(x) + y


async def foobar_plain(x: int, y: int) -> int:
    return x * y


def str_to_regex(text: str) -> re.Pattern[str]:
    return re.compile(text)


class MyClass:
    def my_method(self) -> bool:
        return True


decimal_function_agent = Agent(output_type=foobar_ctx)
assert_type(decimal_function_agent, Agent[None, Decimal])

bool_method_agent = Agent(output_type=MyClass().my_method)
assert_type(bool_method_agent, Agent[None, bool])

if MYPY:
    # mypy requires the generic parameters to be specified explicitly to be happy here
    async_int_function_agent = Agent[None, int](output_type=foobar_plain)
    assert_type(async_int_function_agent, Agent[None, int])

    two_models_output_agent = Agent[None, Foo | Bar](output_type=[Foo, Bar])
    assert_type(two_models_output_agent, Agent[None, Foo | Bar])

    two_scalars_output_agent = Agent[None, int | str](output_type=[int, str])
    assert_type(two_scalars_output_agent, Agent[None, int | str])

    marker: ToolOutput[bool | tuple[str, int]] = ToolOutput(bool | tuple[str, int])  # type: ignore
    complex_output_agent = Agent[None, Foo | Bar | Decimal | int | bool | tuple[str, int] | str | re.Pattern[str]](
        output_type=[str, Foo, Bar, foobar_ctx, ToolOutput[int](foobar_plain), marker, TextOutput(str_to_regex)]
    )
    assert_type(
        complex_output_agent, Agent[None, Foo | Bar | Decimal | int | bool | tuple[str, int] | str | re.Pattern[str]]
    )

    complex_deferred_output_agent = Agent[
        None, Foo | Bar | Decimal | int | bool | tuple[str, int] | str | re.Pattern[str] | DeferredToolRequests
    ](output_type=[complex_output_agent.output_type, DeferredToolRequests])
    assert_type(
        complex_deferred_output_agent,
        Agent[None, Foo | Bar | Decimal | int | bool | tuple[str, int] | str | re.Pattern[str] | DeferredToolRequests],
    )
else:
    # pyright is able to correctly infer the type here
    async_int_function_agent = Agent(output_type=foobar_plain)
    assert_type(async_int_function_agent, Agent[None, int])

    two_models_output_agent = Agent(output_type=[Foo, Bar])
    assert_type(two_models_output_agent, Agent[None, Foo | Bar])

    two_scalars_output_agent = Agent(output_type=[int, str])
    assert_type(two_scalars_output_agent, Agent[None, int | str])

    marker: ToolOutput[bool | tuple[str, int]] = ToolOutput(bool | tuple[str, int])  # type: ignore
    complex_output_agent = Agent(
        output_type=[str, Foo, Bar, foobar_ctx, ToolOutput(foobar_plain), marker, TextOutput(str_to_regex)]
    )
    assert_type(
        complex_output_agent, Agent[None, Foo | Bar | Decimal | int | bool | tuple[str, int] | str | re.Pattern[str]]
    )

    complex_deferred_output_agent = Agent(output_type=[complex_output_agent.output_type, DeferredToolRequests])
    assert_type(
        complex_deferred_output_agent,
        Agent[None, Foo | Bar | Decimal | int | bool | tuple[str, int] | str | re.Pattern[str] | DeferredToolRequests],
    )


Tool(foobar_ctx, takes_ctx=True)
Tool(foobar_ctx)
Tool(foobar_plain, takes_ctx=False)
assert_type(Tool(foobar_plain), Tool[object])
assert_type(Tool(foobar_plain), Tool)

# unfortunately we can't type check these cases, since from a typing perspect `foobar_ctx` is valid as a plain tool
Tool(foobar_ctx, takes_ctx=False)
Tool(foobar_plain, takes_ctx=True)

Agent('test', tools=[foobar_ctx], deps_type=int)
Agent('test', tools=[foobar_plain], deps_type=int)
Agent('test', tools=[foobar_plain])
Agent('test', tools=[Tool(foobar_ctx)], deps_type=int)
Agent('test', tools=[Tool(foobar_ctx), foobar_ctx, foobar_plain], deps_type=int)
Agent('test', tools=[Tool(foobar_ctx), foobar_ctx, Tool(foobar_plain)], deps_type=int)

Agent('test', tools=[foobar_ctx], deps_type=str)  # pyright: ignore[reportArgumentType,reportCallIssue]
Agent('test', tools=[Tool(foobar_ctx), Tool(foobar_plain)], deps_type=str)  # pyright: ignore[reportArgumentType,reportCallIssue]
Agent('test', tools=[foobar_ctx])  # pyright: ignore[reportArgumentType,reportCallIssue]
Agent('test', tools=[Tool(foobar_ctx)])  # pyright: ignore[reportArgumentType,reportCallIssue]
# since deps are not set, they default to `None`, so can't be `int`
Agent('test', tools=[Tool(foobar_plain)], deps_type=int)  # pyright: ignore[reportArgumentType,reportCallIssue]

# prepare example from docs:


def greet(name: str) -> str:
    return f'hello {name}'


async def prepare_greet(ctx: RunContext[str], tool_def: ToolDefinition) -> ToolDefinition | None:
    d = f'Name of the {ctx.deps} to greet.'
    tool_def.parameters_json_schema['properties']['name']['description'] = d
    return tool_def


greet_tool = Tool(greet, prepare=prepare_greet)
assert_type(greet_tool, Tool[str])
greet_agent = Agent[str, str]('test', tools=[greet_tool], deps_type=str)

result = greet_agent.run_sync('testing...', deps='human')
assert result.output == '{"greet":"hello a"}'

if not MYPY:
    default_agent = Agent()
    assert_type(default_agent, Agent[None, str])
    assert_type(default_agent, Agent[None])
    assert_type(default_agent, Agent)

partial_agent: Agent[MyDeps] = Agent(deps_type=MyDeps)
assert_type(partial_agent, Agent[MyDeps, str])
assert_type(partial_agent, Agent[MyDeps])
