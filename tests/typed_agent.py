"""This file is used to test static typing, it's analyzed with pyright and mypy."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Union, assert_type

from pydantic_ai import Agent, CallContext
from pydantic_ai.result import RunResult


@dataclass
class MyDeps:
    foo: int
    bar: int


typed_agent1 = Agent(deps_type=MyDeps, result_type=str)
assert_type(typed_agent1, Agent[MyDeps, str])


@contextmanager
def expect_error(error_type: type[Exception]) -> Iterator[None]:
    try:
        yield None
    except Exception as e:
        assert isinstance(e, error_type), f'Expected {error_type}, got {type(e)}'
    else:
        raise AssertionError('Expected an error')


@typed_agent1.retriever_context
async def ok_retriever(ctx: CallContext[MyDeps], x: str) -> str:
    assert_type(ctx.deps, MyDeps)
    total = ctx.deps.foo + ctx.deps.bar
    return f'{x} {total}'


@typed_agent1.retriever_plain
def ok_retriever_plain(x: str) -> dict[str, str]:
    return {'x': x}


@typed_agent1.retriever_context
async def bad_retriever1(ctx: CallContext[MyDeps], x: str) -> str:
    total = ctx.deps.foo + ctx.deps.spam  # type: ignore[attr-defined]
    return f'{x} {total}'


@typed_agent1.retriever_context  # type: ignore[arg-type]
async def bad_retriever2(ctx: CallContext[int], x: str) -> str:
    return f'{x} {ctx.deps}'


@typed_agent1.retriever_plain  # type: ignore[arg-type]
async def bad_retriever_return(x: int) -> list[int]:
    return [x]


with expect_error(ValueError):

    @typed_agent1.retriever_context  # type: ignore[arg-type]
    async def bad_retriever3(x: str) -> str:
        return x


def run_sync() -> None:
    result = typed_agent1.run_sync('testing')
    assert_type(result, RunResult[str])
    assert_type(result.data, str)


async def run_stream() -> None:
    async with typed_agent1.run_stream('testing') as streamed_result:
        result_items = [chunk async for chunk in streamed_result.stream()]
        assert_type(result_items, list[str])


@dataclass
class Foo:
    a: int


@dataclass
class Bar:
    b: str


union_agent: Agent[None, Union[Foo, Bar]] = Agent(
    result_type=Union[Foo, Bar],  # type: ignore[arg-type]
)
assert_type(union_agent, Agent[None, Union[Foo, Bar]])


def run_sync3() -> None:
    result = union_agent.run_sync('testing')
    assert_type(result, RunResult[Union[Foo, Bar]])
    assert_type(result.data, Union[Foo, Bar])
