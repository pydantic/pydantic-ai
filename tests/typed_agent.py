"""This file is used to test static typing, it's analyzed with pyright and mypy."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Union

from pydantic_ai import Agent, CallContext


@dataclass
class MyDeps:
    foo: int
    bar: int


typed_agent1 = Agent(result_type=str, deps=MyDeps(foo=1, bar=2))


@contextmanager
def expect_error(error_type: type[Exception]) -> Iterator[None]:
    try:
        yield None
    except Exception as e:
        assert isinstance(e, error_type), f'Expected {error_type}, got {type(e)}'
    else:
        raise AssertionError('Expected an error')


def never() -> bool:
    return False


@typed_agent1.retriever_context
async def ok_retriever(ctx: CallContext[MyDeps], x: str) -> str:
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


if never():
    typed_agent1.run_sync('testing')


typed_agent2: Agent[MyDeps, str] = Agent()


@typed_agent2.retriever_context
async def ok_retriever2(ctx: CallContext[MyDeps], x: str) -> str:
    total = ctx.deps.foo + ctx.deps.bar
    return f'{x} {total}'


if never():
    typed_agent2.run_sync('testing', model='openai:gpt-4o', deps=MyDeps(foo=1, bar=2))
    typed_agent2.run_sync('testing', deps=123)  # type: ignore[arg-type]


@dataclass
class Foo:
    a: int


@dataclass
class Bar:
    b: str


union_agent: Agent[None, Union[Foo, Bar]] = Agent(
    result_type=Union[Foo, Bar],  # type: ignore[arg-type]
)


def foo_result(response: Union[Foo, Bar]) -> str:
    return str(response)


if never():
    result = union_agent.run_sync('testing')
    foo_result(result.response)
