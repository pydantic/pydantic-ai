"""
This file is used to test static typing, it's analyzed with pyright and mypy.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, CallContext


@dataclass
class MyContext:
    foo: int
    bar: int


typed_agent = Agent(response_type=str, deps=MyContext(foo=1, bar=2))


@typed_agent.retriever_context
async def ok_retriever(ctx: CallContext[MyContext], x: str) -> str:
    total = ctx.deps.foo + ctx.deps.bar
    return f'{x} {total}'


@typed_agent.retriever_context
async def bad_retriever1(ctx: CallContext[MyContext], x: str) -> str:
    total = ctx.deps.foo + ctx.deps.spam  # type: ignore[attr-defined]
    return f'{x} {total}'


@typed_agent.retriever_context  # type: ignore[arg-type]
async def bad_retriever2(ctx: CallContext[int], x: str) -> str:
    return f'{x} {ctx.deps}'


@typed_agent.retriever_context  # type: ignore[arg-type]
async def bad_retriever3(x: str) -> str:
    return x


typed_agent.run_sync('testing')
