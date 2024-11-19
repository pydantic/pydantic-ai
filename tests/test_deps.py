from dataclasses import dataclass

from pydantic_ai import Agent, CallContext
from pydantic_ai.models.test import TestModel


@dataclass
class MyDeps:
    foo: int
    bar: int


agent = Agent(TestModel(), deps_type=MyDeps)


@agent.retriever
async def example_retriever(ctx: CallContext[MyDeps]) -> str:
    return f'{ctx.deps}'


def test_deps_used():
    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.data == '{"example_retriever":"MyDeps(foo=1, bar=2)"}'


def test_deps_override():
    with agent.override_deps(MyDeps(foo=3, bar=4)):
        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.data == '{"example_retriever":"MyDeps(foo=3, bar=4)"}'

        with agent.override_deps(MyDeps(foo=5, bar=6)):
            result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
            assert result.data == '{"example_retriever":"MyDeps(foo=5, bar=6)"}'

        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.data == '{"example_retriever":"MyDeps(foo=3, bar=4)"}'

    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.data == '{"example_retriever":"MyDeps(foo=1, bar=2)"}'
