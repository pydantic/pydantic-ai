from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class MyDeps:
    foo: int
    bar: int


agent = Agent(TestModel(), deps_type=MyDeps)


@agent.tool
async def example_tool(ctx: RunContext[MyDeps]) -> str:
    return f'{ctx.deps}'


def test_deps_used():
    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.output == '{"example_tool":"MyDeps(foo=1, bar=2)"}'


def test_deps_override():
    with agent.override(deps=MyDeps(foo=3, bar=4)):
        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.output == '{"example_tool":"MyDeps(foo=3, bar=4)"}'

        with agent.override(deps=MyDeps(foo=5, bar=6)):
            result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
            assert result.output == '{"example_tool":"MyDeps(foo=5, bar=6)"}'

        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.output == '{"example_tool":"MyDeps(foo=3, bar=4)"}'

    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.output == '{"example_tool":"MyDeps(foo=1, bar=2)"}'
