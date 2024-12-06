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


def test_deps_used(set_event_loop: None):
    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.data == '{"example_tool":"MyDeps(foo=1, bar=2)"}'
    agent.model.agent_model = None


def test_deps_override(set_event_loop: None):
    with agent.override(deps=MyDeps(foo=3, bar=4)):
        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.data == '{"example_tool":"MyDeps(foo=3, bar=4)"}'
        agent.model.agent_model = None

        with agent.override(deps=MyDeps(foo=5, bar=6)):
            result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
            assert result.data == '{"example_tool":"MyDeps(foo=5, bar=6)"}'
            agent.model.agent_model = None

        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.data == '{"example_tool":"MyDeps(foo=3, bar=4)"}'
        agent.model.agent_model = None

    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.data == '{"example_tool":"MyDeps(foo=1, bar=2)"}'
