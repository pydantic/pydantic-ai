from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr


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
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='foobar', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='example_tool', args={}, tool_call_id='pyd_ai_tool_call_id__example_tool')
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='example_tool',
                        content='MyDeps(foo=1, bar=2)',
                        tool_call_id='pyd_ai_tool_call_id__example_tool',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"example_tool":"MyDeps(foo=1, bar=2)"}')],
                usage=RequestUsage(input_tokens=53, output_tokens=7),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


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
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='foobar', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='example_tool', args={}, tool_call_id='pyd_ai_tool_call_id__example_tool')
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='example_tool',
                        content='MyDeps(foo=1, bar=2)',
                        tool_call_id='pyd_ai_tool_call_id__example_tool',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"example_tool":"MyDeps(foo=1, bar=2)"}')],
                usage=RequestUsage(input_tokens=53, output_tokens=7),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
