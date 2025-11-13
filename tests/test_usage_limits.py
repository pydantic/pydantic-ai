import asyncio
import functools
import operator
import re
from datetime import timezone
from decimal import Decimal

import pytest
from genai_prices import Usage as GenaiPricesUsage, calc_price
from inline_snapshot import snapshot
from inline_snapshot.extra import warns
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    ToolCallPart,
    ToolReturnPart,
    UsageLimitExceeded,
    UserPromptPart,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import ToolOutput
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from .conftest import IsNow, IsStr

pytestmark = pytest.mark.anyio


def test_genai_prices():
    usage = GenaiPricesUsage(input_tokens=100, output_tokens=50)
    assert calc_price(usage, model_ref='gpt-4o').total_price == snapshot(Decimal('0.00075'))


def test_request_token_limit() -> None:
    test_agent = Agent(TestModel())

    with pytest.raises(UsageLimitExceeded, match=re.escape('Exceeded the input_tokens_limit of 5 (input_tokens=59)')):
        test_agent.run_sync(
            'Hello, this prompt exceeds the request tokens limit.', usage_limits=UsageLimits(input_tokens_limit=5)
        )


def test_response_token_limit() -> None:
    test_agent = Agent(
        TestModel(custom_output_text='Unfortunately, this response exceeds the response tokens limit by a few!')
    )

    with pytest.raises(UsageLimitExceeded, match=re.escape('Exceeded the output_tokens_limit of 5 (output_tokens=11)')):
        test_agent.run_sync('Hello', usage_limits=UsageLimits(output_tokens_limit=5))


def test_total_token_limit() -> None:
    test_agent = Agent(TestModel(custom_output_text='This utilizes 4 tokens!'))

    with pytest.raises(UsageLimitExceeded, match=re.escape('Exceeded the total_tokens_limit of 50 (total_tokens=55)')):
        test_agent.run_sync('Hello', usage_limits=UsageLimits(total_tokens_limit=50))


def test_retry_limit() -> None:
    test_agent = Agent(TestModel())

    @test_agent.tool_plain
    async def foo(x: str) -> str:
        return x

    @test_agent.tool_plain
    async def bar(y: str) -> str:
        return y

    with pytest.raises(UsageLimitExceeded, match=re.escape('The next request would exceed the request_limit of 1')):
        test_agent.run_sync('Hello', usage_limits=UsageLimits(request_limit=1))


async def test_streamed_text_limits() -> None:
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    succeeded = False

    with pytest.raises(
        UsageLimitExceeded, match=re.escape('Exceeded the output_tokens_limit of 10 (output_tokens=11)')
    ):
        async with test_agent.run_stream('Hello', usage_limits=UsageLimits(output_tokens_limit=10)) as result:
            assert test_agent.name == 'test_agent'
            assert not result.is_complete
            assert result.all_messages() == snapshot(
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                        run_id=IsStr(),
                    ),
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name='ret_a',
                                args={'x': 'a'},
                                tool_call_id=IsStr(),
                            )
                        ],
                        usage=RequestUsage(input_tokens=51),
                        model_name='test',
                        timestamp=IsNow(tz=timezone.utc),
                        provider_name='test',
                        run_id=IsStr(),
                    ),
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name='ret_a',
                                content='a-apple',
                                timestamp=IsNow(tz=timezone.utc),
                                tool_call_id=IsStr(),
                            )
                        ],
                        run_id=IsStr(),
                    ),
                ]
            )
            assert result.usage() == snapshot(
                RunUsage(
                    requests=2,
                    input_tokens=103,
                    output_tokens=5,
                    tool_calls=1,
                )
            )
            succeeded = True

    assert succeeded


def test_usage_so_far() -> None:
    test_agent = Agent(TestModel())

    with pytest.raises(
        UsageLimitExceeded, match=re.escape('Exceeded the total_tokens_limit of 105 (total_tokens=163)')
    ):
        test_agent.run_sync(
            'Hello, this prompt exceeds the request tokens limit.',
            usage_limits=UsageLimits(total_tokens_limit=105),
            usage=RunUsage(input_tokens=50, output_tokens=50),
        )


async def test_multi_agent_usage_no_incr():
    delegate_agent = Agent(TestModel(), output_type=int)

    controller_agent1 = Agent(TestModel())
    run_1_usages: list[RunUsage] = []

    @controller_agent1.tool
    async def delegate_to_other_agent1(ctx: RunContext[None], sentence: str) -> int:
        delegate_result = await delegate_agent.run(sentence)
        delegate_usage = delegate_result.usage()
        run_1_usages.append(delegate_usage)
        assert delegate_usage == snapshot(RunUsage(requests=1, input_tokens=51, output_tokens=4))
        return delegate_result.output

    result1 = await controller_agent1.run('foobar')
    assert result1.output == snapshot('{"delegate_to_other_agent1":0}')
    run_1_usages.append(result1.usage())
    assert result1.usage() == snapshot(RunUsage(requests=2, input_tokens=103, output_tokens=13, tool_calls=1))

    controller_agent2 = Agent(TestModel())

    @controller_agent2.tool
    async def delegate_to_other_agent2(ctx: RunContext[None], sentence: str) -> int:
        delegate_result = await delegate_agent.run(sentence, usage=ctx.usage)
        delegate_usage = delegate_result.usage()
        assert delegate_usage == snapshot(RunUsage(requests=2, input_tokens=102, output_tokens=9))
        return delegate_result.output

    result2 = await controller_agent2.run('foobar')
    assert result2.output == snapshot('{"delegate_to_other_agent2":0}')
    assert result2.usage() == snapshot(RunUsage(requests=3, input_tokens=154, output_tokens=17, tool_calls=1))

    # confirm the usage from result2 is the sum of the usage from result1
    assert result2.usage() == functools.reduce(operator.add, run_1_usages)

    result1_usage = result1.usage()
    result1_usage.details = {'custom1': 10, 'custom2': 20, 'custom3': 0}
    assert result1_usage.opentelemetry_attributes() == {
        'gen_ai.usage.input_tokens': 103,
        'gen_ai.usage.output_tokens': 13,
        'gen_ai.usage.details.custom1': 10,
        'gen_ai.usage.details.custom2': 20,
    }


async def test_multi_agent_usage_sync():
    """As in `test_multi_agent_usage_async`, with a sync tool."""
    controller_agent = Agent(TestModel())

    @controller_agent.tool
    def delegate_to_other_agent(ctx: RunContext[None], sentence: str) -> int:
        new_usage = RunUsage(requests=5, input_tokens=2, output_tokens=3)
        ctx.usage.incr(new_usage)
        return 0

    result = await controller_agent.run('foobar')
    assert result.output == snapshot('{"delegate_to_other_agent":0}')
    assert result.usage() == snapshot(RunUsage(requests=7, input_tokens=105, output_tokens=16, tool_calls=1))


def test_request_usage_basics():
    usage = RequestUsage()
    assert usage.output_audio_tokens == 0
    assert usage.requests == 1


def test_add_usages():
    usage = RunUsage(
        requests=2,
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=30,
        cache_write_tokens=40,
        input_audio_tokens=50,
        cache_audio_read_tokens=60,
        tool_calls=3,
        details={
            'custom1': 10,
            'custom2': 20,
        },
    )
    assert usage + usage == snapshot(
        RunUsage(
            requests=4,
            input_tokens=20,
            output_tokens=40,
            cache_write_tokens=80,
            cache_read_tokens=60,
            input_audio_tokens=100,
            cache_audio_read_tokens=120,
            tool_calls=6,
            details={'custom1': 20, 'custom2': 40},
        )
    )
    assert usage + RunUsage() == usage
    assert RunUsage() + RunUsage() == RunUsage()


async def test_tool_call_limit() -> None:
    test_agent = Agent(TestModel())

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    with pytest.raises(
        UsageLimitExceeded,
        match=re.escape('The next tool call(s) would exceed the tool_calls_limit of 0 (tool_calls=1).'),
    ):
        await test_agent.run('Hello', usage_limits=UsageLimits(tool_calls_limit=0))

    result = await test_agent.run('Hello', usage_limits=UsageLimits(tool_calls_limit=1))
    assert result.usage() == snapshot(RunUsage(requests=2, input_tokens=103, output_tokens=14, tool_calls=1))


async def test_output_tool_not_counted() -> None:
    """Test that output tools are not counted in tool_calls usage metric."""
    test_agent = Agent(TestModel())

    @test_agent.tool_plain
    async def regular_tool(x: str) -> str:
        return f'{x}-processed'

    class MyOutput(BaseModel):
        result: str

    result_regular = await test_agent.run('test')
    assert result_regular.usage() == snapshot(RunUsage(requests=2, input_tokens=103, output_tokens=14, tool_calls=1))

    test_agent_with_output = Agent(TestModel(), output_type=ToolOutput(MyOutput))

    @test_agent_with_output.tool_plain
    async def another_regular_tool(x: str) -> str:
        return f'{x}-processed'

    result_output = await test_agent_with_output.run('test')

    assert result_output.usage() == snapshot(RunUsage(requests=2, input_tokens=103, output_tokens=15, tool_calls=1))


async def test_output_tool_allowed_at_limit() -> None:
    """Test that output tools can be called even when at the tool_calls_limit."""

    class MyOutput(BaseModel):
        result: str

    def call_output_after_regular(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 'test'}, 'call_1'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        else:
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'result': 'success'}, 'call_2'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

    test_agent = Agent(FunctionModel(call_output_after_regular), output_type=ToolOutput(MyOutput))

    @test_agent.tool_plain
    async def regular_tool(x: str) -> str:
        return f'{x}-processed'

    result = await test_agent.run('test', usage_limits=UsageLimits(tool_calls_limit=1))

    assert result.output.result == 'success'
    assert result.usage() == snapshot(RunUsage(requests=2, input_tokens=20, output_tokens=10, tool_calls=1))


async def test_failed_tool_calls_not_counted() -> None:
    """Test that failed tool calls (raising ModelRetry) are not counted in usage or against limits."""
    test_agent = Agent(TestModel())

    call_count = 0

    @test_agent.tool_plain
    async def flaky_tool(x: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ModelRetry('Temporary failure, please retry')
        return f'{x}-success'

    result = await test_agent.run('test', usage_limits=UsageLimits(tool_calls_limit=1))
    assert call_count == 2
    assert result.usage() == snapshot(RunUsage(requests=3, input_tokens=176, output_tokens=29, tool_calls=1))


def test_deprecated_usage_limits():
    with warns(
        snapshot(['DeprecationWarning: `request_tokens_limit` is deprecated, use `input_tokens_limit` instead'])
    ):
        assert UsageLimits(input_tokens_limit=100).request_tokens_limit == 100  # type: ignore

    with warns(
        snapshot(['DeprecationWarning: `response_tokens_limit` is deprecated, use `output_tokens_limit` instead'])
    ):
        assert UsageLimits(output_tokens_limit=100).response_tokens_limit == 100  # type: ignore


async def test_parallel_tool_calls_limit_enforced():
    """Parallel tool calls must not exceed the limit and should raise immediately."""
    executed_tools: list[str] = []

    model_call_count = 0

    def test_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_call_count
        model_call_count += 1

        if model_call_count == 1:
            # First response: 5 parallel tool calls (within limit)
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, 'call_1'),
                    ToolCallPart('tool_b', {}, 'call_2'),
                    ToolCallPart('tool_c', {}, 'call_3'),
                    ToolCallPart('tool_a', {}, 'call_4'),
                    ToolCallPart('tool_b', {}, 'call_5'),
                ]
            )
        else:
            assert model_call_count == 2
            # Second response: 3 parallel tool calls (would exceed limit of 6)
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_c', {}, 'call_6'),
                    ToolCallPart('tool_a', {}, 'call_7'),
                    ToolCallPart('tool_b', {}, 'call_8'),
                ]
            )

    test_model = FunctionModel(test_model_function)
    agent = Agent(test_model)

    @agent.tool_plain
    async def tool_a() -> str:
        await asyncio.sleep(0.01)
        executed_tools.append('a')
        return 'result a'

    @agent.tool_plain
    async def tool_b() -> str:
        await asyncio.sleep(0.01)
        executed_tools.append('b')
        return 'result b'

    @agent.tool_plain
    async def tool_c() -> str:
        await asyncio.sleep(0.01)
        executed_tools.append('c')
        return 'result c'

    # Run with tool call limit of 6; expecting an error when trying to execute 3 more tools
    with pytest.raises(
        UsageLimitExceeded,
        match=re.escape('The next tool call(s) would exceed the tool_calls_limit of 6 (tool_calls=8).'),
    ):
        await agent.run('Use tools', usage_limits=UsageLimits(tool_calls_limit=6))

    # Only the first batch of 5 tools should have executed
    assert len(executed_tools) == 5


def test_usage_unknown_provider():
    assert RequestUsage.extract({}, provider='unknown', provider_url='', provider_fallback='') == RequestUsage()
