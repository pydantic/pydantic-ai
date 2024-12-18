import re
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, UsageLimitExceeded
from pydantic_ai.messages import ArgsDict, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Usage
from pydantic_ai.settings import UsageLimits

from .conftest import IsNow

pytestmark = pytest.mark.anyio


def test_request_token_limit(set_event_loop: None) -> None:
    test_agent = Agent(TestModel())

    with pytest.raises(
        UsageLimitExceeded, match=re.escape('Exceeded the request_tokens_limit of 5 (request_tokens=59)')
    ):
        test_agent.run_sync(
            'Hello, this prompt exceeds the request tokens limit.', usage_limits=UsageLimits(request_tokens_limit=5)
        )


def test_response_token_limit(set_event_loop: None) -> None:
    test_agent = Agent(
        TestModel(custom_result_text='Unfortunately, this response exceeds the response tokens limit by a few!')
    )

    with pytest.raises(
        UsageLimitExceeded, match=re.escape('Exceeded the response_tokens_limit of 5 (response_tokens=11)')
    ):
        test_agent.run_sync('Hello', usage_limits=UsageLimits(response_tokens_limit=5))


def test_total_token_limit(set_event_loop: None) -> None:
    test_agent = Agent(TestModel(custom_result_text='This utilizes 4 tokens!'))

    with pytest.raises(UsageLimitExceeded, match=re.escape('Exceeded the total_tokens_limit of 50 (total_tokens=55)')):
        test_agent.run_sync('Hello', usage_limits=UsageLimits(total_tokens_limit=50))


def test_retry_limit(set_event_loop: None) -> None:
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

    async with test_agent.run_stream('Hello', usage_limits=UsageLimits(response_tokens_limit=10)) as result:
        assert test_agent.name == 'test_agent'
        assert not result.is_structured
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args=ArgsDict(args_dict={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
                ),
            ]
        )
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=5,
                total_tokens=108,
            )
        )
        with pytest.raises(
            UsageLimitExceeded, match=re.escape('Exceeded the response_tokens_limit of 10 (response_tokens=11)')
        ):
            await result.get_data()
