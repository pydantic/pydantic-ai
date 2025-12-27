"""Tests for streaming response handling."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage, RunUsage

from ...conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_openai_responses_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Paris'

    output_text: list[str] = []
    async with agent.run_stream('What is the capital of France?') as result:
        async for output in result.stream_text():
            output_text.append(output)
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[
                            TextPart(
                                content='The capital of France is Paris.',
                                id='msg_67e554a28bec8191b56d3e2331eff88006c52f0e511c76ed',
                            )
                        ],
                        usage=RequestUsage(input_tokens=278, output_tokens=9, details={'reasoning_tokens': 0}),
                        model_name='gpt-4o-2024-08-06',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_url='https://api.openai.com/v1/',
                        provider_details={
                            'finish_reason': 'completed',
                            'timestamp': datetime(2025, 3, 27, 13, 37, 38, tzinfo=timezone.utc),
                        },
                        provider_response_id='resp_67e554a21aa88191b65876ac5e5bbe0406c52f0e511c76ed',
                        finish_reason='stop',
                    )
                )

    assert output_text == snapshot(['The capital of France is Paris.'])


async def test_openai_responses_model_http_error(allow_model_requests: None, openai_api_key: str):
    """Set temperature to -1 to trigger an error, given only values between 0 and 1 are allowed."""
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(temperature=-1))

    with pytest.raises(ModelHTTPError):
        async with agent.run_stream('What is the capital of France?'):
            ...  # pragma: lax no cover


async def test_openai_responses_streaming_usage(allow_model_requests: None, openai_api_key: str):
    class Result(BaseModel):
        result: int

    agent = Agent(
        model=OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key)),
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='low',
            openai_service_tier='flex',
        ),
        output_type=Result,
    )

    async with agent.iter('Calculate 100 * 200 / 3') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as response_stream:
                    async for _ in response_stream:
                        pass
                    assert response_stream.get().usage == snapshot(
                        RequestUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448})
                    )
                    assert response_stream.usage() == snapshot(
                        RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
                    )
                    assert run.usage() == snapshot(RunUsage(requests=1))
                assert run.usage() == snapshot(
                    RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
                )
    assert run.usage() == snapshot(
        RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
    )


async def test_openai_responses_non_reasoning_model_no_item_ids(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4.1', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    @agent.tool_plain
    def get_meaning_of_life() -> int:
        return 42

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the meaning of life?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_meaning_of_life',
                        args='{}',
                        tool_call_id='call_3WCunBU7lCG1HHaLmnnRJn8I',
                        id='fc_68cc4fa649ac8195b0c6c239cd2c14470548824120ffcf74',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=15, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 18, 18, 29, 57, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cc4fa5603481958e2143685133fe530548824120ffcf74',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_meaning_of_life',
                        content=42,
                        tool_call_id='call_3WCunBU7lCG1HHaLmnnRJn8I',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The meaning of life, according to popular culture and famously in Douglas Adams' "The Hitchhiker's Guide to the Galaxy," is 42!

If you're looking for a deeper or philosophical answer, let me know your perspective or context, and I can elaborate further.\
""",
                        id='msg_68cc4fa7693081a184ff6f32e5209ab00307c6d4d2ee5985',
                    )
                ],
                usage=RequestUsage(input_tokens=61, output_tokens=56, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 18, 18, 29, 58, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cc4fa6a8a881a187b0fe1603057bff0307c6d4d2ee5985',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
            {
                'name': 'get_meaning_of_life',
                'arguments': '{}',
                'call_id': 'call_3WCunBU7lCG1HHaLmnnRJn8I',
                'type': 'function_call',
            },
            {'type': 'function_call_output', 'call_id': 'call_3WCunBU7lCG1HHaLmnnRJn8I', 'output': '42'},
            {
                'role': 'assistant',
                'content': """\
The meaning of life, according to popular culture and famously in Douglas Adams' "The Hitchhiker's Guide to the Galaxy," is 42!

If you're looking for a deeper or philosophical answer, let me know your perspective or context, and I can elaborate further.\
""",
            },
        ]
    )
