from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Union, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from ..conftest import IsNow, raise_if_exception, try_import

with try_import() as imports_successful:
    import cohere
    from cohere import (
        AssistantMessageResponse,
        AsyncClientV2,
        ChatResponse,
        TextAssistantMessageResponseContentItem,
        ToolCallV2,
        ToolCallV2Function,
    )
    from cohere.core.api_error import ApiError

    from pydantic_ai.models.cohere import CohereModel

    # note: we use Union here for compatibility with Python 3.9
    MockChatResponse = Union[ChatResponse, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='cohere not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = CohereModel('command-r7b-12-2024', api_key='foobar')
    assert m.model_name == 'command-r7b-12-2024'
    assert m.system == 'cohere'
    assert m.base_url == 'https://api.cohere.com'


@dataclass
class MockAsyncClientV2:
    completions: MockChatResponse | Sequence[MockChatResponse] | None = None
    index = 0

    @classmethod
    def create_mock(cls, completions: MockChatResponse | Sequence[MockChatResponse]) -> AsyncClientV2:
        return cast(AsyncClientV2, cls(completions=completions))

    async def chat(  # pragma: no cover
        self, *_args: Any, **_kwargs: Any
    ) -> ChatResponse:
        assert self.completions is not None
        if isinstance(self.completions, Sequence):
            raise_if_exception(self.completions[self.index])
            response = cast(ChatResponse, self.completions[self.index])
        else:
            raise_if_exception(self.completions)
            response = cast(ChatResponse, self.completions)
        self.index += 1
        return response


def completion_message(message: AssistantMessageResponse, *, usage: cohere.Usage | None = None) -> ChatResponse:
    return ChatResponse(
        id='123',
        finish_reason='COMPLETE',
        message=message,
        usage=usage,
    )


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(
        AssistantMessageResponse(
            content=[
                TextAssistantMessageResponseContentItem(text='world'),
            ],
        )
    )
    mock_client = MockAsyncClientV2.create_mock(c)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')], model_name='command-r7b-12-2024', timestamp=IsNow(tz=timezone.utc)
            ),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')], model_name='command-r7b-12-2024', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        AssistantMessageResponse(
            content=[TextAssistantMessageResponseContentItem(text='world')],
            role='assistant',
        ),
        usage=cohere.Usage(
            tokens=cohere.UsageTokens(input_tokens=1, output_tokens=1),
            billed_units=cohere.UsageBilledUnits(input_tokens=1, output_tokens=1),
        ),
    )
    mock_client = MockAsyncClientV2.create_mock(c)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=1,
            response_tokens=1,
            total_tokens=2,
            details={
                'input_tokens': 1,
                'output_tokens': 1,
            },
        )
    )


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        AssistantMessageResponse(
            content=None,
            role='assistant',
            tool_calls=[
                ToolCallV2(
                    id='123',
                    function=ToolCallV2Function(arguments='{"response": [1, 2, 123]}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockAsyncClientV2.create_mock(c)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m, result_type=list[int])

    result = await agent.run('Hello')
    assert result.data == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response": [1, 2, 123]}',
                        tool_call_id='123',
                    )
                ],
                model_name='command-r7b-12-2024',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            AssistantMessageResponse(
                content=None,
                role='assistant',
                tool_calls=[
                    ToolCallV2(
                        id='1',
                        function=ToolCallV2Function(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=cohere.Usage(),
        ),
        completion_message(
            AssistantMessageResponse(
                content=None,
                role='assistant',
                tool_calls=[
                    ToolCallV2(
                        id='2',
                        function=ToolCallV2Function(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=cohere.Usage(
                tokens=cohere.UsageTokens(input_tokens=5, output_tokens=3),
                billed_units=cohere.UsageBilledUnits(input_tokens=4, output_tokens=2),
            ),
        ),
        completion_message(
            AssistantMessageResponse(
                content=[TextAssistantMessageResponseContentItem(text='final response')],
                role='assistant',
            )
        ),
    ]
    mock_client = MockAsyncClientV2.create_mock(responses)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.data == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                model_name='command-r7b-12-2024',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                model_name='command-r7b-12-2024',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name='command-r7b-12-2024',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert result.usage() == snapshot(
        Usage(
            requests=3,
            request_tokens=5,
            response_tokens=3,
            total_tokens=8,
            details={'input_tokens': 4, 'output_tokens': 2},
        )
    )


async def test_multimodal(allow_model_requests: None):
    c = completion_message(AssistantMessageResponse(content=[TextAssistantMessageResponseContentItem(text='world')]))
    mock_client = MockAsyncClientV2.create_mock(c)
    m = CohereModel('command-r7b-12-2024', cohere_client=mock_client)
    agent = Agent(m)

    with pytest.raises(RuntimeError, match='Cohere does not yet support multi-modal inputs.'):
        await agent.run(
            [
                'hello',
                ImageUrl(
                    url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                ),
            ]
        )


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockAsyncClientV2.create_mock(
        ApiError(
            status_code=500,
            body={'error': 'test error'},
        )
    )
    m = CohereModel('command-r', cohere_client=mock_client)
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot("status_code: 500, model_name: command-r, body: {'error': 'test error'}")
