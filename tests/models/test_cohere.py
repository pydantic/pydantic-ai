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
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, IsNow, raise_if_exception, try_import

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
    from pydantic_ai.providers.cohere import CohereProvider

    # note: we use Union here for compatibility with Python 3.9
    MockChatResponse = Union[ChatResponse, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='cohere not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='foobar'))
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

    async def chat(self, *_args: Any, **_kwargs: Any) -> ChatResponse:
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
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(cohere_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(Usage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1),
                model_name='command-r7b-12-2024',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1),
                model_name='command-r7b-12-2024',
                timestamp=IsNow(tz=timezone.utc),
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
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(cohere_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'world'
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
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(cohere_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
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
                usage=Usage(requests=1),
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
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(cohere_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
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
                usage=Usage(requests=1, total_tokens=0, details={}),
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
                usage=Usage(
                    requests=1,
                    request_tokens=5,
                    response_tokens=3,
                    total_tokens=8,
                    details={'input_tokens': 4, 'output_tokens': 2},
                ),
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
                usage=Usage(requests=1),
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
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(cohere_client=mock_client))
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
    m = CohereModel('command-r', provider=CohereProvider(cohere_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot("status_code: 500, model_name: command-r, body: {'error': 'test error'}")


@pytest.mark.vcr()
async def test_request_simple_success_with_vcr(allow_model_requests: None, co_api_key: str):
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key=co_api_key))
    agent = Agent(m)
    result = await agent.run('hello')
    assert result.output == snapshot('Hello! How can I assist you today?')


@pytest.mark.vcr()
async def test_cohere_model_instructions(allow_model_requests: None, co_api_key: str):
    m = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key=co_api_key))

    def simple_instructions(ctx: RunContext):
        return 'You are a helpful assistant.'

    agent = Agent(m, instructions=simple_instructions)

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="The capital of France is Paris. It is the country's largest city and serves as the economic, cultural, and political center of France. Paris is known for its rich history, iconic landmarks such as the Eiffel Tower and the Louvre Museum, and its significant influence on fashion, cuisine, and the arts."
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=542,
                    response_tokens=63,
                    total_tokens=605,
                    details={'input_tokens': 13, 'output_tokens': 61},
                ),
                model_name='command-r7b-12-2024',
                timestamp=IsDatetime(),
            ),
        ]
    )
