"""Basic tests for OpenAI Responses model: initialization, config, validation, simple responses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, cast

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    BinaryContent,
    ImageGenerationTool,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import ImageAspectRatio
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsNow, IsStr, TestEnv, try_import
from ..mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
    )
    from pydantic_ai.models.openai._shared import (
        _resolve_openai_image_generation_size,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def test_openai_responses_model(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    model = OpenAIResponsesModel('gpt-4o')
    assert model.model_name == 'gpt-4o'
    assert model.system == 'openai'
    assert model.base_url == 'https://api.openai.com/v1/'
    assert model.client.api_key == 'test'


async def test_openai_responses_model_simple_response(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_responses_image_detail_vendor_metadata(allow_model_requests: None):
    c = response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=cast(list[Content], [ResponseOutputText(text='done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model)

    image_url = ImageUrl('https://example.com/image.png', vendor_metadata={'detail': 'high'})
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'high'})

    result = await agent.run(['Describe these inputs.', image_url, binary_image])
    assert result.output == 'done'

    response_kwargs = get_mock_responses_kwargs(mock_client)
    image_parts = [
        item
        for message in response_kwargs[0]['input']
        if message.get('role') == 'user'
        for item in message['content']
        if item['type'] == 'input_image'
    ]
    assert image_parts
    assert all(part['detail'] == 'high' for part in image_parts)


@pytest.mark.parametrize(
    ('aspect_ratio', 'explicit_size', 'expected_size'),
    [
        ('1:1', 'auto', '1024x1024'),
        ('2:3', '1024x1536', '1024x1536'),
        ('3:2', 'auto', '1536x1024'),
    ],
)
def test_openai_responses_image_generation_tool_aspect_ratio_mapping(
    aspect_ratio: ImageAspectRatio,
    explicit_size: Literal['1024x1024', '1024x1536', '1536x1024', 'auto'],
    expected_size: Literal['1024x1024', '1024x1536', '1536x1024'],
) -> None:
    tool = ImageGenerationTool(aspect_ratio=aspect_ratio, size=explicit_size)
    assert _resolve_openai_image_generation_size(tool) == expected_size


def test_openai_responses_image_generation_tool_aspect_ratio_invalid() -> None:
    from pydantic_ai import UserError

    tool = ImageGenerationTool(aspect_ratio='16:9')

    with pytest.raises(UserError, match='OpenAI image generation only supports `aspect_ratio` values'):
        _resolve_openai_image_generation_size(tool)


def test_openai_responses_image_generation_tool_aspect_ratio_conflicts_with_size() -> None:
    from pydantic_ai import UserError

    tool = ImageGenerationTool(aspect_ratio='1:1', size='1536x1024')

    with pytest.raises(UserError, match='cannot combine `aspect_ratio` with a conflicting `size`'):
        _resolve_openai_image_generation_size(tool)


def test_openai_responses_image_generation_tool_unsupported_size_raises_error() -> None:
    from pydantic_ai import UserError

    tool = ImageGenerationTool(size='2K')
    with pytest.raises(UserError, match='OpenAI image generation only supports `size` values'):
        _resolve_openai_image_generation_size(tool)


async def test_openai_responses_model_simple_response_with_tool_call(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Potato City'

    result = await agent.run('What is the capital of PotatoLand?')
    assert result.output == snapshot('The capital of PotatoLand is Potato City.')


async def test_openai_responses_output_type(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class MyOutput(TypedDict):
        name: str
        age: int

    agent = Agent(model=model, output_type=MyOutput)
    result = await agent.run('Give me the name and age of Brazil, Argentina, and Chile.')
    assert result.output == snapshot({'name': 'Brazil', 'age': 2023})


async def test_openai_responses_system_prompt(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, system_prompt='You are a helpful assistant.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_responses_model_retry(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, I only know about "London".')

    result = await agent.run('What is the location of Londos and London?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the location of Londos and London?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"Londos"}',
                        tool_call_id=IsStr(),
                        id='fc_67e547c540648191bc7505ac667e023f0ae6111e84dd5c08',
                    ),
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"London"}',
                        tool_call_id=IsStr(),
                        id='fc_67e547c55c3081919da7a3f7fe81a1030ae6111e84dd5c08',
                    ),
                ],
                usage=RequestUsage(details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 3, 27, 12, 42, 44, tzinfo=timezone.utc),
                },
                provider_response_id='resp_67e547c48c9481918c5c4394464ce0c60ae6111e84dd5c08',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, I only know about "London".',
                        tool_name='get_location',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
It seems "Londos" might be incorrect or unknown. If you meant something else, please clarify.

For **London**, it's located at approximately latitude 51° N and longitude 0° W.\
""",
                        id='msg_67e547c615ec81918d6671a184f82a1803a2086afed73b47',
                    )
                ],
                usage=RequestUsage(input_tokens=335, output_tokens=44, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 3, 27, 12, 42, 45, tzinfo=timezone.utc),
                },
                provider_response_id='resp_67e547c5a2f08191802a1f43620f348503a2086afed73b47',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
