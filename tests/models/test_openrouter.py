import json
import os
from typing import Any, Literal, cast
from unittest.mock import patch

import pydantic_core
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ImageUrl,
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import
from .mock_openai import (
    MockOpenAI,
    completion_message,
    get_mock_chat_completion_kwargs,
)

with try_import() as imports_successful:
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessage,
    )
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function as ChatCompletionMessageFunctionToolCall,
    )
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    def create_openrouter_model(model_name: str, mock_client: Any) -> OpenRouterModel:
        """Helper to create OpenRouterModel with mock client using provider pattern."""
        provider = OpenRouterProvider(openai_client=mock_client)
        return OpenRouterModel(model_name, provider=provider)

    def text_chunk(
        text: str,
        finish_reason: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None = None,
    ) -> ChatCompletionChunk:
        """Create a streaming chunk with text content."""
        return ChatCompletionChunk(
            id='test-chunk',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=text, role='assistant'),
                    finish_reason=finish_reason,
                    index=0,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
        )

    def chunk(choices: list[ChunkChoice]) -> ChatCompletionChunk:
        """Create a custom streaming chunk."""
        return ChatCompletionChunk(
            id='test-chunk',
            choices=choices,
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
        )


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_openrouter_model_init():
    c = completion_message(ChatCompletionMessage(content='test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(openai_client=mock_client)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', provider=provider)
    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert model.system == 'openrouter'


def test_openrouter_model_init_with_string_provider():
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-api-key'}, clear=False):
        model = OpenRouterModel('google/gemini-2.5-flash-lite', provider='openrouter')
        assert model.model_name == 'google/gemini-2.5-flash-lite'
        assert model.system == 'openrouter'
        assert model.client is not None


async def test_openrouter_basic_request(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='Hello from OpenRouter!', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)

    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'Hello from OpenRouter!'


async def test_openrouter_no_reasoning_extra_body(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='No reasoning', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='hi')])]
    params = ModelRequestParameters(function_tools=[], output_tools=[], allow_text_output=True)

    response = await model.request(messages, None, params)
    assert isinstance(response.parts[0], TextPart)

    kwargs = cast(MockOpenAI, mock_client).chat_completion_kwargs[0]
    extra_body = cast(dict[str, Any] | None, kwargs.get('extra_body'))
    assert not extra_body or 'reasoning' not in extra_body


async def test_openrouter_thinking_part_response():
    message = ChatCompletionMessage(content='Final answer after thinking', role='assistant')
    setattr(cast(Any, message), 'reasoning', 'Let me think about this step by step...')

    c = completion_message(message)
    mock_client = MockOpenAI.create_mock(c)
    model = create_openrouter_model('anthropic/claude-3.7-sonnet', mock_client)

    processed_response = model._process_response(c)  # type: ignore[reportPrivateUsage]

    assert len(processed_response.parts) == 2
    assert isinstance(processed_response.parts[0], ThinkingPart)
    assert processed_response.parts[0].content == 'Let me think about this step by step...'
    assert isinstance(processed_response.parts[1], TextPart)
    assert processed_response.parts[1].content == 'Final answer after thinking'


def test_openrouter_reasoning_param_building():
    c = completion_message(ChatCompletionMessage(content='Test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = create_openrouter_model('anthropic/claude-3.7-sonnet', mock_client)

    settings = cast(ModelSettings, {'openrouter_reasoning_effort': 'high'})
    reasoning_param = model._build_reasoning_param(settings)  # type: ignore[reportPrivateUsage]
    assert reasoning_param == {'effort': 'high'}

    settings = cast(ModelSettings, {'openrouter_reasoning_max_tokens': 2000})
    reasoning_param = model._build_reasoning_param(settings)  # type: ignore[reportPrivateUsage]
    assert reasoning_param == {'max_tokens': 2000}

    settings = cast(ModelSettings, {'openrouter_reasoning_enabled': True})
    reasoning_param = model._build_reasoning_param(settings)  # type: ignore[reportPrivateUsage]
    assert reasoning_param == {'enabled': True}

    settings = cast(ModelSettings, {'openrouter_reasoning_effort': 'medium', 'openrouter_reasoning_exclude': True})
    reasoning_param = model._build_reasoning_param(settings)  # type: ignore[reportPrivateUsage]
    assert reasoning_param == {'effort': 'medium', 'exclude': True}

    settings = cast(ModelSettings, {})
    reasoning_param = model._build_reasoning_param(settings)  # type: ignore[reportPrivateUsage]
    assert reasoning_param is None


async def test_openrouter_stream_text(allow_model_requests: None):
    """Test basic text streaming."""
    stream = [text_chunk('Hello '), text_chunk('from '), text_chunk('OpenRouter!'), chunk([])]
    mock_client = MockOpenAI.create_mock_stream(stream)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)
    agent = Agent(model)

    async with agent.run_stream('test prompt') as result:
        assert not result.is_complete
        chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert chunks == snapshot(['Hello ', 'Hello from ', 'Hello from OpenRouter!'])
        assert result.is_complete


async def test_openrouter_stream_with_finish_reason(allow_model_requests: None):
    """Test streaming with finish_reason."""
    stream = [
        text_chunk('Response '),
        text_chunk('complete', finish_reason='stop'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    model = create_openrouter_model('anthropic/claude-3.7-sonnet', mock_client)
    agent = Agent(model)

    async with agent.run_stream('test') as result:
        chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert chunks == snapshot(['Response ', 'Response complete'])
        assert result.is_complete


async def test_openrouter_tool_call(allow_model_requests: None):
    """Test single tool call."""
    responses = [
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='call_1',
                        function=ChatCompletionMessageFunctionToolCall(
                            arguments='{"location": "San Francisco"}',
                            name='get_weather',
                        ),
                        type='function',
                    )
                ],
            ),
        ),
        completion_message(ChatCompletionMessage(content='The weather is sunny!', role='assistant')),
    ]
    mock_client = MockOpenAI.create_mock(responses)
    model = create_openrouter_model('openai/gpt-4o', mock_client)
    agent = Agent(model)

    @agent.tool_plain
    async def get_weather(location: str) -> str:
        return f'Weather data for {location}'

    result = await agent.run('What is the weather?')
    assert result.output == 'The weather is sunny!'


async def test_openrouter_multiple_tool_calls(allow_model_requests: None):
    """Test multiple sequential tool calls."""
    responses = [
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='call_1',
                        function=ChatCompletionMessageFunctionToolCall(
                            arguments='{"city": "London"}',
                            name='get_location',
                        ),
                        type='function',
                    )
                ],
            ),
        ),
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='call_2',
                        function=ChatCompletionMessageFunctionToolCall(
                            arguments='{"city": "Paris"}',
                            name='get_location',
                        ),
                        type='function',
                    )
                ],
            ),
        ),
        completion_message(ChatCompletionMessage(content='Both locations found!', role='assistant')),
    ]
    mock_client = MockOpenAI.create_mock(responses)
    model = create_openrouter_model('anthropic/claude-3.5-sonnet', mock_client)
    agent = Agent(model)

    @agent.tool_plain
    async def get_location(city: str) -> str:
        return json.dumps({'city': city, 'lat': 0, 'lng': 0})

    result = await agent.run('Get locations')
    assert result.output == 'Both locations found!'


async def test_openrouter_stream_tool_call(allow_model_requests: None):
    """Test streaming with tool calls."""
    stream = [
        chunk(
            [
                ChunkChoice(
                    delta=ChoiceDelta(
                        role='assistant',
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id='call_1',
                                function=ChoiceDeltaToolCallFunction(name='calculator', arguments=''),
                                type='function',
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ]
        ),
        chunk(
            [
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='{"num": 5}'))
                        ]
                    ),
                    finish_reason=None,
                    index=0,
                )
            ]
        ),
        chunk([ChunkChoice(delta=ChoiceDelta(), finish_reason='tool_calls', index=0)]),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Calculate 5')])]
    tool_def = ToolDefinition(
        name='calculator',
        description='Do math',
        parameters_json_schema={'type': 'object', 'properties': {'num': {'type': 'number'}}},
        outer_typed_dict_key=None,
    )
    params = ModelRequestParameters(
        function_tools=[tool_def],
        output_tools=[],
        allow_text_output=True,
    )

    async with model.request_stream(messages, None, params) as response:
        events = [event async for event in response]
        assert len(events) > 0


async def test_openrouter_structured_response(allow_model_requests: None):
    """Test structured/native output."""
    response_content = '{"name": "John", "age": 30}'
    mock_response = completion_message(ChatCompletionMessage(content=response_content, role='assistant'))
    mock_client = MockOpenAI.create_mock(mock_response)
    model = create_openrouter_model('openai/gpt-4o', mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Get user info')])]
    params = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, None, params)
    assert isinstance(response.parts[0], TextPart)
    assert 'John' in response.parts[0].content


async def test_openrouter_usage_tracking(allow_model_requests: None):
    """Test usage metrics are tracked correctly."""
    mock_response = completion_message(
        ChatCompletionMessage(content='Response', role='assistant'),
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        ),
    )
    mock_client = MockOpenAI.create_mock(mock_response)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)
    agent = Agent(model)

    result = await agent.run('test')
    usage = result.usage()
    assert usage.input_tokens == 20
    assert usage.output_tokens == 10
    assert usage.total_tokens == 30


async def test_openrouter_with_reasoning_settings(allow_model_requests: None):
    """Test OpenRouter-specific reasoning settings."""
    mock_response = completion_message(ChatCompletionMessage(content='Answer', role='assistant'))
    mock_client = MockOpenAI.create_mock(mock_response)
    model = create_openrouter_model('openai/o1-preview', mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Think about this')])]
    settings = cast(ModelSettings, {'openrouter_reasoning_effort': 'high'})
    params = ModelRequestParameters(function_tools=[], output_tools=[], allow_text_output=True)

    response = await model.request(messages, settings, params)
    assert isinstance(response.parts[0], TextPart)

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    extra_body = kwargs.get('extra_body', {})
    assert 'reasoning' in extra_body
    assert extra_body['reasoning'] == {'effort': 'high'}


async def test_openrouter_model_custom_base_url(allow_model_requests: None):
    """Test OpenRouterModel with provider."""
    # Test with provider using default base URL
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4o', provider=provider)
    assert model.model_name == 'openai/gpt-4o'
    assert model.system == 'openrouter'
    assert str(model.client.base_url) == 'https://openrouter.ai/api/v1/'


async def test_openrouter_model_list_content(allow_model_requests: None):
    """Test OpenRouterModel with list content in UserPromptPart."""
    c = completion_message(ChatCompletionMessage(content='Response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)

    # Create a UserPromptPart with list content (not just a string)
    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content=['Hello', 'world', '!'])])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'Response'

    # Verify the list content was properly joined
    kwargs = cast(MockOpenAI, mock_client).chat_completion_kwargs[0]
    assert kwargs['messages'][0]['content'] == 'Hello world !'


async def test_openrouter_system_prompt_in_user_message(allow_model_requests: None):
    """Test OpenRouterModel with SystemPromptPart in user message."""
    c = completion_message(ChatCompletionMessage(content='Response with system prompt', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = create_openrouter_model('google/gemini-2.5-flash-lite', mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([SystemPromptPart(content='You are a helpful assistant.'), UserPromptPart(content='Hello')])
    ]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'Response with system prompt'

    kwargs = cast(MockOpenAI, mock_client).chat_completion_kwargs[0]
    messages_sent = kwargs['messages']
    assert len(messages_sent) == 2
    assert messages_sent[0]['role'] == 'system'
    assert messages_sent[0]['content'] == 'You are a helpful assistant.'
    assert messages_sent[1]['role'] == 'user'
    assert messages_sent[1]['content'] == 'Hello'


async def test_openrouter_retry_prompt_scenarios(allow_model_requests: None):
    """Test RetryPromptPart handling for different retry scenarios."""
    mock_client = MockOpenAI.create_mock([])
    model = create_openrouter_model('openai/gpt-4o', mock_client)

    retry_part_no_tool = RetryPromptPart(
        content='Invalid input, please try again',
        tool_name=None,
    )

    request_no_tool = ModelRequest(
        parts=[retry_part_no_tool],
    )

    messages_no_tool: list[dict[str, Any]] = []
    async for msg in model._map_user_message(request_no_tool):  # type: ignore[reportPrivateUsage]
        messages_no_tool.append(msg)  # type: ignore[reportUnknownMemberType]

    assert len(messages_no_tool) == 1
    assert isinstance(messages_no_tool[0], dict)
    assert messages_no_tool[0]['role'] == 'user'
    assert 'Invalid input, please try again' in messages_no_tool[0]['content']
    assert 'Fix the errors and try again.' in messages_no_tool[0]['content']

    retry_part_with_tool = RetryPromptPart(
        content='Tool execution failed', tool_name='get_weather', tool_call_id='call_12345'
    )

    request_with_tool = ModelRequest(
        parts=[retry_part_with_tool],
    )

    messages_with_tool: list[dict[str, Any]] = []
    async for msg in model._map_user_message(request_with_tool):  # type: ignore[reportPrivateUsage]
        messages_with_tool.append(msg)  # type: ignore[reportUnknownMemberType]

    assert len(messages_with_tool) == 1
    assert isinstance(messages_with_tool[0], dict)
    assert messages_with_tool[0]['role'] == 'tool'
    assert messages_with_tool[0]['tool_call_id'] == 'call_12345'
    assert 'Tool execution failed' in messages_with_tool[0]['content']
    assert 'Fix the errors and try again.' in messages_with_tool[0]['content']

    validation_errors = [
        pydantic_core.ErrorDetails(
            type='string_type',
            loc=('field_name',),
            msg='Input should be a valid string',
            input=123,
        )
    ]

    retry_part_validation = RetryPromptPart(
        content=validation_errors, tool_name='validate_input', tool_call_id='call_67890'
    )

    request_validation = ModelRequest(
        parts=[retry_part_validation],
    )

    messages_validation: list[dict[str, Any]] = []
    async for msg in model._map_user_message(request_validation):  # type: ignore[reportPrivateUsage]
        messages_validation.append(msg)  # type: ignore[reportUnknownMemberType]

    assert len(messages_validation) == 1
    assert isinstance(messages_validation[0], dict)
    assert messages_validation[0]['role'] == 'tool'
    assert messages_validation[0]['tool_call_id'] == 'call_67890'
    content: str = messages_validation[0]['content']
    assert '1 validation errors' in content
    assert 'Input should be a valid string' in content
    assert 'Fix the errors and try again.' in content


async def test_openrouter_user_prompt_mixed_content(allow_model_requests: None):
    """Test UserPromptPart with mixed string and non-string content."""

    mock_client = MockOpenAI.create_mock([])
    model = create_openrouter_model('openai/gpt-4o', mock_client)

    user_prompt_mixed = UserPromptPart(
        content=[
            'Hello, here is an image: ',
            ImageUrl(url='https://example.com/image.jpg'),
            ' and some more text.',
        ]
    )

    result = model._map_user_prompt(user_prompt_mixed)  # type: ignore[reportPrivateUsage]

    assert result['role'] == 'user'
    assert result['content'] == 'Hello, here is an image:   and some more text.'
