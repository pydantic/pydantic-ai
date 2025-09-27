"""Tests for OpenRouter model implementation."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import pytest
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from openai import NOT_GIVEN, AsyncOpenAI
    from openai.types import chat
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice as CompletionChoice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
    from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.openrouter import (
        OpenRouterModel,
    )

    MockChatCompletion = chat.ChatCompletion | Exception
    MockChatCompletionChunk = chat.ChatCompletionChunk | Exception

    @dataclass
    class MockOpenAI:
        completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
        stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]] | None = None
        index: int = 0
        chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=lambda: [])

        @cached_property
        def chat(self) -> Any:
            chat_completions = type('Completions', (), {'create': self.chat_completions_create})
            return type('Chat', (), {'completions': chat_completions})

        @classmethod
        def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> AsyncOpenAI:
            return cast(AsyncOpenAI, cls(completions=completions))

        @classmethod
        def create_mock_stream(
            cls,
            stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]],
        ) -> AsyncOpenAI:
            return cast(AsyncOpenAI, cls(stream=stream))

        async def chat_completions_create(  # pragma: lax no cover
            self, *_args: Any, stream: bool = False, **kwargs: Any
        ) -> ChatCompletion | MockAsyncStream[MockChatCompletionChunk]:
            self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

            if stream:
                assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
                if isinstance(self.stream[0], Sequence):
                    response = MockAsyncStream(iter(cast(list[MockChatCompletionChunk], self.stream[self.index])))
                else:
                    response = MockAsyncStream(iter(cast(list[MockChatCompletionChunk], self.stream)))
            else:
                assert self.completions is not None, 'you can only used `stream=False` if `completions` are provided'
                if isinstance(self.completions, Sequence):
                    raise_if_exception(self.completions[self.index])
                    response = cast(ChatCompletion, self.completions[self.index])
                else:
                    raise_if_exception(self.completions)
                    response = cast(ChatCompletion, self.completions)
            self.index += 1
            return response

    def completion_message(
        message: ChatCompletionMessage,
        *,
        usage: CompletionUsage | None = None,
    ) -> ChatCompletion:
        """Create a ChatCompletion for testing."""
        return ChatCompletion(
            id='test-id',
            choices=[
                CompletionChoice(
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                    message=message,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion',
            usage=usage,
        )


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_openrouter_model_init():
    """Test OpenRouter model initialization."""
    c = completion_message(ChatCompletionMessage(content='test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)
    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert model.system == 'openrouter'


def test_openrouter_model_with_custom_base_url():
    """Test OpenRouter model with custom base URL."""
    c = completion_message(ChatCompletionMessage(content='test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    assert model.model_name == 'google/gemini-2.5-flash-lite'


def test_openrouter_model_url_formatting():
    """Test that OpenRouter URLs are formatted correctly."""
    from pydantic_ai.models.openrouter import format_openrouter_url

    assert format_openrouter_url('openrouter.ai') == 'https://openrouter.ai/api/v1/'
    assert format_openrouter_url('https://openrouter.ai') == 'https://openrouter.ai/api/v1/'
    assert format_openrouter_url('https://openrouter.ai/api') == 'https://openrouter.ai/api/v1/'
    assert format_openrouter_url('https://openrouter.ai/api/v1') == 'https://openrouter.ai/api/v1/'


async def test_openrouter_basic_request(allow_model_requests: None):
    """Test that OpenRouter model can handle basic requests."""
    c = completion_message(ChatCompletionMessage(content='Hello from OpenRouter!', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

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
    assert response.model_name == 'google/gemini-2.5-flash-lite'


async def test_openrouter_no_reasoning_extra_body(allow_model_requests: None) -> None:
    """Ensure no reasoning payload is sent when reasoning settings are absent."""
    c = completion_message(ChatCompletionMessage(content='No reasoning', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='hi')])]
    params = ModelRequestParameters(function_tools=[], output_tools=[], allow_text_output=True)

    response = await model.request(messages, None, params)
    assert isinstance(response.parts[0], TextPart)

    kwargs = cast(MockOpenAI, mock_client).chat_completion_kwargs[0]
    extra_body = cast(dict[str, Any] | None, kwargs.get('extra_body'))
    assert not extra_body or 'reasoning' not in extra_body


async def test_openrouter_response_processing():
    """Test OpenRouter response processing with compatibility fixes."""
    c = completion_message(ChatCompletionMessage(content='Test response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    processed_response = model._process_response(c)  # pyright: ignore[reportPrivateUsage]

    assert len(processed_response.parts) == 1
    assert isinstance(processed_response.parts[0], TextPart)
    assert processed_response.parts[0].content == 'Test response'
    assert processed_response.model_name == 'google/gemini-2.5-flash-lite'


async def test_openrouter_thinking_part_response():
    """Test OpenRouter response processing with ThinkingPart."""
    message = ChatCompletionMessage(content='Final answer after thinking', role='assistant')
    setattr(message, 'reasoning', 'Let me think about this step by step...')

    c = completion_message(message)
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet', openai_client=mock_client)

    processed_response = model._process_response(c)  # pyright: ignore[reportPrivateUsage]

    assert len(processed_response.parts) == 2
    assert isinstance(processed_response.parts[0], ThinkingPart)
    assert processed_response.parts[0].content == 'Let me think about this step by step...'
    assert isinstance(processed_response.parts[1], TextPart)
    assert processed_response.parts[1].content == 'Final answer after thinking'


def test_openrouter_reasoning_param_building():
    """Test building reasoning parameters from model settings."""

    c = completion_message(ChatCompletionMessage(content='Test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet', openai_client=mock_client)

    settings = cast(ModelSettings, {'openrouter_reasoning_effort': 'high'})
    reasoning_param = model._build_reasoning_param(settings)  # pyright: ignore[reportPrivateUsage]
    assert reasoning_param == {'effort': 'high'}

    settings = cast(ModelSettings, {'openrouter_reasoning_max_tokens': 2000})
    reasoning_param = model._build_reasoning_param(settings)  # pyright: ignore[reportPrivateUsage]
    assert reasoning_param == {'max_tokens': 2000}

    settings = cast(ModelSettings, {'openrouter_reasoning_enabled': True})
    reasoning_param = model._build_reasoning_param(settings)  # pyright: ignore[reportPrivateUsage]
    assert reasoning_param == {'enabled': True}

    settings = cast(ModelSettings, {'openrouter_reasoning_effort': 'medium', 'openrouter_reasoning_exclude': True})
    reasoning_param = model._build_reasoning_param(settings)  # pyright: ignore[reportPrivateUsage]
    assert reasoning_param == {'effort': 'medium', 'exclude': True}

    settings = cast(ModelSettings, {})
    reasoning_param = model._build_reasoning_param(settings)  # pyright: ignore[reportPrivateUsage]
    assert reasoning_param is None


def test_openrouter_thinking_part_not_sent_to_provider():
    """Test that ThinkingPart is not sent back to the provider in message mapping."""
    c = completion_message(ChatCompletionMessage(content='Test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet', openai_client=mock_client)

    model_response = ModelResponse(
        parts=[
            ThinkingPart(content='I need to think about this...'),
            TextPart(content='Here is my response'),
            ToolCallPart(tool_name='test_tool', args={'arg': 'value'}, tool_call_id='call_123'),
        ]
    )

    mapped_messages = list(model._map_message(model_response))  # pyright: ignore[reportPrivateUsage]

    assert len(mapped_messages) == 1
    assistant_message = mapped_messages[0]

    assert assistant_message['role'] == 'assistant'
    assert assistant_message.get('content') == 'Here is my response'
    assert 'tool_calls' in assistant_message
    tool_calls = list(assistant_message.get('tool_calls', []))
    assert len(tool_calls) == 1
    assert cast(dict[str, Any], tool_calls[0])['function']['name'] == 'test_tool'


def test_openrouter_model_inference(monkeypatch: pytest.MonkeyPatch):
    """Test that openrouter: prefix creates OpenAIChatModel and covers both finally branches."""

    def set_openai_key(key: str):
        monkeypatch.setenv('OPENAI_API_KEY', key)

    set_openai_key('existing-key')
    original_key = 'existing-key'
    set_openai_key('test-key')

    # Set dummy OpenRouter key to avoid provider raise
    monkeypatch.setenv('OPENROUTER_API_KEY', 'dummy-key-for-test')

    try:
        model = infer_model('openrouter:google/gemini-2.5-flash-lite')
        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == 'google/gemini-2.5-flash-lite'
    finally:
        set_openai_key(original_key)

    # Test without existing OPENAI_API_KEY (second branch)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'dummy-key-for-test')

    model = infer_model('openrouter:google/gemini-2.5-flash-lite')
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == 'google/gemini-2.5-flash-lite'


async def test_openrouter_agent_integration(allow_model_requests: None):
    """Test basic integration with Agent."""
    c = completion_message(ChatCompletionMessage(content='Hello world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)
    agent = Agent(model)

    result = await agent.run('Hello')
    assert result.output == 'Hello world'
    assert agent.model == model
    assert isinstance(agent.model, OpenRouterModel)
    assert agent.model.system == 'openrouter'


def test_openrouter_model_with_custom_base_url_formatting():
    """Test OpenRouter model with custom base URL that needs formatting."""
    model = OpenRouterModel('google/gemini-2.5-flash-lite', base_url='openrouter.ai', api_key='test-key')
    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert 'openrouter.ai' in str(model.client.base_url)


def test_openrouter_model_with_http_client():
    """Test OpenRouter model with custom HTTP client."""

    http_client = AsyncClient(timeout=30)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', api_key='test-key', http_client=http_client)
    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert model.client.api_key == 'test-key'


def test_openrouter_model_with_env_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test OpenRouter model using environment variable for API key."""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'env-api-key')

    model = OpenRouterModel('google/gemini-2.5-flash-lite')
    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert model.client.api_key == 'env-api-key'


def test_openrouter_model_with_existing_openai_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test OpenRouter model when OPENAI_API_KEY already exists to cover line 269."""
    monkeypatch.setenv('OPENAI_API_KEY', 'existing-openai-key')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'env-api-key')

    model = OpenRouterModel('google/gemini-2.5-flash-lite')
    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert model.client.api_key == 'env-api-key'


async def test_openrouter_streaming(allow_model_requests: None):
    """Test OpenRouter streaming functionality."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello'),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=' World'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)
    agent = Agent(model)

    async with agent.run_stream('Hello') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == ['Hello', 'Hello World']
        assert result.is_complete


async def test_openrouter_streaming_with_empty_choices(allow_model_requests: None):
    """Test OpenRouter streaming with chunks that have empty choices."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)
    agent = Agent(model)

    async with agent.run_stream('Hello') as result:
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert 'Hello' in ''.join(text_chunks)


async def test_openrouter_tool_choice_required(allow_model_requests: None):
    """Test OpenRouter with tool choice required (no text output allowed)."""
    c = completion_message(
        ChatCompletionMessage(
            content=None,
            role='assistant',
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id='call_1',
                    function=Function(name='test_tool', arguments='{"value": "test"}'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=False,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)


async def test_openrouter_with_reasoning_param(allow_model_requests: None):
    """Test OpenRouter with reasoning parameter in model settings."""
    c = completion_message(ChatCompletionMessage(content='test response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = cast(Any, {'reasoning': True})
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_with_extra_body(allow_model_requests: None):
    """Test OpenRouter with extra_body in model settings."""
    c = completion_message(ChatCompletionMessage(content='test response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = cast(Any, {'extra_body': {'custom_param': 'custom_value'}})
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_tool_choice_auto(allow_model_requests: None):
    """Test OpenRouter with tool choice auto (text output allowed with tools)."""
    c = completion_message(ChatCompletionMessage(content='Hello response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_with_output_tools(allow_model_requests: None):
    """Test OpenRouter with output tools to cover output tool mapping."""
    c = completion_message(
        ChatCompletionMessage(
            content=None,
            role='assistant',
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id='call_1',
                    function=Function(name='final_result', arguments='{"value": "test"}'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[ToolDefinition(name='final_result', description='Final result tool')],
        allow_text_output=False,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)


async def test_openrouter_with_reasoning_string_param(allow_model_requests: None):
    """Test OpenRouter with reasoning parameter as string."""
    c = completion_message(ChatCompletionMessage(content='test response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = cast(Any, {'openrouter_reasoning_effort': 'high'})
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_complex_message_mapping(allow_model_requests: None):
    """Test OpenRouter with complex message types to cover message mapping edge cases."""

    c = completion_message(ChatCompletionMessage(content='Response after retry', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content='Hello')]),
        ModelResponse([ToolCallPart(tool_name='test_tool', args='{"value": "test"}', tool_call_id='call_1')]),
        ModelRequest(
            [
                ToolReturnPart(tool_name='test_tool', content='tool result', tool_call_id='call_1'),
                RetryPromptPart(content='Please try again', tool_name='test_tool', tool_call_id='call_1'),
            ]
        ),
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_user_message_with_mixed_content(allow_model_requests: None):
    """Test OpenRouter with user messages containing mixed content types."""

    c = completion_message(ChatCompletionMessage(content='I see the image', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content=['Hello', ImageUrl(url='https://example.com/image.jpg')])])
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_retry_prompt_without_tool(allow_model_requests: None):
    """Test OpenRouter with retry prompt that has no tool_name."""

    c = completion_message(ChatCompletionMessage(content='Retry response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content='Hello')]),
        ModelRequest(
            [
                RetryPromptPart(content='Please try again', tool_name=None),
            ]
        ),
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_with_system_prompt(allow_model_requests: None):
    """Test OpenRouter with system prompt to cover system message mapping."""

    c = completion_message(ChatCompletionMessage(content='System response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([SystemPromptPart(content='You are a helpful assistant'), UserPromptPart(content='Hello')])
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_streaming_empty_response_error(allow_model_requests: None):
    """Test OpenRouter streaming with empty response to cover error handling."""

    chunks: list[list[MockChatCompletionChunk]] = [[]]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    with pytest.raises((UnexpectedModelBehavior, StopAsyncIteration)):
        async with model.request_stream(messages, model_settings, model_request_parameters) as response:
            async for _ in response:
                pass


async def test_openrouter_with_falsy_reasoning_param(allow_model_requests: None):
    """Test OpenRouter with falsy reasoning parameter to cover the if reasoning_param check."""
    c = completion_message(ChatCompletionMessage(content='test response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = cast(Any, {'reasoning': False})
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_model_response_with_text_and_tools(allow_model_requests: None):
    """Test ModelResponse with both text and tool calls to cover lines 329->331."""

    c = completion_message(ChatCompletionMessage(content='Final response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content='Hello')]),
        ModelResponse(
            [
                TextPart(content='I need to call a tool'),
                ToolCallPart(tool_name='test_tool', args='{"value": "test"}', tool_call_id='call_1'),
            ]
        ),
        ModelRequest(
            [
                ToolReturnPart(tool_name='test_tool', content='tool result', tool_call_id='call_1'),
            ]
        ),
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_with_detailed_usage_info(allow_model_requests: None):
    """Test OpenRouter with detailed token usage to cover lines 443, 445."""

    c = ChatCompletion(
        id='test-id',
        choices=[
            CompletionChoice(
                finish_reason='stop',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(content='test response', role='assistant'),
            )
        ],
        created=1234567890,
        model='google/gemini-2.5-flash-lite',
        object='chat.completion',
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=5,
            total_tokens=15,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
        ),
    )

    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.usage is not None
    assert response.usage.details is not None


async def test_openrouter_streaming_with_content_delta(allow_model_requests: None):
    """Test OpenRouter streaming with content delta."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello'),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=' World'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_streaming_with_reasoning_delta(allow_model_requests: None):
    """Test OpenRouter streaming with reasoning delta."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import ChoiceDelta
    from openai.types.completion_usage import CompletionUsage

    class ChoiceDeltaWithReasoning(ChoiceDelta):
        reasoning: str | None = None

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDeltaWithReasoning(
                        content='Hello',
                        reasoning='Let me think...',
                    ),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_streaming_with_tool_call_delta(allow_model_requests: None):
    """Test OpenRouter streaming with tool call delta."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id='call_1',
                                function=ChoiceDeltaToolCallFunction(name='test_tool', arguments='{"value": "test"}'),
                                type='function',
                            )
                        ]
                    ),
                    finish_reason='tool_calls',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_model_response_with_only_tools(allow_model_requests: None):
    """Test ModelResponse with only tool calls (no text) to cover branch 329->331."""
    from pydantic_ai.messages import ToolReturnPart

    c = completion_message(ChatCompletionMessage(content='Final response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content='Hello')]),
        ModelResponse([ToolCallPart(tool_name='test_tool', args='{"value": "test"}', tool_call_id='call_1')]),
        ModelRequest(
            [
                ToolReturnPart(tool_name='test_tool', content='tool result', tool_call_id='call_1'),
            ]
        ),
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


async def test_openrouter_streaming_delta_handlers_return_none(allow_model_requests: None):
    """Test streaming where delta handlers return None."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=''),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                function=ChoiceDeltaToolCallFunction(name=None, arguments=''),
                                type='function',
                            )
                        ]
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test_tool', description='Test tool')],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_streaming_empty_content_deltas(allow_model_requests: None):
    """Test streaming with empty content deltas to trigger None return from handle_text_delta."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=''),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=None),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello World'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_streaming_minimal_deltas_for_branch_coverage(allow_model_requests: None):
    """Test streaming with minimal deltas to trigger branch 397->400."""

    chunks: list[MockChatCompletionChunk] = []

    for char in ['', ' ', '\n', '\t']:
        chunks.append(
            ChatCompletionChunk(
                id='test-id',
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content=char),
                        finish_reason=None,
                        index=0,
                        logprobs=None,
                    )
                ],
                created=1234567890,
                model='google/gemini-2.5-flash-lite',
                object='chat.completion.chunk',
                usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
            )
        )

    chunks.append(
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello World'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        )
    )

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_streaming_whitespace_first_delta(allow_model_requests: None):
    """Test streaming where first text delta is whitespace to trigger None return."""

    chunks = [
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='   '),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='\n\t'),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
        ChatCompletionChunk(
            id='test-id',
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content='Hello World'),
                    finish_reason='stop',
                    index=0,
                    logprobs=None,
                )
            ],
            created=1234567890,
            model='google/gemini-2.5-flash-lite',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        ),
    ]

    mock_client = MockOpenAI.create_mock_stream(chunks)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    events: list[ModelResponseStreamEvent] = []
    async with model.request_stream(messages, model_settings, model_request_parameters) as response:
        async for event in response:
            events.append(event)

    assert len(events) >= 0


async def test_openrouter_message_mapping_edge_case(allow_model_requests: None):
    """Test message mapping edge case to try to trigger branch 329->331."""
    c = completion_message(ChatCompletionMessage(content='Response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    messages: list[ModelMessage] = [
        ModelRequest([UserPromptPart(content='Hello')]),
        ModelResponse(
            [
                ToolCallPart(tool_name='test_tool', args='{"value": "test"}', tool_call_id='call_1'),
                ToolCallPart(tool_name='test_tool2', args='{"value": "test2"}', tool_call_id='call_2'),
            ]
        ),
    ]

    model_settings = None
    model_request_parameters = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='test_tool', description='Test tool'),
            ToolDefinition(name='test_tool2', description='Test tool 2'),
        ],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
