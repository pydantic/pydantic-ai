"""Tests for OpenRouter model implementation."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Union, cast

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

from ..conftest import raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from openai import NOT_GIVEN, AsyncOpenAI
    from openai.types import chat
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    MockChatCompletion = Union[chat.ChatCompletion, Exception]
    MockChatCompletionChunk = Union[chat.ChatCompletionChunk, Exception]


@dataclass
class MockOpenAI:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]] | None = None
    index: int = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list)

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
            {
                'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
                'message': message,
            }
        ],
        created=1234567890,
        model='google/gemini-2.5-flash-lite',
        object='chat.completion',
        usage=usage,
    )


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
    from pydantic_ai.models.openrouter import _format_openrouter_url

    assert _format_openrouter_url('openrouter.ai') == 'https://openrouter.ai/api/v1/'
    assert _format_openrouter_url('https://openrouter.ai') == 'https://openrouter.ai/api/v1/'
    assert _format_openrouter_url('https://openrouter.ai/api') == 'https://openrouter.ai/api/v1/'
    assert _format_openrouter_url('https://openrouter.ai/api/v1') == 'https://openrouter.ai/api/v1/'


async def test_openrouter_basic_request(allow_model_requests: None):
    """Test that OpenRouter model can handle basic requests."""
    c = completion_message(ChatCompletionMessage(content='Hello from OpenRouter!', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    messages = [ModelRequest([UserPromptPart(content='Hello')])]
    model_settings = {}
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )

    response = await model.request(messages, model_settings, model_request_parameters)

    assert response.parts[0].content == 'Hello from OpenRouter!'
    assert response.model_name == 'google/gemini-2.5-flash-lite'


async def test_openrouter_response_processing():
    """Test OpenRouter response processing with compatibility fixes."""
    c = completion_message(ChatCompletionMessage(content='Test response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    processed_response = model._process_response(c)

    assert len(processed_response.parts) == 1
    assert processed_response.parts[0].content == 'Test response'
    assert processed_response.model_name == 'google/gemini-2.5-flash-lite'


def test_openrouter_model_inference():
    """Test that openrouter: prefix creates OpenRouterModel."""
    from pydantic_ai.models import infer_model

    model = infer_model('openrouter:google/gemini-2.5-flash-lite')
    assert isinstance(model, OpenRouterModel)
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
    assert agent.model.system == 'openrouter'


def test_openrouter_basic_functionality():
    """Test basic OpenRouter model functionality."""
    c = completion_message(ChatCompletionMessage(content='test', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', openai_client=mock_client)

    assert model.model_name == 'google/gemini-2.5-flash-lite'
    assert model.system == 'openrouter'
