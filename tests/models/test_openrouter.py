import datetime
import json
import os
from collections.abc import AsyncIterable, Sequence
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError
from vcr.cassette import Cassette

from pydantic_ai import (
    Agent,
    BinaryContent,
    DocumentUrl,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    RunUsage,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolDefinition,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AdvisorTool, WebSearchTool

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import IsDatetime, IsStr, RequestCapture, message, try_import
from .mock_openai import MockOpenAI, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.openrouter import (
        OpenRouterModel,
        OpenRouterModelSettings,
        _map_openrouter_provider_details,  # pyright: ignore[reportPrivateUsage]
        _openrouter_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
        _OpenRouterChatCompletion,  # pyright: ignore[reportPrivateUsage]
        _OpenRouterChatCompletionChunk,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


async def test_openrouter_with_preset(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', provider=provider)
    settings = OpenRouterModelSettings(openrouter_preset='@preset/comedian')
    response = await model_request(model, [ModelRequest.user_text_prompt('Trains')], model_settings=settings)
    text_part = cast(TextPart, response.parts[0])
    assert text_part.content == snapshot(
        """\
Why did the train break up with the track?

Because it felt like their relationship was going nowhere.\
"""
    )


async def test_openrouter_with_native_options(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    # These specific settings will force OpenRouter to use the fallback model, since Gemini is not available via the xAI provider.
    settings = OpenRouterModelSettings(
        openrouter_models=['x-ai/grok-4'],
        openrouter_transforms=['middle-out'],
        openrouter_provider={'only': ['xai']},
    )
    response = await model_request(model, [ModelRequest.user_text_prompt('Who are you')], model_settings=settings)
    text_part = cast(TextPart, response.parts[0])
    assert text_part.content == snapshot(
        """\
I'm Grok, a helpful and maximally truthful AI built by xAI. I'm not based on any other companies' models—instead, I'm inspired by the Hitchhiker's Guide to the Galaxy and JARVIS from Iron Man. My goal is to assist with questions, provide information, and maybe crack a joke or two along the way.

What can I help you with today?\
"""
    )
    assert response.provider_details is not None
    assert response.provider_details['downstream_provider'] == 'xAI'
    assert response.provider_details['finish_reason'] == 'stop'


async def test_openrouter_stream_with_native_options(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    # These specific settings will force OpenRouter to use the fallback model, since Gemini is not available via the xAI provider.
    settings = OpenRouterModelSettings(
        openrouter_models=['x-ai/grok-4'],
        openrouter_transforms=['middle-out'],
        openrouter_provider={'only': ['xai']},
    )

    async with model_request_stream(
        model, [ModelRequest.user_text_prompt('Who are you')], model_settings=settings
    ) as stream:
        assert stream.provider_details == snapshot(None)
        assert stream.finish_reason == snapshot(None)

        _ = [chunk async for chunk in stream]

        assert stream.provider_details is not None
        assert stream.provider_details == snapshot(
            {
                'timestamp': datetime.datetime(2025, 11, 2, 6, 14, 57, tzinfo=datetime.timezone.utc),
                'finish_reason': 'completed',
                'cost': 0.00333825,
                'upstream_inference_cost': None,
                'is_byok': False,
                'upstream_inference_prompt_cost': 0.00053325,
                'upstream_inference_completions_cost': 0.002805,
                'downstream_provider': 'xAI',
            }
        )
        # Explicitly verify native_finish_reason is 'completed' and wasn't overwritten by the
        # final usage chunk (which has native_finish_reason: null, see cassette for details)
        assert stream.provider_details['finish_reason'] == 'completed'
        assert stream.finish_reason == snapshot('stop')


async def test_openrouter_stream_with_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel(
        'openai/o3',
        provider=provider,
        settings=OpenRouterModelSettings(openrouter_reasoning={'effort': 'high'}),
    )

    async with model_request_stream(model, [ModelRequest.user_text_prompt('Who are you')]) as stream:
        chunks = [chunk async for chunk in stream]

        thinking_event_start = chunks[0]
        assert isinstance(thinking_event_start, PartStartEvent)
        thinking_part = thinking_event_start.part
        assert isinstance(thinking_part, ThinkingPart)
        assert thinking_part.id == 'rs_0aa4f2c435e6d1dc0169082486816c8193a029b5fc4ef1764f'
        assert thinking_part.content == ''
        assert thinking_part.provider_name == 'openrouter'
        # After fix: signature and provider_details are now properly preserved
        assert thinking_part.signature is not None
        assert thinking_part.provider_details is not None
        assert thinking_part.provider_details['type'] == 'reasoning.encrypted'
        assert thinking_part.provider_details['format'] == 'openai-responses-v1'

        thinking_event_end = chunks[1]
        assert isinstance(thinking_event_end, PartEndEvent)
        thinking_part_end = thinking_event_end.part
        assert isinstance(thinking_part_end, ThinkingPart)
        assert thinking_part_end.id == 'rs_0aa4f2c435e6d1dc0169082486816c8193a029b5fc4ef1764f'
        assert thinking_part_end.signature is not None


async def test_openrouter_stream_error(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('minimax/minimax-m2:free', provider=provider)
    settings = OpenRouterModelSettings(max_tokens=10)

    with pytest.raises(ModelHTTPError):
        async with model_request_stream(
            model, [ModelRequest.user_text_prompt('Hello there')], model_settings=settings
        ) as stream:
            _ = [chunk async for chunk in stream]


async def test_openrouter_tool_calling(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)

    class Divide(BaseModel):
        """Divide two numbers."""

        numerator: float
        denominator: float
        on_inf: Literal['error', 'infinity'] = 'infinity'

    model = OpenRouterModel('mistralai/mistral-small', provider=provider)
    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('What is 123 / 456?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=Divide.__name__.lower(),
                    description=Divide.__doc__,
                    parameters_json_schema=Divide.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )

    assert len(response.parts) == 1

    tool_call_part = response.parts[0]
    assert isinstance(tool_call_part, ToolCallPart)
    assert tool_call_part.tool_call_id == snapshot('3sniiMddS')
    assert tool_call_part.tool_name == 'divide'
    assert tool_call_part.args == snapshot('{"numerator": 123, "denominator": 456, "on_inf": "infinity"}')

    mapped_messages = await model._map_messages([response], ModelRequestParameters())  # type: ignore[reportPrivateUsage]
    tool_call_message = mapped_messages[0]
    assert tool_call_message['role'] == 'assistant'
    assert tool_call_message.get('content') is None
    assert tool_call_message.get('tool_calls') == snapshot(
        [
            {
                'id': '3sniiMddS',
                'type': 'function',
                'function': {
                    'name': 'divide',
                    'arguments': '{"numerator": 123, "denominator": 456, "on_inf": "infinity"}',
                },
            }
        ]
    )


async def test_openrouter_with_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    request = ModelRequest.user_text_prompt(
        "What was the impact of Voltaire's writings on modern french culture? Think about your answer."
    )

    model = OpenRouterModel('z-ai/glm-4.6', provider=provider)
    response = await model_request(model, [request])

    assert len(response.parts) == 2

    thinking_part = response.parts[0]
    assert isinstance(thinking_part, ThinkingPart)
    assert thinking_part.id == snapshot(None)
    assert thinking_part.content is not None
    assert thinking_part.signature is None


async def test_openrouter_preserve_reasoning_block(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-5-mini', provider=provider)

    messages: Sequence[ModelMessage] = []
    messages.append(ModelRequest.user_text_prompt('Hello!'))
    messages.append(await model_request(model, messages))
    messages.append(
        ModelRequest.user_text_prompt("What was the impact of Voltaire's writings on modern french culture?")
    )
    messages.append(await model_request(model, messages))

    openai_messages = await model._map_messages(messages, ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assistant_message = openai_messages[1]
    assert assistant_message['role'] == 'assistant'
    assert 'reasoning_details' not in assistant_message

    assistant_message = openai_messages[3]
    assert assistant_message['role'] == 'assistant'
    assert 'reasoning_details' in assistant_message

    reasoning_details = assistant_message['reasoning_details']
    assert len(reasoning_details) == 2

    reasoning_summary = reasoning_details[0]

    assert 'summary' in reasoning_summary
    assert reasoning_summary['type'] == 'reasoning.summary'
    assert reasoning_summary['format'] == 'openai-responses-v1'

    reasoning_encrypted = reasoning_details[1]

    assert 'data' in reasoning_encrypted
    assert reasoning_encrypted['type'] == 'reasoning.encrypted'
    assert reasoning_encrypted['format'] == 'openai-responses-v1'


async def test_openrouter_thinking_only_response_mapping() -> None:
    """A `ModelResponse` containing only OpenRouter `ThinkingPart`s still produces an assistant
    message carrying `reasoning_details`, even though the base class would skip emitting any
    message for an otherwise-empty response.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-5-mini', provider=provider)

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello!')]),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='thinking summary text',
                    provider_name='openrouter',
                    provider_details={
                        'type': 'reasoning.summary',
                        'format': 'openai-responses-v1',
                    },
                )
            ],
        ),
        ModelRequest(parts=[UserPromptPart(content='Follow up?')]),
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    assistant_message = mapped[1]
    assert assistant_message['role'] == 'assistant'
    assert assistant_message.get('content') is None
    assert assistant_message['reasoning_details'] == [  # type: ignore[reportGeneralTypeIssues]
        {
            'type': 'reasoning.summary',
            'id': None,
            'format': 'openai-responses-v1',
            'index': None,
            'summary': 'thinking summary text',
        }
    ]


async def test_openrouter_video_url_mapping() -> None:
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Count the students.',
                        VideoUrl(url='https://example.com/video.mp4'),
                    ]
                )
            ]
        )
    ]

    mapped_messages = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped_messages[0].get('content')
    assert content is not None
    assert isinstance(content, list)

    assert content[0] == {'type': 'text', 'text': 'Count the students.'}
    assert content[1] == {'type': 'video_url', 'video_url': {'url': 'https://example.com/video.mp4'}}


async def test_openrouter_binary_content_video_mapping() -> None:
    """Test that `BinaryContent` with a video media type maps to a `video_url` part."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    binary_video = BinaryContent(data=b'video-bytes', media_type='video/mp4')

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Count the students.',
                        binary_video,
                    ]
                )
            ]
        )
    ]

    mapped_messages = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped_messages[0].get('content')
    assert content is not None
    assert isinstance(content, list)

    assert content[0] == {'type': 'text', 'text': 'Count the students.'}
    assert content[1] == {
        'type': 'video_url',
        'video_url': {'url': binary_video.data_uri},
    }


async def test_openrouter_video_url_force_download() -> None:
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    with patch('pydantic_ai.models.openrouter.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:video/mp4;base64,AAAA',
            'data_type': 'mp4',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Count the students.',
                            VideoUrl(url='https://example.com/video.mp4', force_download=True),
                        ]
                    )
                ]
            )
        ]

        mapped_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
            messages, ModelRequestParameters()
        )
        content = mapped_messages[0].get('content')
        assert content is not None
        assert isinstance(content, list)

        assert content[1] == {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,AAAA'}}
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[0][0].url == 'https://example.com/video.mp4'
        assert call_args[1]['data_format'] == 'base64_uri'
        assert call_args[1]['type_format'] == 'extension'


async def test_openrouter_video_url_no_force_download() -> None:
    """Test that `force_download=False` does not call `download_item` for `VideoUrl`."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    with patch('pydantic_ai.models.openrouter.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Count the students.',
                            VideoUrl(url='https://example.com/video.mp4', force_download=False),
                        ]
                    )
                ]
            )
        ]

        mapped_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
            messages, ModelRequestParameters()
        )
        content = mapped_messages[0].get('content')
        assert content is not None
        assert isinstance(content, list)

        assert content[1] == {'type': 'video_url', 'video_url': {'url': 'https://example.com/video.mp4'}}
        mock_download.assert_not_called()


async def test_openrouter_video_url_public_api(
    allow_model_requests: None, openrouter_api_key: str
) -> None:  # pragma: lax no cover
    """Test `VideoUrl` support through the public `Agent.run` API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        [
            'What is in this video?',
            VideoUrl(url='https://upload.wikimedia.org/wikipedia/commons/8/8f/Panda_at_Smithsonian_zoo.webm'),
        ]
    )

    assert isinstance(result.output, str)
    assert result.output == snapshot("""\
This video features a giant panda in an enclosure designed to resemble its natural habitat. The enclosure includes:
- **Rocks and terrain:** Various sized rocks create a textured landscape.
- **Bamboo:** Fresh bamboo shoots are scattered around, which the panda is seen eating.
- **Background mural:** A painted mural on the back wall depicts a mountainous, green landscape, enhancing the immersive feel of the habitat.
- **Window:** A clear window is visible in the upper part of the background, likely part of the viewing area for visitors.
- **Enrichment toy:** A large, round, light brown object (possibly a ball or feeder) is seen on the rocks, likely an enrichment toy for the panda.
- **Panda:** The main subject is a black and white giant panda, which is actively eating bamboo at the bottom right of the frame, occasionally looking up.\
""")


async def test_openrouter_binary_content_video_public_api(
    allow_model_requests: None, openrouter_api_key: str, video_content: BinaryContent, vcr: Cassette
) -> None:  # pragma: lax no cover
    """Test `BinaryContent` video support through the public `Agent.run` API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run(['What is in this video? Answer in one short sentence.', video_content])
    assert isinstance(result.output, str)
    assert result.output == snapshot(
        "The video shows a camera on a tripod recording a scenic mountain landscape, with a preview of the shot visible on the camera's screen."
    )

    assert vcr is not None
    request_body = single_request_body(vcr)

    video_content_part = request_body['messages'][0]['content'][1]
    assert video_content_part['type'] == 'video_url'
    assert video_content_part['video_url']['url'].startswith('data:video/mp4;base64,')


async def test_openrouter_errors_raised(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.', retries={'tools': 1, 'output': 1})
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Tell me a joke.')
    assert str(exc_info.value) == snapshot(
        "status_code: 429, model_name: google/gemini-2.0-flash-exp:free, body: {'code': 429, 'message': 'Provider returned error', 'metadata': {'provider_name': 'Google', 'raw': 'google/gemini-2.0-flash-exp:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations'}}"
    )


async def test_openrouter_usage(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-5-mini', provider=provider)
    agent = Agent(model, instructions='Be helpful.', retries={'tools': 1, 'output': 1})

    result = await agent.run('Tell me about Venus')

    assert result.usage == snapshot(
        RunUsage(
            input_tokens=17,
            output_tokens=1515,
            details={'reasoning_tokens': 704},
            output_reasoning_tokens=704,
            requests=1,
            cost=Decimal('0.00303425'),
        )
    )

    settings = OpenRouterModelSettings(openrouter_usage={'include': True})

    result = await agent.run('Tell me about Mars', model_settings=settings)

    assert result.usage == snapshot(
        RunUsage(
            input_tokens=17,
            output_tokens=2177,
            details={'is_byok': 0, 'reasoning_tokens': 960, 'image_tokens': 0},
            output_reasoning_tokens=960,
            requests=1,
            cost=Decimal('0.00435825'),
        )
    )

    last_message = message(result.all_messages(), ModelResponse, index=-1)
    assert last_message.provider_details == snapshot(
        {
            'finish_reason': 'completed',
            'downstream_provider': 'OpenAI',
            'cost': 0.00435825,
            'upstream_inference_cost': None,
            'upstream_inference_prompt_cost': 4.25e-06,
            'upstream_inference_completions_cost': 0.004354,
            'is_byok': False,
            'timestamp': IsDatetime(),
        }
    )


async def test_openrouter_validate_non_json_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        model._process_response('This is not JSON!')  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'Invalid response from openrouter chat completions endpoint, expected JSON data'
    )


async def test_openrouter_validate_error_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant'}, finish_reason='error', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )
    response.error = {'message': 'This response has an error attribute', 'code': 200}  # type: ignore[reportAttributeAccessIssue]

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(response)  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'status_code: 200, model_name: test, body: This response has an error attribute'
    )


async def test_openrouter_with_provider_details_but_no_parent_details(openrouter_api_key: str) -> None:
    class TestOpenRouterModel(OpenRouterModel):
        def _process_provider_details(self, response: ChatCompletion) -> dict[str, Any] | None:
            assert isinstance(response, _OpenRouterChatCompletion)
            openrouter_details = _map_openrouter_provider_details(response)
            return openrouter_details or None

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = TestOpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant', 'content': 'test'}, finish_reason='stop', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=1704067200, object='chat.completion', model='test', provider='TestProvider'
    )
    result = model._process_response(response)  # type: ignore[reportPrivateUsage]

    assert result.provider_details == snapshot(
        {
            'downstream_provider': 'TestProvider',
            'finish_reason': 'stop',
            'timestamp': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        }
    )


async def test_openrouter_map_messages_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet:thinking', provider=provider)

    user_message = ModelRequest.user_text_prompt('Who are you. Think about it.')
    response = await model_request(model, [user_message])

    mapped_messages = await model._map_messages([user_message, response], ModelRequestParameters())  # type: ignore[reportPrivateUsage]

    assert len(mapped_messages) == 2
    assert mapped_messages[1]['reasoning_details'] == snapshot(  # type: ignore[reportGeneralTypeIssues]
        [
            {
                'id': None,
                'type': 'reasoning.text',
                'text': """\
This question is asking me about my identity. Let me think about how to respond clearly and accurately.

I am Claude, an AI assistant created by Anthropic. I'm designed to be helpful, harmless, and honest in my interactions with humans. I don't have a physical form - I exist as a large language model running on computer hardware. I don't have consciousness, sentience, or feelings in the way humans do. I don't have personal experiences or a life outside of these conversations.

My capabilities include understanding and generating natural language text, reasoning about various topics, and attempting to be helpful to users in a wide range of contexts. I have been trained on a large corpus of text data, but my training data has a cutoff date, so I don't have knowledge of events that occurred after my training.

I have certain limitations - I don't have the ability to access the internet, run code, or interact with external systems unless given specific tools to do so. I don't have perfect knowledge and can make mistakes.

I'm designed to be conversational and to engage with users in a way that's helpful and informative, while respecting important ethical boundaries.\
""",
                'signature': 'ErcBCkgICBACGAIiQHtMxpqcMhnwgGUmSDWGoOL9ZHTbDKjWnhbFm0xKzFl0NmXFjQQxjFj5mieRYY718fINsJMGjycTVYeiu69npakSDDrsnKYAD/fdcpI57xoMHlQBxI93RMa5CSUZIjAFVCMQF5GfLLQCibyPbb7LhZ4kLIFxw/nqsTwDDt6bx3yipUcq7G7eGts8MZ6LxOYqHTlIDx0tfHRIlkkcNCdB2sUeMqP8e7kuQqIHoD52GAI=',
                'format': 'anthropic-claude-v1',
                'index': 0,
            }
        ]
    )


async def test_openrouter_tool_optional_parameters(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)

    class FindEducationContentFilters(BaseModel):
        title: str | None = None

    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('Can you find me any education content?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name='find_education_content',
                    description='',
                    parameters_json_schema=FindEducationContentFilters.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )

    assert len(response.parts) == 2

    tool_call_part = response.parts[1]
    assert isinstance(tool_call_part, ToolCallPart)
    assert tool_call_part.tool_call_id == snapshot('toolu_vrtx_015QAXScZzRDPttiPoc34AdD')
    assert tool_call_part.tool_name == 'find_education_content'
    assert tool_call_part.args == snapshot(None)

    mapped_messages = await model._map_messages([response], ModelRequestParameters())  # type: ignore[reportPrivateUsage]
    tool_call_message = mapped_messages[0]
    assert tool_call_message['role'] == 'assistant'
    assert tool_call_message.get('content') == snapshot("I'll search for education content for you.")
    assert tool_call_message.get('tool_calls') == snapshot(
        [
            {
                'id': 'toolu_vrtx_015QAXScZzRDPttiPoc34AdD',
                'type': 'function',
                'function': {
                    'name': 'find_education_content',
                    'arguments': '{}',
                },
            }
        ]
    )


async def test_openrouter_streaming_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    agent = Agent(
        model=model,
        model_settings=OpenRouterModelSettings(openrouter_reasoning={'enabled': True}),
    )

    async with agent.run_stream('What is 2+2?') as stream:
        _ = await stream.get_output()

        assert stream.response.parts == snapshot(
            [
                ThinkingPart(
                    content='This is a simple arithmetic question. 2+2 equals 4.',
                    signature='Et0BCkgIChACGAIqQA2s7h7tA7IG35fbwVkou9PM2hANVJNUwcEM4q12fTRDK6y3v6YoEvJ+7bko8wnW/GLsQFXadaJPAEMCpLkhI9ISDLjFkeR1aVUIvdCtyBoMrUTovh0jwk+wpnZWIjANV3e6VVdgbGSsEyyTHO6KMmVtqqs79f9blnVdJmmMIwMyTi6bEtG59+jTU7v1zlsqQ2IKGZILOlr6adh0Aam7zYttvisys+wjyZZXU1y/Srz0nmp1cFgVOJe1BLKQI3SSRrjsqQC0uAEUZy0GX0Rq1AXjvIcYAQ==',
                    provider_name='openrouter',
                    provider_details={'format': 'anthropic-claude-v1', 'index': 0, 'type': 'reasoning.text'},
                ),
                TextPart(content='2 + 2 = 4'),
            ]
        )


async def test_openrouter_streamed_reasoning_details_are_preserved(
    allow_model_requests: None,
) -> None:
    """A streamed OpenRouter response keeps details with distinct indexes separate.

    Mock-based rather than VCR — reasoning details spread across distinct indexes can't be reliably elicited
    from a live provider, so the chunks are hand-built.
    """

    async def consume_events(_: RunContext[object], event_stream: AsyncIterable[Any]) -> None:
        async for _event in event_stream:
            pass

    def reasoning_chunk(
        reasoning_detail: dict[str, Any], *, content: str = '', finish_reason: str | None = None
    ) -> ChatCompletionChunk:
        return _OpenRouterChatCompletionChunk.model_validate(
            {
                'id': 'gen-123',
                'choices': [
                    {
                        'index': 0,
                        'delta': {'role': 'assistant', 'content': content, 'reasoning_details': [reasoning_detail]},
                        'finish_reason': finish_reason,
                    }
                ],
                'created': 1704067200,
                'model': 'openai/gpt-5.6-luna',
                'object': 'chat.completion.chunk',
                'provider': 'OpenAI',
            }
        )

    mock_client = MockOpenAI(
        stream=[
            reasoning_chunk(
                {'type': 'reasoning.summary', 'summary': 'first summary', 'format': 'openai-responses-v1', 'index': 0}
            ),
            reasoning_chunk(
                {
                    'type': 'reasoning.encrypted',
                    'id': 'rs_123',
                    'data': 'encrypted reasoning',
                    'format': 'openai-responses-v1',
                    'index': 1,
                }
            ),
            reasoning_chunk(
                {'type': 'reasoning.summary', 'summary': 'second summary', 'format': 'openai-responses-v1', 'index': 2},
                content='first answer',
                finish_reason='stop',
            ),
        ],
    )
    model = OpenRouterModel('openai/gpt-5.6-luna', provider=OpenRouterProvider(openai_client=cast(Any, mock_client)))
    agent = Agent(model)

    result = await agent.run('first prompt', event_stream_handler=consume_events)

    assert result.response.parts == [
        ThinkingPart(
            content='first summary',
            provider_name='openrouter',
            provider_details={'format': 'openai-responses-v1', 'index': 0, 'type': 'reasoning.summary'},
        ),
        ThinkingPart(
            content='',
            id='rs_123',
            signature='encrypted reasoning',
            provider_name='openrouter',
            provider_details={'format': 'openai-responses-v1', 'index': 1, 'type': 'reasoning.encrypted'},
        ),
        ThinkingPart(
            content='second summary',
            provider_name='openrouter',
            provider_details={'format': 'openai-responses-v1', 'index': 2, 'type': 'reasoning.summary'},
        ),
        TextPart(content='first answer'),
    ]


async def test_openrouter_no_openrouter_details(openrouter_api_key: str) -> None:
    """Test _process_provider_details when _map_openrouter_provider_details returns empty dict."""
    from unittest.mock import patch

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant', 'content': 'test'}, finish_reason='stop', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=1704067200, object='chat.completion', model='test', provider='TestProvider'
    )

    with patch('pydantic_ai.models.openrouter._map_openrouter_provider_details', return_value={}):
        result = model._process_response(response)  # type: ignore[reportPrivateUsage]

    # With empty openrouter_details, we should still get the parent's provider_details (timestamp + finish_reason)
    assert result.provider_details == snapshot(
        {'finish_reason': 'stop', 'timestamp': datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)}
    )


async def test_openrouter_google_nested_schema(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Test that nested schemas with $defs/$ref work correctly with OpenRouter + Gemini.

    This verifies the fix for https://github.com/pydantic/pydantic-ai/issues/3617
    where OpenRouter's translation layer didn't support modern JSON Schema features.
    """
    from enum import Enum

    provider = OpenRouterProvider(api_key=openrouter_api_key)

    class LevelType(str, Enum):
        ground = 'ground'
        basement = 'basement'
        floor = 'floor'
        attic = 'attic'

    class SpaceType(str, Enum):
        entryway = 'entryway'
        living_room = 'living-room'
        kitchen = 'kitchen'
        bedroom = 'bedroom'
        bathroom = 'bathroom'
        garage = 'garage'

    class InsertLevelArg(BaseModel):
        level_name: str
        level_type: LevelType

    class SpaceArg(BaseModel):
        space_name: str
        space_type: SpaceType

    class InsertedLevel(BaseModel):
        """Result of inserting a level."""

        level_name: str
        level_type: LevelType
        space_count: int

    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent: Agent[object, InsertedLevel] = Agent(model, output_type=InsertedLevel)

    @agent.tool_plain
    def insert_level_with_spaces(level: InsertLevelArg | None, spaces: list[SpaceArg]) -> str:
        """Insert a level with its spaces."""
        return f'Inserted level {level} with {len(spaces)} spaces'

    result = await agent.run("It's a house with a ground floor that has an entryway, a living room and a garage.")

    tool_call_message = result.all_messages()[1]
    assert tool_call_message.parts == snapshot(
        [
            ToolCallPart(
                tool_name='insert_level_with_spaces',
                args='{"spaces":[{"space_type":"entryway","space_name":"entryway"},{"space_name":"living_room","space_type":"living-room"},{"space_name":"garage","space_type":"garage"}],"level":{"level_type":"ground","level_name":"ground_floor"}}',
                tool_call_id='tool_insert_level_with_spaces_3ZiChYzj8xER8HixJe7W',
            )
        ]
    )

    assert result.output.level_type == LevelType.ground
    assert result.output.space_count == 3


async def test_openrouter_file_annotation(
    allow_model_requests: None, openrouter_api_key: str, document_content: BinaryContent
) -> None:
    """Test that file annotations from OpenRouter are handled correctly.

    When sending files (e.g., PDFs) to OpenRouter, the response can include
    annotations with type="file". This test ensures those annotations are
    parsed without validation errors.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-5.1-codex-mini', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        user_prompt=[
            'What does this PDF contain? Answer in one short sentence.',
            document_content,
        ]
    )

    # The response should contain text (model may or may not include file annotations)
    assert isinstance(result.output, str)
    assert len(result.output) > 0


async def test_openrouter_file_annotation_validation(openrouter_api_key: str) -> None:
    """Test that file annotations from OpenRouter are correctly validated.

    This unit test verifies that responses containing type="file" annotations
    are parsed without validation errors, which was failing before the fix.
    """
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    message = ChatCompletionMessage.model_construct(
        role='assistant',
        content='Here is the summary of your file.',
        annotations=[
            {'type': 'file', 'file': {'filename': 'test.pdf', 'file_id': 'file-123'}},
        ],
    )
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    response = ChatCompletion.model_construct(
        id='test', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )

    # This should not raise a validation error
    result = model._process_response(response)  # type: ignore[reportPrivateUsage]
    text_part = cast(TextPart, result.parts[0])
    assert text_part.content == 'Here is the summary of your file.'
    assert result.provider_details is not None
    assert result.provider_details['annotations'] == snapshot(
        [{'type': 'file', 'file': {'filename': 'test.pdf', 'file_id': 'file-123'}}]
    )


async def test_openrouter_service_tier_completion(openrouter_api_key: str) -> None:
    """OpenRouter providers can return service_tier values outside the OpenAI Literal."""
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)

    message = ChatCompletionMessage.model_construct(role='assistant', content='hi')
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    response = ChatCompletion.model_construct(
        id='gen-123',
        choices=[choice],
        created=1234567890,
        object='chat.completion',
        model='google/gemini-2.5-flash',
        provider='Google',
        service_tier='standard',
    )

    result = model._process_response(response)  # type: ignore[reportPrivateUsage]
    text_part = cast(TextPart, result.parts[0])
    assert text_part.content == 'hi'


async def test_openrouter_service_tier_chunk() -> None:
    """OpenRouter streaming chunks can return service_tier values outside the OpenAI Literal."""
    data = {
        'id': 'gen-123',
        'choices': [
            {
                'index': 0,
                'delta': {'role': 'assistant', 'content': 'hi'},
                'finish_reason': 'stop',
                'native_finish_reason': 'stop',
            }
        ],
        'created': 1234567890,
        'model': 'google/gemini-2.5-flash',
        'object': 'chat.completion.chunk',
        'provider': 'Google',
        'service_tier': 'on_demand',
    }
    result = _OpenRouterChatCompletionChunk.model_validate(data)
    assert result.service_tier == 'on_demand'


async def test_openrouter_document_url_no_force_download(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test that OpenRouter passes DocumentUrl directly without downloading when force_download=False.

    OpenRouter supports file URLs directly in the Chat API, unlike native OpenAI which only
    supports base64-encoded data. This test verifies that when using OpenRouter, the URL
    is passed directly without being downloaded first.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)
    agent = Agent(model)

    pdf_url = 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'
    document_url = DocumentUrl(url=pdf_url, force_download=False)

    result = await agent.run(['What is the main content of this document?', document_url])
    assert 'dummy' in result.output.lower() or 'pdf' in result.output.lower()

    # Verify URL was passed directly (not downloaded and base64-encoded)
    assert vcr is not None
    request_body = single_request_body(vcr)
    file_content = request_body['messages'][0]['content'][1]
    assert file_content == snapshot(
        {
            'file': {
                'file_data': 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
                'filename': 'filename.pdf',
            },
            'type': 'file',
        }
    )


async def test_openrouter_supported_native_tools() -> None:
    """Test that OpenRouterModel declares support for WebSearchTool."""
    supported = OpenRouterModel.supported_native_tools()
    assert WebSearchTool in supported


async def test_openrouter_web_search_tool_request(allow_model_requests: None) -> None:
    """`WebSearchTool` maps portable settings to an `openrouter:web_search` server tool.

    A mocked client pins the exact request-payload mapping, so this is a unit test rather than a
    VCR test despite the module-level `vcr` mark; the provider-acceptance side is covered by the
    VCR test.
    """
    mock_client = MockOpenAI.create_mock(_openrouter_completion('done'))
    model = OpenRouterModel('openai/gpt-4.1', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(
        model,
        capabilities=[
            NativeTool(
                WebSearchTool(
                    search_context_size='high',
                    user_location={'city': 'London', 'country': 'GB', 'region': 'England', 'timezone': 'Europe/London'},
                    allowed_domains=['pydantic.dev'],
                    blocked_domains=['example.com'],
                    max_uses=2,
                    external_web_access=False,
                )
            )
        ],
    )

    result = await agent.run('hello')

    assert result.output == 'done'
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['tools'] == [
        {
            'type': 'openrouter:web_search',
            'parameters': {
                'search_context_size': 'high',
                'user_location': {
                    'type': 'approximate',
                    'city': 'London',
                    'country': 'GB',
                    'region': 'England',
                    'timezone': 'Europe/London',
                },
                'allowed_domains': ['pydantic.dev'],
                'excluded_domains': ['example.com'],
                'max_uses': 2,
            },
        }
    ]
    assert kwargs['extra_body'] == {}


async def test_openrouter_web_search_tool_is_after_tool_cache_and_advisor(allow_model_requests: None) -> None:
    """Server tools follow the cached function definition so the cache breakpoint remains stable.

    A mocked client pins the exact tool ordering and cache breakpoint, so this is a unit test
    rather than a VCR test despite the module-level `vcr` mark; the provider-acceptance side is
    covered by the VCR test.
    """
    mock_client = MockOpenAI.create_mock(_openrouter_completion('done'))
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(
        model,
        model_settings=OpenRouterModelSettings(openrouter_cache_tool_definitions=True),
        capabilities=[
            NativeTool(AdvisorTool(model='anthropic/claude-opus-4.8')),
            NativeTool(WebSearchTool()),
        ],
    )

    @agent.tool_plain
    def get_weather() -> str:
        return 'sunny'  # pragma: no cover

    result = await agent.run('hello')

    assert result.output == 'done'
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['tools'] == snapshot(
        [
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': '',
                    'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                },
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {
                'type': 'openrouter:advisor',
                'parameters': {'model': 'anthropic/claude-opus-4.8', 'forward_transcript': False},
            },
            {'type': 'openrouter:web_search', 'parameters': {'search_context_size': 'medium'}},
        ]
    )


async def test_openrouter_settings_to_openai_settings_does_not_add_web_search() -> None:
    """`WebSearchTool` is converted to a server tool after OpenAI settings are prepared.

    This calls the private converter directly with no provider request, so it is a unit test
    rather than a VCR test despite the module-level `vcr` mark.
    """
    settings = OpenRouterModelSettings()
    model_request_parameters = ModelRequestParameters(
        native_tools=[WebSearchTool(search_context_size='high')],
    )

    result = _openrouter_settings_to_openai_settings(settings, model_request_parameters)

    extra_body = cast(dict[str, Any], result.get('extra_body', {}))
    assert extra_body == {}


async def test_openrouter_prepare_request_does_not_mutate_caller_settings() -> None:
    """Repeated `prepare_request` calls must not mutate the caller's settings.

    `merge_model_settings` can return the model's own `settings` by identity, so the converter's
    `openrouter_` pops and `extra_body` updates would otherwise leak back onto the caller's object
    across calls. This is an API-free preparation-path test (no provider request),
    so it is a unit test rather than a VCR test despite the module-level `vcr` mark.
    """
    provider = OpenRouterProvider(api_key='mock-api-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)
    settings = OpenRouterModelSettings(
        openrouter_models=['vendor/model'],
        openrouter_provider={'only': ['provider']},
        openrouter_usage={'include': True},
        extra_body={'caller_key': 'kept', 'plugins': [{'id': 'custom'}]},
    )
    original = deepcopy(settings)
    params = ModelRequestParameters(native_tools=[WebSearchTool(search_context_size='medium')])

    first, _ = model.prepare_request(settings, params)
    assert first is not None
    second, _ = model.prepare_request(settings, params)
    assert second is not None

    # The caller's settings object is preserved across both calls.
    assert settings == original

    first_extra_body = cast(dict[str, Any], first.get('extra_body', {}))
    second_extra_body = cast(dict[str, Any], second.get('extra_body', {}))
    assert first_extra_body['plugins'] == [{'id': 'custom'}]
    assert second_extra_body['plugins'] == [{'id': 'custom'}]
    # openrouter_* values are moved into extra_body without stripping the caller's originals.
    assert first_extra_body['models'] == ['vendor/model']
    assert first_extra_body['provider'] == {'only': ['provider']}
    assert first_extra_body['usage'] == {'include': True}
    # The caller's pre-existing extra_body entries are preserved.
    assert first_extra_body['caller_key'] == 'kept'
    # The openrouter_* keys remain on the original caller settings object.
    assert settings.get('openrouter_models') == ['vendor/model']
    assert settings.get('openrouter_provider') == {'only': ['provider']}
    assert settings.get('openrouter_usage') == {'include': True}


def _openrouter_completion(content: str) -> ChatCompletion:
    """Build a minimal completion that satisfies `_OpenRouterChatCompletion` validation.

    The shared `completion_message` helper builds a plain OpenAI `ChatCompletion`, which lacks
    the `provider` field OpenRouter responses always carry.
    """
    message = ChatCompletionMessage.model_construct(role='assistant', content=content)
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    return ChatCompletion.model_construct(
        id='123', choices=[choice], created=1704067200, model='test', object='chat.completion', provider='test'
    )


async def test_openrouter_wraps_mid_conversation_system_prompt(allow_model_requests: None) -> None:
    """OpenRouter gets the fallback while the same history stays native on OpenAI.

    Mocked clients pin the rendered request bodies directly: OpenRouter accepts an inline `system`
    message but silently transforms it, so a successful gateway response cannot prove native support.
    """
    history = [
        ModelRequest(parts=[SystemPromptPart(content='You are helpful.'), UserPromptPart(content='Hello.')]),
        ModelResponse(parts=[TextPart(content='Hi.')]),
        ModelRequest(parts=[SystemPromptPart(content='Answer in one sentence.')]),
    ]
    openrouter_client = MockOpenAI.create_mock(_openrouter_completion('Done.'))
    openai_client = MockOpenAI.create_mock(_openrouter_completion('Done.'))

    openrouter_model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(openai_client=openrouter_client))
    openai_model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=openai_client))

    await Agent(openrouter_model).run('Continue.', message_history=history)
    await Agent(openai_model).run('Continue.', message_history=history)

    assert openrouter_model.profile.get('supports_inline_system_prompts') is False
    assert openai_model.profile.get('supports_inline_system_prompts') is True
    assert get_mock_chat_completion_kwargs(openrouter_client)[0]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hello.'},
            {'role': 'assistant', 'content': 'Hi.'},
            {'role': 'user', 'content': '<system>Answer in one sentence.</system>'},
            {'role': 'user', 'content': 'Continue.'},
        ]
    )
    assert get_mock_chat_completion_kwargs(openai_client)[0]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hello.'},
            {'role': 'assistant', 'content': 'Hi.'},
            {'role': 'system', 'content': 'Answer in one sentence.'},
            {'role': 'user', 'content': 'Continue.'},
        ]
    )


@pytest.mark.parametrize(
    'executor,advisor_kwargs,expected_parameters',
    [
        pytest.param(
            'openai/gpt-4.1',
            {'model': 'anthropic/claude-opus-4.8', 'max_tokens': 2048},
            {'model': 'anthropic/claude-opus-4.8', 'forward_transcript': False, 'max_completion_tokens': 2048},
            id='max-tokens-set',
        ),
        # claude-3.5-haiku is NOT a valid advisor executor on the Claude API, so the Anthropic
        # vendor profile excludes AdvisorTool — but the advisor is an OpenRouter gateway feature
        # available to any executor, so the gateway layer of `OpenRouterProvider.model_profile`
        # must win over the vendor profile. Also pins that `max_completion_tokens` is omitted
        # (not null) when `max_tokens` is unset.
        pytest.param(
            'anthropic/claude-3.5-haiku',
            {'model': 'anthropic/claude-opus-4.8'},
            {'model': 'anthropic/claude-opus-4.8', 'forward_transcript': False},
            id='any-executor',
        ),
    ],
)
async def test_openrouter_advisor_tool_request(
    allow_model_requests: None,
    executor: str,
    advisor_kwargs: dict[str, Any],
    expected_parameters: dict[str, Any],
) -> None:
    """`AdvisorTool` maps to an `openrouter:advisor` server-tool entry in the request `tools` array.

    Unit test with a mocked client because our cassette matchers aren't sensitive to the request
    body, so a VCR test wouldn't pin the mapped payload. `forward_transcript` remains `False` so
    OpenRouter uses its default context behavior until provider-specific native tool parameters
    can expose this choice to users.
    """
    mock_client = MockOpenAI.create_mock(_openrouter_completion('done'))
    model = OpenRouterModel(executor, provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(model, capabilities=[NativeTool(AdvisorTool(**advisor_kwargs))])

    result = await agent.run('hello')

    assert result.output == 'done'
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['tools'] == [{'type': 'openrouter:advisor', 'parameters': expected_parameters}]


@pytest.mark.parametrize('field_kwargs', [{'caching': '5m'}, {'max_uses': 3}])
async def test_openrouter_advisor_tool_unsupported_fields(
    allow_model_requests: None, field_kwargs: dict[str, Any]
) -> None:
    """OpenRouter silently ignores the unsupported `caching` and `max_uses` fields.

    This mocked-client test pins the request body because the VCR matcher would not detect these
    unsupported fields accidentally being forwarded.
    """
    c = _openrouter_completion('done')
    mock_client = MockOpenAI.create_mock(c)
    model = OpenRouterModel('openai/gpt-4.1', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(model, capabilities=[NativeTool(AdvisorTool(model='anthropic/claude-opus-4.8', **field_kwargs))])

    result = await agent.run('hello')

    assert result.output == 'done'
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['tools'] == [
        {
            'type': 'openrouter:advisor',
            'parameters': {'model': 'anthropic/claude-opus-4.8', 'forward_transcript': False},
        }
    ]


_TOOL_FORCING_REQUEST_PARAMETERS = ModelRequestParameters(
    function_tools=[
        ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object', 'properties': {}}),
    ],
    output_tools=[
        ToolDefinition(name='final_result', parameters_json_schema={'type': 'object', 'properties': {}}, kind='output'),
    ],
    output_mode='tool',
    allow_text_output=False,
)


@pytest.mark.parametrize(
    ('model_name', 'settings', 'expected_tool_choice', 'expected_tool_names'),
    [
        # The headline case: structured output alone resolves to `tool_choice='required'` without the user
        # asking for forcing at all, which would make OpenRouter drop `reasoning` for an Anthropic model.
        # The inferred forcing falls back to `auto` so the reasoning request survives.
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'thinking': 'low'},
            'auto',
            ['get_weather', 'final_result'],
            id='anthropic-thinking-inferred-required',
        ),
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'openrouter_reasoning': {'effort': 'low'}},
            'auto',
            ['get_weather', 'final_result'],
            id='anthropic-openrouter-reasoning-inferred-required',
        ),
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'thinking': 'low', 'openrouter_reasoning': {'enabled': False}},
            'required',
            ['get_weather', 'final_result'],
            id='anthropic-openrouter-reasoning-disabled-overrides-thinking',
        ),
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'thinking': 'low', 'openrouter_reasoning': {'effort': 'none'}},
            'required',
            ['get_weather', 'final_result'],
            id='anthropic-openrouter-reasoning-none-overrides-thinking',
        ),
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'thinking': 'low', 'openrouter_reasoning': {}},
            'required',
            ['get_weather', 'final_result'],
            id='anthropic-empty-openrouter-reasoning-overrides-thinking',
        ),
        # Without thinking there is no incompatibility, so forcing still goes on the wire.
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {},
            'required',
            ['get_weather', 'final_result'],
            id='anthropic-no-thinking',
        ),
        # `thinking=False` maps to `reasoning.effort='none'`, which is not thinking either.
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'thinking': False},
            'required',
            ['get_weather', 'final_result'],
            id='anthropic-thinking-disabled',
        ),
        # `tool_choice='none'` with an output tool and no direct output resolves to that single tool, so
        # the fallback also has to filter the tools — `auto` alone wouldn't keep `get_weather` off limits.
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'thinking': 'low', 'tool_choice': 'none'},
            'auto',
            ['final_result'],
            id='anthropic-thinking-inferred-named',
        ),
        pytest.param(
            'anthropic/claude-sonnet-4.6',
            {'tool_choice': 'none'},
            {'type': 'function', 'function': {'name': 'final_result'}},
            ['get_weather', 'final_result'],
            id='anthropic-no-thinking-named',
        ),
        # Other downstream providers honor `reasoning` alongside a forced `tool_choice`, so nothing changes
        # for them — including when the user asks for forcing explicitly.
        pytest.param(
            'google/gemini-2.5-flash',
            {'thinking': 'low'},
            'required',
            ['get_weather', 'final_result'],
            id='google-thinking-inferred-required',
        ),
        pytest.param(
            'google/gemini-2.5-flash',
            {'thinking': 'low', 'tool_choice': 'required'},
            'required',
            ['get_weather', 'final_result'],
            id='google-thinking-explicit-required',
        ),
    ],
)
async def test_openrouter_forced_tool_choice_with_thinking(
    allow_model_requests: None,
    model_name: str,
    settings: dict[str, Any],
    expected_tool_choice: Any,
    expected_tool_names: list[str],
) -> None:
    """OpenRouter drops `reasoning` when a forced `tool_choice` reaches an Anthropic downstream model.

    Anthropic itself rejects that combination with an error, but the gateway swallows it and returns a
    response with zero reasoning tokens, so an inferred forcing is downgraded to `auto` to keep thinking.

    Unit test with a mocked client because our cassette matchers aren't sensitive to the request body, so
    a VCR test would keep passing against a stale recording if `tool_choice` regressed.
    """
    mock_client = MockOpenAI.create_mock(_openrouter_completion('done'))
    model = OpenRouterModel(model_name, provider=OpenRouterProvider(openai_client=mock_client))

    await model_request(
        model,
        [ModelRequest.user_text_prompt('hello')],
        model_settings=cast(OpenRouterModelSettings, settings),
        model_request_parameters=_TOOL_FORCING_REQUEST_PARAMETERS,
    )

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['tool_choice'] == expected_tool_choice
    assert [tool['function']['name'] for tool in kwargs['tools']] == expected_tool_names
    # The point of the fallback: the reasoning the user asked for still goes on the wire.
    if 'openrouter_reasoning' in settings:
        expected_reasoning = settings['openrouter_reasoning'] or None
    elif 'thinking' not in settings:
        expected_reasoning = None
    elif settings['thinking']:
        expected_reasoning = {'effort': 'low', 'enabled': True}
    else:
        expected_reasoning = {'effort': 'none'}
    assert kwargs['extra_body'].get('reasoning') == expected_reasoning


@pytest.mark.parametrize(
    ('tool_choice', 'expected_error'),
    [
        pytest.param(
            'required',
            "OpenRouter does not support tool_choice='required' with thinking mode. Disable thinking or use "
            "`tool_choice='auto'`; otherwise OpenRouter silently drops reasoning.",
            id='required',
        ),
        pytest.param(
            ['get_weather'],
            'OpenRouter does not support forcing specific tools with thinking mode. Disable thinking or use '
            "`tool_choice='auto'`; otherwise OpenRouter silently drops reasoning.",
            id='list',
        ),
    ],
)
async def test_openrouter_explicit_forced_tool_choice_with_thinking_errors(
    allow_model_requests: None, tool_choice: Any, expected_error: str
) -> None:
    """An explicitly forced `tool_choice` errors instead of silently losing either the forcing or thinking.

    Mirrors `AnthropicModel`, which raises for explicit forcing under thinking and only falls back softly
    for a forcing the `tool_choice` resolution logic inferred.
    """
    mock_client = MockOpenAI.create_mock(_openrouter_completion('done'))
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(openai_client=mock_client))

    with pytest.raises(UserError) as exc_info:
        await model_request(
            model,
            [ModelRequest.user_text_prompt('hello')],
            model_settings=cast(OpenRouterModelSettings, {'thinking': 'low', 'tool_choice': tool_choice}),
            model_request_parameters=_TOOL_FORCING_REQUEST_PARAMETERS,
        )

    assert str(exc_info.value) == expected_error


async def test_openrouter_advisor_tool(allow_model_requests: None, openrouter_api_key: str) -> None:
    """End-to-end advisor consult through OpenRouter, recorded live.

    In this recorded Chat Completions response, the server-side consultation is not exposed as
    message parts. The server-tool-use counts in `usage`, surfaced via
    `provider_details['server_tool_use']`, verify that the advisor was invoked.

    The prompt explicitly asks the executor to consult so the recording reliably contains an
    advisor exchange.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4o-mini', provider=provider)
    agent = Agent(model, capabilities=[NativeTool(AdvisorTool(model='~anthropic/claude-opus-latest', max_tokens=1024))])

    result = await agent.run(
        'Consult your advisor tool for a recommendation first, then answer in one sentence: '
        'what should I name a Python retry library?'
    )

    assert result.output
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.provider_details is not None
    assert response.provider_details['server_tool_use'] == {'tool_calls_requested': 1, 'tool_calls_executed': 1}


async def test_openrouter_advisor_tool_stream(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Streaming counterpart of `test_openrouter_advisor_tool`.

    The advisor sub-inference does not stream; the server-tool-use counts arrive with the final
    usage chunk. Asserts they reach `provider_details['server_tool_use']` on the streamed response
    too, proving the non-streaming and streaming paths surface the advisor consult identically.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4o-mini', provider=provider)
    agent = Agent(model, capabilities=[NativeTool(AdvisorTool(model='~anthropic/claude-opus-latest', max_tokens=1024))])

    async with agent.run_stream(
        'Consult your advisor tool for a recommendation first, then answer in one sentence: '
        'what should I name a Python retry library?'
    ) as stream:
        assert await stream.get_output()

    assert stream.response.provider_details is not None
    assert stream.response.provider_details['server_tool_use'] == {'tool_calls_requested': 1, 'tool_calls_executed': 1}


async def test_openrouter_web_search_tool_usage(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """A recorded OpenRouter web search exposes its request count in provider details."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])

    result = await agent.run("Use web search to find Pydantic AI's GitHub repository and answer with its URL only.")

    assert result.output
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.provider_details is not None
    raw_response = json.loads(vcr.responses[0]['body']['string'])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert raw_response['usage']['server_tool_use_details'] == {'web_search_requests': 1}
    assert response.provider_details['server_tool_use'] == {'web_search_requests': 1}
    assert 'annotations' not in response.provider_details


_OPENROUTER_WEB_SEARCH_FULL_PARAMS_CASSETTE = (
    Path(__file__).parent / 'cassettes' / 'test_openrouter' / 'test_openrouter_web_search_tool_full_params.yaml'
)


@pytest.mark.skipif(
    not os.environ.get('OPENROUTER_API_KEY') and not _OPENROUTER_WEB_SEARCH_FULL_PARAMS_CASSETTE.is_file(),
    reason=(
        'verifies OpenRouter accepts the `user_location`, `allowed_domains`, and `blocked_domains` '
        'web-search wire names with a real request; requires OPENROUTER_API_KEY to record '
        '(run with --record-mode=rewrite) when the cassette is unavailable'
    ),
)
async def test_openrouter_web_search_tool_full_params(
    allow_model_requests: None, openrouter_api_key: str, request_capture: RequestCapture
) -> None:
    """Live provider-acceptance check for the web-search wire names.

    Unlike `test_openrouter_web_search_tool_request` (which asserts the outgoing payload against a mock
    client), this sends `user_location`, `allowed_domains`, and `blocked_domains` to OpenRouter so its
    acceptance of those names is verified by a recorded response, not just request construction. When
    the cassette is unavailable, record it with:
    `uv run pytest tests/models/test_openrouter.py::test_openrouter_web_search_tool_full_params --record-mode=rewrite`.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key, http_client=request_capture.client)
    model = OpenRouterModel('google/gemini-3.6-flash', provider=provider)
    agent = Agent(
        model,
        capabilities=[
            NativeTool(
                WebSearchTool(
                    search_context_size='low',
                    user_location={'city': 'London', 'country': 'GB', 'region': 'England', 'timezone': 'Europe/London'},
                    allowed_domains=['pydantic.dev'],
                    blocked_domains=['example.com'],
                    max_uses=1,
                )
            )
        ],
    )

    result = await agent.run('Reply with `ready`.')

    assert result.output
    assert request_capture.body()['tools'] == snapshot(
        [
            {
                'type': 'openrouter:web_search',
                'parameters': {
                    'search_context_size': 'low',
                    'user_location': {
                        'type': 'approximate',
                        'city': 'London',
                        'country': 'GB',
                        'region': 'England',
                        'timezone': 'Europe/London',
                    },
                    'allowed_domains': ['pydantic.dev'],
                    'excluded_domains': ['example.com'],
                    'max_uses': 1,
                },
            }
        ]
    )


async def test_openrouter_web_search_tool_usage_stream(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Streaming preserves the OpenRouter web-search request count from the final usage chunk."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])

    async with agent.run_stream(
        "Use web search to find Pydantic AI's GitHub repository and answer with its URL only."
    ) as stream:
        assert await stream.get_output()

    assert stream.response.provider_details is not None
    assert stream.response.provider_details['server_tool_use'] == {'web_search_requests': 1}
    assert 'annotations' not in stream.response.provider_details


async def test_openrouter_web_search_annotations(allow_model_requests: None, openrouter_api_key: str) -> None:
    """OpenRouter's `url_citation` annotations surface as the response's `annotations` provider detail.

    Uses a model whose downstream provider has no native search, because OpenRouter only returns
    annotations when a non-native search engine ran; native provider search returns none.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('deepseek/deepseek-chat', provider=provider)
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])

    result = await agent.run("Use web search to find Pydantic AI's GitHub repository and answer with its URL only.")

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.provider_details is not None
    annotations = response.provider_details['annotations']
    assert annotations[0] == snapshot(
        {
            'type': 'url_citation',
            'url_citation': {
                'end_index': 0,
                'start_index': 0,
                'title': 'AI Agent Framework, the Pydantic way',
                'url': 'https://github.com/pydantic/pydantic-ai',
                'content': """\
# pydantic/pydantic-ai

...

- Stars: 19265
- Forks: 2514
- Watchers: 19265
- Open issues: 702
- License: MIT License
- Homepage: https://pydantic.dev/pydantic-ai
- Default branch: main
- Created: 2024-06-21T15:55:04Z

...

AI Agent Framework, the Pydantic way

...

### Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.\
""",
            },
        }
    )
    assert [annotation['url_citation']['url'] for annotation in annotations] == snapshot(
        [
            'https://github.com/pydantic/pydantic-ai',
            'https://pydantic.dev/pydantic-ai',
            'https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0',
            'https://pydantic.dev/docs/ai/overview/',
            'https://github.com/pydantic/pydantic-ai/tree/refs/tags/v1.44.0',
        ]
    )


async def test_openrouter_web_search_annotations_stream(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Streaming accumulates annotations across deltas rather than keeping only the last chunk's."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('deepseek/deepseek-chat', provider=provider)
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool(max_uses=1))])

    async with agent.run_stream(
        "Use web search to find Pydantic AI's GitHub repository and answer with its URL only."
    ) as stream:
        assert await stream.get_output()

    assert stream.response.provider_details is not None
    annotations = stream.response.provider_details['annotations']
    assert [annotation['url_citation']['url'] for annotation in annotations] == snapshot(
        [
            'https://github.com/pydantic/pydantic-ai',
            'https://pydantic.dev/pydantic-ai',
            'https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0',
            'https://pydantic.dev/docs/ai/overview/',
            'https://github.com/pydantic/pydantic-ai/tree/refs/tags/v1.44.0',
        ]
    )


def test_openrouter_nested_provider_response() -> None:
    """OpenRouter sometimes nests the real response inside the 'provider' dict.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/3994.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    nested_completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider={
            'id': 'gen-123',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Hello from nested!'},
                    'finish_reason': 'stop',
                    'native_finish_reason': 'STOP',
                    'logprobs': None,
                }
            ],
            'model': 'google/gemini-3-flash-preview',
            'object': 'chat.completion',
            'provider': 'Google',
        },
        created=1234567890,
        usage=None,
    )

    model_response = model._process_response(nested_completion)  # type: ignore[reportPrivateUsage]

    assert model_response.parts == snapshot([TextPart(content='Hello from nested!')])
    assert model_response.provider_details == snapshot(
        {
            'downstream_provider': 'Google',
            'finish_reason': 'STOP',
            'timestamp': datetime.datetime(2009, 2, 13, 23, 31, 30, tzinfo=datetime.timezone.utc),
        }
    )


def test_openrouter_nested_provider_null_name() -> None:
    """Nested provider dict with provider=None falls back to 'unknown'."""
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider={
            'id': 'nested-gen-1',
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Hi'},
                    'finish_reason': 'stop',
                    'native_finish_reason': 'STOP',
                    'logprobs': None,
                }
            ],
            'model': 'openai/gpt-4.1-mini',
            'object': 'chat.completion',
            'provider': None,
            'created': 1234567890,
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        },
        created=1234567890,
    )

    result = model._process_response(completion)  # type: ignore[reportPrivateUsage]
    assert result.provider_details == snapshot(
        {
            'downstream_provider': 'unknown',
            'finish_reason': 'STOP',
            'timestamp': datetime.datetime(2009, 2, 13, 23, 31, 30, tzinfo=datetime.timezone.utc),
        }
    )


def test_openrouter_provider_dict_without_choices_raises() -> None:
    """Provider is a dict with no 'choices' key — no unwrap happens, validation fails.

    Also pins the boundary the no-completion shape depends on: `_OpenRouterNoCompletionResponse.provider`
    is typed `str | None`, so this body does not match it and stays fatal instead of becoming retryable.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider={'some_key': 'some_value'},
        created=1234567890,
    )

    with pytest.raises(UnexpectedModelBehavior):
        model._process_response(completion)  # type: ignore[reportPrivateUsage]


def test_openrouter_error_with_null_fields() -> None:
    """Error responses with null standard fields raise ModelHTTPError.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/3994.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    error_completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider=None,
        created=1234567890,
        usage=None,
        error={'code': 400, 'message': 'Invalid request parameters'},
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(error_completion)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert 'Invalid request parameters' in str(exc_info.value)


def test_openrouter_malformed_error_fallthrough() -> None:
    """Malformed error data falls through to validation, surfacing as UnexpectedModelBehavior.

    Also pins the second boundary the no-completion shape depends on: `_OpenRouterNoCompletionResponse.error`
    is typed `Literal[None]`, so this body does not match it and stays fatal. Widening that annotation would
    flip this test to a retryable `ModelAPIError`.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        model=None,
        object=None,
        provider=None,
        created=1234567890,
        usage=None,
        error='something went wrong',
    )

    with pytest.raises(UnexpectedModelBehavior):
        model._process_response(completion)  # type: ignore[reportPrivateUsage]


def _null_choices_completion(*, model: str | None = None, provider: str | None = None) -> ChatCompletion:
    """The no-completion body: null `choices`, no error envelope."""
    return ChatCompletion.model_construct(
        id=None, choices=None, model=model, object=None, provider=provider, created=1234567890, usage=None
    )


async def test_openrouter_null_choices_without_error_envelope_raises_model_api_error(
    allow_model_requests: None,
) -> None:
    """A null-`choices` body with no error envelope raises `ModelAPIError`, not a bare `ValidationError`.

    OpenRouter intermittently returns `{"choices": null, ...}` with no error field at all (a provider
    hiccup). A bare `ValidationError` surfaces as `UnexpectedModelBehavior`, which `FallbackModel`'s
    default `fallback_on=(ModelAPIError,)` does not match, so the transient never reaches fallback.

    Mock-based rather than VCR — and so are the null-`choices` tests below: the shape is an intermittent
    downstream-provider fault with no provoking input, so `--record-mode=rewrite` cannot capture it, and a
    hand-written cassette would give a synthesized body the provenance of a live recording.
    """
    mock_client = MockOpenAI.create_mock(_null_choices_completion())
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))

    with pytest.raises(ModelAPIError) as exc_info:
        await Agent(model).run('hello')

    assert not isinstance(exc_info.value, ModelHTTPError)  # not a faked HTTP status
    assert str(exc_info.value) == snapshot('OpenRouter returned a response with null `choices` and no error envelope')
    assert exc_info.value.model_name == snapshot('openai/gpt-4.1-mini')


async def test_openrouter_null_choices_named_provider_reports_body_model(allow_model_requests: None) -> None:
    """The realistic body names its downstream `provider`, and its `model` wins over the configured one.

    `provider` as a name string is the accept-side of the `str | None` discriminator that keeps a
    `provider` *dict* fatal, so it needs its own coverage.
    """
    completion = _null_choices_completion(model='google/gemini-2.5-flash', provider='Google')
    mock_client = MockOpenAI.create_mock(completion)
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))

    with pytest.raises(ModelAPIError) as exc_info:
        await Agent(model).run('hello')

    assert exc_info.value.model_name == snapshot('google/gemini-2.5-flash')


def _null_choices_chunk(provider: dict[str, str] | None = None) -> ChatCompletionChunk:
    """The no-completion body as a streaming chunk: null `choices`, no error envelope."""
    return ChatCompletionChunk.model_construct(
        id='gen-123',
        choices=None,
        created=1234567890,
        model=None,
        object=None,
        provider=provider,
        usage=None,
    )


async def test_openrouter_null_choices_streaming_raises_model_api_error(allow_model_requests: None) -> None:
    """Streaming parity: the same no-completion body is a mapped `ModelAPIError` on the stream too.

    `_OpenRouterChatCompletionChunk.choices` is required and non-null, so this chunk is rejected in
    `OpenRouterStreamedResponse._validate_response` before `OpenAIStreamedResponse` reaches its tolerant
    `if not chunk.choices` guard. It already failed there; what this pins is that it now fails as
    `ModelAPIError` rather than as the raw `ValidationError` this PR removes from the non-streamed path.

    The error surfaces while entering `agent.run_stream` — the agent consumes the first event there — so
    the block body never runs. Kept beside its non-streamed twin rather than moved to
    `tests/test_streaming_errors.py`: this PR exists to make the two paths agree, and reading them
    together is the point.
    """
    mock_client = MockOpenAI.create_mock_stream([_null_choices_chunk()])
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(model)

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert not isinstance(exc_info.value, ModelHTTPError)  # not a faked HTTP status
    assert str(exc_info.value) == snapshot('OpenRouter returned a response with null `choices` and no error envelope')
    assert exc_info.value.model_name == snapshot('openai/gpt-4.1-mini')


def _text_chunk(content: str, model: str) -> ChatCompletionChunk:
    """A healthy content chunk, so the no-completion chunk that follows it lands mid-stream rather than first."""
    return ChatCompletionChunk.model_validate(
        {
            'id': 'gen-123',
            'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}],
            'created': 1234567890,
            'model': model,
            'object': 'chat.completion.chunk',
            'provider': 'Google',
        }
    )


async def test_openrouter_null_choices_mid_stream_reports_first_chunk_model(allow_model_requests: None) -> None:
    """Arriving mid-stream rather than first, the same body still raises — and names the model differently.

    The classification is position-independent, but the reported `model_name` is not: `OpenAIChatModel`
    fixes `_model_name` from `first_chunk.model` when one is present, and the streamed call site passes
    that, where the non-streamed one passes the configured name. So a stream opened by a chunk naming
    `google/gemini-2.5-flash` reports that, not the configured `openai/gpt-4.1-mini` its first-chunk twin
    above asserts. Pinned because the twin alone reads as if the configured name always wins.
    """
    mock_client = MockOpenAI.create_mock_stream(
        [_text_chunk('hello ', model='google/gemini-2.5-flash'), _null_choices_chunk()]
    )
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(model)

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello') as result:
            async for _ in result.stream_text(delta=True):
                pass

    assert str(exc_info.value) == snapshot('OpenRouter returned a response with null `choices` and no error envelope')
    assert exc_info.value.model_name == snapshot('google/gemini-2.5-flash')


async def test_openrouter_streaming_malformed_chunk_stays_fatal(allow_model_requests: None) -> None:
    """A chunk that fails validation for some *other* reason keeps re-raising the original error.

    Streaming twin of `test_openrouter_provider_dict_without_choices_raises`: a `provider` dict is not the
    no-completion shape, so it must not be laundered into a retryable `ModelAPIError`. `match` pins the
    `provider` rejection specifically — without it the test also passes when the chunk is rejected for one
    of the reasons the no-completion shape shares (`choices`, `model`, `object`).
    """
    mock_client = MockOpenAI.create_mock_stream([_null_choices_chunk(provider={'some_key': 'some_value'})])
    model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(model)

    with pytest.raises(ValidationError, match='provider'):
        async with agent.run_stream('hello'):
            pass


async def test_openrouter_null_choices_triggers_fallback(allow_model_requests: None) -> None:
    """The point of `ModelAPIError`: a no-completion body now reaches `FallbackModel`.

    `fallback_on` defaults to `(ModelAPIError,)`, which a bare `ValidationError` never matched.

    Non-streamed only, deliberately: `FallbackModel`'s window is `Model.request_stream`'s own
    `__aenter__`, which returns before any chunk is validated, so the streamed twin raises instead of
    falling back. That asymmetry is `FallbackModel`'s, not OpenRouter's, and the test below pins it.
    """
    mock_client = MockOpenAI.create_mock(_null_choices_completion())
    openrouter_model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(FallbackModel(openrouter_model, TestModel(custom_output_text='fallback used')))

    result = await agent.run('hello')
    assert result.output == snapshot('fallback used')


async def test_openrouter_null_choices_streaming_does_not_trigger_fallback(allow_model_requests: None) -> None:
    """The streamed half of the same composition raises instead of falling back.

    `FallbackModel.request_stream` only guards `__aenter__`; once it has yielded, its own docstring says
    mid-stream failures propagate. `OpenAIChatModel._process_streamed_response` peeks the raw SDK chunk,
    so OpenRouter's validation runs later, during iteration — after the guard has closed. Pinned because
    this PR's first attempt at the test above asserted the streamed path *does* fall back, and failed.
    """
    mock_client = MockOpenAI.create_mock_stream([_null_choices_chunk()])
    openrouter_model = OpenRouterModel('openai/gpt-4.1-mini', provider=OpenRouterProvider(openai_client=mock_client))
    agent = Agent(FallbackModel(openrouter_model, TestModel(custom_output_text='fallback used')))

    with pytest.raises(ModelAPIError) as exc_info:
        async with agent.run_stream('hello'):
            pass

    assert str(exc_info.value) == snapshot('OpenRouter returned a response with null `choices` and no error envelope')


def test_openrouter_error_with_metadata() -> None:
    """Real-world error response with metadata field from #3994.

    OpenRouter returns error code 524 with extra metadata including the raw
    error and provider name. The extra fields should be ignored.
    """
    provider = OpenRouterProvider(api_key='test-key')
    model = OpenRouterModel('google/gemini-3-flash-preview', provider=provider)

    completion = ChatCompletion.model_construct(
        id=None,
        choices=None,
        created=1768361801,
        model=None,
        object=None,
        service_tier=None,
        system_fingerprint=None,
        usage=None,
        error={
            'message': 'Provider returned error',
            'code': 524,
            'metadata': {'raw': 'error code: 524', 'provider_name': 'Google'},
        },
        user_id='org_xxx',
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(completion)  # type: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 524
    assert 'Provider returned error' in str(exc_info.value)


async def test_openrouter_thinking_false_profile_gated_model(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Hybrid model whose intrinsic profile reports `supports_thinking=False` —
    `thinking=False` still reaches the wire as `reasoning.effort='none'` because
    OpenRouter's provider profile carries `supports_thinking=True`. See
    `test_openrouter_with_reasoning` above for the default-on baseline on glm-4.6."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('z-ai/glm-4.6', provider=provider)
    settings = OpenRouterModelSettings(thinking=False)

    response = await model_request(
        model, [ModelRequest.user_text_prompt('Reply with the single word: ok')], model_settings=settings
    )

    sent = single_request_body(vcr)
    assert sent['reasoning'] == {'effort': 'none'}

    assert not any(isinstance(part, ThinkingPart) for part in response.parts)


async def test_openrouter_thinking_true_emits_effort_medium(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """`thinking=True` is forwarded as `reasoning={'effort': 'medium', 'enabled': True}`.

    The explicit `enabled: True` matters for reasoning-optional OpenRouter routes
    (e.g. parts of `google/gemma-*`) that silently leave reasoning disabled when
    only `effort` is set. No-op for reasoning-by-default models."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    settings = OpenRouterModelSettings(thinking=True)

    response = await model_request(
        model, [ModelRequest.user_text_prompt('Reply with the single word: ok')], model_settings=settings
    )

    sent = single_request_body(vcr)
    assert sent['reasoning'] == {'effort': 'medium', 'enabled': True}

    # Response shape — pinning that `ThinkingPart` parsing survives the new wire format.
    assert response.parts == snapshot(
        [
            ThinkingPart(
                content=IsStr(),
                id=None,
                signature=IsStr(),
                provider_name='openrouter',
                provider_details={'format': 'anthropic-claude-v1', 'index': 0, 'type': 'reasoning.text'},
            ),
            TextPart(content='ok'),
        ]
    )
    assert response.timestamp == IsDatetime()
    assert response.provider_response_id == IsStr()


async def test_openrouter_thinking_false_supports_thinking_model(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Reasoning model whose intrinsic profile reports `supports_thinking=True` —
    `thinking=False` reaches the wire as `reasoning.effort='none'` via the
    transformer's unified-emit path."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    settings = OpenRouterModelSettings(thinking=False)

    response = await model_request(
        model, [ModelRequest.user_text_prompt('Reply with the single word: ok')], model_settings=settings
    )

    sent = single_request_body(vcr)
    assert sent['reasoning'] == {'effort': 'none'}

    assert not any(isinstance(part, ThinkingPart) for part in response.parts)


async def test_openrouter_thinking_high_emits_effort_high(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """`thinking='high'` is forwarded as `reasoning={'effort': 'high', 'enabled': True}`.

    Companion to `test_openrouter_thinking_true_emits_effort_medium` — exercises the
    `_OPENROUTER_EFFORT_MAP['high'] → 'high'` branch on the wire. Without this cassette
    the only wire-level effort value covered was `'medium'` (via `thinking=True`),
    leaving the `high`/`low`/`xhigh` branches unit-only."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=provider)
    settings = OpenRouterModelSettings(thinking='high')

    await model_request(
        model, [ModelRequest.user_text_prompt('Reply with the single word: ok')], model_settings=settings
    )

    sent = single_request_body(vcr)
    assert sent['reasoning'] == {'effort': 'high', 'enabled': True}


@pytest.mark.parametrize(
    'model_name,eager_enabled,expected_eager_key',
    [
        ('anthropic/claude-sonnet-4-5', True, True),
        ('anthropic/claude-sonnet-4-5', False, False),
        ('openai/gpt-5-mini', True, False),
    ],
    ids=['anthropic-enabled', 'anthropic-disabled', 'non-anthropic-enabled'],
)
async def test_eager_input_streaming_sent_to_openrouter(
    allow_model_requests: None,
    openrouter_api_key: str,
    vcr: Cassette,
    model_name: str,
    eager_enabled: bool,
    expected_eager_key: bool,
) -> None:
    """`eager_input_streaming` should appear on the outgoing tool payload only when enabled AND routed to Anthropic."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel(model_name, provider=provider)
    my_tool = ToolDefinition(name='get_weather', description='Get weather for a city')

    await model_request(
        model,
        [ModelRequest(parts=[UserPromptPart(content='hello')])],
        model_settings=AnthropicModelSettings(anthropic_eager_input_streaming=eager_enabled),
        model_request_parameters=ModelRequestParameters(function_tools=[my_tool], allow_text_output=True),
    )

    request_body = single_request_body(vcr)
    tool_param = request_body['tools'][0]
    assert tool_param['function']['name'] == 'get_weather'
    assert ('eager_input_streaming' in tool_param) is expected_eager_key
    if expected_eager_key:
        assert tool_param['eager_input_streaming'] is True
