"""Tests for OpenAI GPT-5.6 explicit prompt caching on the Chat Completions and Responses APIs.

Covers the `openai_prompt_cache_options` setting, `CachePoint` to `prompt_cache_breakpoint`
mapping, capability gating per API flavor and provider (OpenAI, OpenRouter, Azure), and
cache-write usage mapping.

These adapter-level tests intentionally use mocked SDK clients rather than VCR recordings:
they pin exact SDK request kwargs, provider-specific omission of unsupported fields, and
pre-request guards where no request may be sent at all. Recordings cannot reliably assert
omitted kwargs or a request that is never made, and cassette matchers are not always
sensitive to the request body. Real OpenAI and OpenRouter recordings of the accept path
would still be valuable additions.
"""

from __future__ import annotations as _annotations

from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import Agent, BinaryContent, CachePoint, ImageUrl
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .mock_openai import (
    MockOpenAI,
    MockOpenAIResponses,
    completion_message,
    get_mock_chat_completion_kwargs,
    get_mock_responses_kwargs,
    response_message,
)

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.types import chat, responses as resp
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIPromptCacheOptions,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def chat_completion(text: str = 'response', usage: CompletionUsage | None = None) -> chat.ChatCompletion:
    return completion_message(ChatCompletionMessage(content=text, role='assistant'), usage=usage)


def responses_completion(text: str = 'done', usage: ResponseUsage | None = None) -> resp.Response:
    return response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=cast('list[Content]', [ResponseOutputText(text=text, type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ],
        usage=usage,
    )


# ===== Chat Completions: breakpoints and request-level options =====


async def test_openai_chat_cache_point_and_options(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run(
        ['Stable context.', CachePoint(ttl='1h'), 'Use the context.']
    )

    assert result.output == 'response'
    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Stable context.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'text', 'text': 'Use the context.'},
                ],
            }
        ]
    )


@pytest.mark.parametrize('mode', ['implicit', 'explicit'])
async def test_openai_chat_prompt_cache_options_without_marker(
    allow_model_requests: None, mode: Literal['implicit', 'explicit']
):
    """Request-wide cache options are independent of explicit breakpoint markers."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': mode})

    await Agent(model, model_settings=settings).run('No explicit marker.')

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': mode}
    assert request['messages'] == [{'role': 'user', 'content': 'No explicit marker.'}]


async def test_openai_chat_multiple_cache_points(allow_model_requests: None):
    """Each marker attaches to its own preceding block; OpenAI writes up to four explicit breakpoints."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(['Product docs.', CachePoint(), 'Session context.', CachePoint(), 'Question.'])

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Product docs.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {
                        'type': 'text',
                        'text': 'Session context.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'text', 'text': 'Question.'},
                ],
            }
        ]
    )


async def test_openai_chat_cache_point_history_prefix_stability(allow_model_requests: None):
    """A serialized history preserves the cacheable prefix and its breakpoint across turns."""
    mock_client = MockOpenAI.create_mock([chat_completion('first'), chat_completion('second')])
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    first_result = await agent.run(['Stable context.', CachePoint(), 'First question.'])
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    await agent.run('Follow-up question.', message_history=history)

    first_request, second_request = get_mock_chat_completion_kwargs(mock_client)
    first_messages = cast('list[dict[str, Any]]', first_request['messages'])
    second_messages = cast('list[dict[str, Any]]', second_request['messages'])
    assert second_messages[0] == first_messages[0]
    assert second_messages[0] == snapshot(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Stable context.',
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                },
                {'type': 'text', 'text': 'First question.'},
            ],
        }
    )
    assert second_messages[-1] == {'role': 'user', 'content': 'Follow-up question.'}


@pytest.mark.parametrize(
    ('content_item', 'expected_type'),
    [
        (ImageUrl('https://example.com/reference.png'), 'image_url'),
        (BinaryContent(b'audio', media_type='audio/wav'), 'input_audio'),
        (BinaryContent(b'%PDF-1.4', media_type='application/pdf'), 'file'),
    ],
)
async def test_openai_chat_cache_point_supported_content_types(
    allow_model_requests: None,
    content_item: ImageUrl | BinaryContent,
    expected_type: Literal['image_url', 'input_audio', 'file'],
):
    """Pin breakpoint translation for every non-text Chat user-content type supported by Pydantic AI."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    result = await Agent(model).run([content_item, CachePoint()])

    assert result.output == 'response'
    request = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = cast('list[dict[str, Any]]', request['messages'])
    content = cast('list[dict[str, Any]]', messages[0]['content'])
    assert content[0]['type'] == expected_type
    assert content[0].get('prompt_cache_breakpoint') == {'mode': 'explicit'}


async def test_openai_chat_cache_point_first_content_raises(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    with pytest.raises(UserError, match='`CachePoint` cannot be the first item in an OpenAI user prompt'):
        await Agent(model).run([CachePoint(), 'This should fail.'])

    assert get_mock_chat_completion_kwargs(mock_client) == []


async def test_openai_chat_cache_point_filtered_without_support(allow_model_requests: None):
    """Models without OpenAI explicit-breakpoint support continue to filter out `CachePoint`."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    result = await Agent(model).run(['text before', CachePoint(), 'text after'])

    assert result.output == 'response'
    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'text before'},
                    {'type': 'text', 'text': 'text after'},
                ],
            }
        ]
    )


# ===== Chat Completions: provider and model gating =====


async def test_openai_prompt_cache_options_not_sent_to_unsupported_chat_model(allow_model_requests: None):
    """Omission must hold before the SDK serializes or sends a request."""
    mock_client = AsyncOpenAI(api_key='test-key')
    create = AsyncMock(return_value=chat_completion())
    mock_client.chat.completions.create = create
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    assert result.output == 'response'
    assert create.await_count == 1
    request = create.call_args.kwargs
    assert 'prompt_cache_options' not in request
    assert request['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Stable context.'}, {'type': 'text', 'text': 'Use it.'}]}
    ]


async def test_openai_prompt_cache_options_not_sent_to_openrouter_chat(allow_model_requests: None):
    c = chat.ChatCompletion.model_validate({**chat_completion().model_dump(), 'provider': 'OpenAI'})
    mock_client = AsyncOpenAI(api_key='test-key')
    create = AsyncMock(return_value=c)
    mock_client.chat.completions.create = create
    model = OpenRouterModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run('Hello')

    assert result.output == 'response'
    assert create.await_count == 1
    assert 'prompt_cache_options' not in create.call_args.kwargs


async def test_openrouter_chat_cache_point_dropped_for_openai_models(allow_model_requests: None):
    """OpenRouter Chat only supports automatic caching for OpenAI models: the marker maps to nothing."""
    c = chat.ChatCompletion.model_validate({**chat_completion().model_dump(), 'provider': 'OpenAI'})
    mock_client = AsyncOpenAI(api_key='test-key')
    create = AsyncMock(return_value=c)
    mock_client.chat.completions.create = create
    model = OpenRouterModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))

    await Agent(model).run(['Stable context.', CachePoint(), 'Use it.'])

    request = create.call_args.kwargs
    assert 'prompt_cache_options' not in request
    assert request['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Stable context.'}, {'type': 'text', 'text': 'Use it.'}]}
    ]


async def test_azure_chat_prompt_cache_fields_are_omitted(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=AzureProvider(openai_client=cast('AsyncAzureOpenAI', mock_client)))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert 'prompt_cache_options' not in request
    assert request['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Stable context.'}, {'type': 'text', 'text': 'Use it.'}]}
    ]


async def test_azure_chat_prompt_cache_fields_are_omitted_with_openai_provider(allow_model_requests: None):
    async with AsyncAzureOpenAI(
        azure_endpoint='https://example-resource.openai.azure.com',
        api_version='2026-07-01-preview',
        api_key='test-key',
    ) as client:
        create = AsyncMock(return_value=chat_completion())
        client.chat.completions.create = create
        model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=client))
        settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

        await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    request = create.call_args.kwargs
    assert 'prompt_cache_options' not in request
    assert request['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Stable context.'}, {'type': 'text', 'text': 'Use it.'}]}
    ]


# ===== Responses API: breakpoints and request-level options =====


@pytest.mark.parametrize('provider_name', ['openai', 'openrouter'])
async def test_openai_responses_cache_point_and_options(
    allow_model_requests: None, provider_name: Literal['openai', 'openrouter']
):
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    if provider_name == 'openai':
        model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    else:
        model = OpenAIResponsesModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run(
        ['Stable reference material.', CachePoint(ttl='1h'), 'Use the reference.']
    )

    assert result.output == 'done'
    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Stable reference material.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'input_text', 'text': 'Use the reference.'},
                ],
            }
        ]
    )


@pytest.mark.parametrize('mode', ['implicit', 'explicit'])
async def test_openai_responses_prompt_cache_options_without_marker(
    allow_model_requests: None, mode: Literal['implicit', 'explicit']
):
    """Request-wide cache options are independent of explicit breakpoint markers."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': mode})

    await Agent(model, model_settings=settings).run('No explicit marker.')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': mode}
    assert request['input'] == [{'role': 'user', 'content': 'No explicit marker.'}]


async def test_openai_responses_prompt_cache_options_without_breakpoint_support(allow_model_requests: None):
    """Request-level cache options do not require explicit breakpoint support."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel(
        'custom-model',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(openai_responses_prompt_cache_supported_modes=frozenset({'implicit'})),
    )

    await Agent(
        model,
        model_settings=OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'implicit'}),
    ).run('No explicit marker.')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'implicit'}


async def test_openai_responses_multiple_cache_points(allow_model_requests: None):
    """Each marker attaches to its own preceding block; OpenAI writes up to four explicit breakpoints."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(['Product docs.', CachePoint(), 'Session context.', CachePoint(), 'Question.'])

    assert get_mock_responses_kwargs(mock_client)[0]['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Product docs.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {
                        'type': 'input_text',
                        'text': 'Session context.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'input_text', 'text': 'Question.'},
                ],
            }
        ]
    )


async def test_openai_responses_cache_point_history_prefix_stability(allow_model_requests: None):
    """A serialized history preserves the cacheable prefix and its breakpoint across turns."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion(), responses_completion()])
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    first_result = await agent.run(['Stable context.', CachePoint(), 'First question.'])
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(first_result.all_messages()))
    await agent.run('Follow-up question.', message_history=history)

    first_request, second_request = get_mock_responses_kwargs(mock_client)
    first_input = cast('list[dict[str, Any]]', first_request['input'])
    second_input = cast('list[dict[str, Any]]', second_request['input'])
    assert second_input[0] == first_input[0]
    assert second_input[0] == snapshot(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'input_text',
                    'text': 'Stable context.',
                    'prompt_cache_breakpoint': {'mode': 'explicit'},
                },
                {'type': 'input_text', 'text': 'First question.'},
            ],
        }
    )
    assert second_input[-1] == {'role': 'user', 'content': 'Follow-up question.'}


async def test_openai_responses_image_cache_point(allow_model_requests: None):
    """Pin OpenAI's image-block breakpoint translation."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run([ImageUrl('https://example.com/reference.png'), CachePoint(), 'Describe the reference.'])

    assert get_mock_responses_kwargs(mock_client)[0]['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'input_image',
                        'detail': 'auto',
                        'image_url': 'https://example.com/reference.png',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'input_text', 'text': 'Describe the reference.'},
                ],
            }
        ]
    )


async def test_openai_responses_file_cache_point(allow_model_requests: None):
    """Pin breakpoint translation for the remaining supported Responses content type."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(
        [BinaryContent(b'%PDF-1.4', media_type='application/pdf'), CachePoint(), 'Summarize the reference.']
    )

    request_input = get_mock_responses_kwargs(mock_client)[0]['input']
    content = request_input[0]['content']
    assert isinstance(content, list)
    first_content = cast('dict[str, Any]', content[0])
    assert first_content['type'] == 'input_file'
    assert first_content.get('prompt_cache_breakpoint') == {'mode': 'explicit'}


async def test_openai_responses_cache_point_first_content_raises(allow_model_requests: None):
    mock_client = MockOpenAIResponses.create_mock(response_message([]))
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    with pytest.raises(UserError, match='`CachePoint` cannot be the first item in an OpenAI user prompt'):
        await Agent(model).run([CachePoint(), 'This should fail.'])

    assert get_mock_responses_kwargs(mock_client) == []


async def test_openai_responses_cache_point_filtered_without_support(allow_model_requests: None):
    """Models without Responses breakpoint support continue to filter out `CachePoint`."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion('response'))
    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(openai_client=mock_client))

    result = await Agent(model).run(['text before', CachePoint(), 'text after'])

    assert result.output == 'response'
    assert get_mock_responses_kwargs(mock_client)[0]['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'text before'},
                    {'type': 'input_text', 'text': 'text after'},
                ],
            }
        ]
    )


# ===== Responses API: provider and model gating =====


@pytest.mark.parametrize(
    'options',
    [
        {'mode': 'implicit'},
        {'ttl': '30m'},
    ],
)
async def test_openrouter_responses_unsupported_prompt_cache_options_omitted(
    allow_model_requests: None,
    options: OpenAIPromptCacheOptions,
):
    """OpenRouter accepts request-level options only when `mode='explicit'`."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))

    await Agent(model, model_settings=OpenAIResponsesModelSettings(openai_prompt_cache_options=options)).run(
        'No explicit marker.'
    )

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'prompt_cache_options' not in request


async def test_openrouter_responses_unsupported_options_omitted_but_breakpoint_kept(allow_model_requests: None):
    """Dropping unsupported request-level options does not affect the marker mapping."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'implicit'})

    await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'prompt_cache_options' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Stable context.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'input_text', 'text': 'Use it.'},
                ],
            }
        ]
    )


async def test_openrouter_responses_image_cache_point_raises(allow_model_requests: None):
    """OpenRouter documents breakpoints only on `input_text` blocks."""
    mock_client = MockOpenAIResponses.create_mock(response_message([]))
    model = OpenAIResponsesModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))

    with pytest.raises(UserError, match="cannot follow an OpenAI 'input_image' content block"):
        await Agent(model).run([ImageUrl('https://example.com/reference.png'), CachePoint(), 'Describe it.'])

    assert get_mock_responses_kwargs(mock_client) == []


async def test_openai_responses_prompt_cache_options_not_sent_to_unsupported_model(allow_model_requests: None):
    """Omission must hold before the SDK serializes or sends a request."""
    mock_client = AsyncOpenAI(api_key='test-key')
    create = AsyncMock(return_value=responses_completion())
    mock_client.responses.create = create
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    assert result.output == 'done'
    assert create.await_count == 1
    request = create.call_args.kwargs
    assert 'prompt_cache_options' not in request
    assert request['input'] == [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'Stable context.'},
                {'type': 'input_text', 'text': 'Use it.'},
            ],
        }
    ]


async def test_azure_responses_prompt_cache_fields_are_omitted(allow_model_requests: None):
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel(
        'gpt-5.6-sol', provider=AzureProvider(openai_client=cast('AsyncAzureOpenAI', mock_client))
    )
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'prompt_cache_options' not in request
    assert request['input'] == [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'Stable context.'},
                {'type': 'input_text', 'text': 'Use it.'},
            ],
        }
    ]


async def test_azure_responses_prompt_cache_fields_are_omitted_with_openai_provider(allow_model_requests: None):
    async with AsyncAzureOpenAI(
        azure_endpoint='https://example-resource.openai.azure.com',
        api_version='2026-07-01-preview',
        api_key='test-key',
    ) as client:
        create = AsyncMock(return_value=responses_completion())
        client.responses.create = create
        model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=client))
        settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

        await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    request = create.call_args.kwargs
    assert 'prompt_cache_options' not in request
    assert request['input'] == [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'Stable context.'},
                {'type': 'input_text', 'text': 'Use it.'},
            ],
        }
    ]


# ===== Usage mapping: cache write tokens =====


async def test_openai_chat_stream_maps_cache_write_usage(allow_model_requests: None):
    """A synthetic usage chunk isolates the internal usage-field mapping."""
    response_chunk = chat.ChatCompletionChunk(
        id='123',
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content='world', role='assistant'), finish_reason='stop')],
        created=1704067200,
        model='gpt-5.6-sol',
        object='chat.completion.chunk',
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=100,
            total_tokens=110,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=20, cache_write_tokens=30),
        ),
    )
    mock_client = MockOpenAI.create_mock_stream([response_chunk])
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    async with Agent(model).run_stream('Hello') as result:
        assert await result.get_output() == 'world'

    assert result.usage == RunUsage(
        requests=1,
        input_tokens=100,
        cache_write_tokens=30,
        cache_read_tokens=20,
        output_tokens=10,
    )


async def test_openai_responses_maps_cache_write_usage(allow_model_requests: None):
    """A synthetic response isolates the internal usage-field mapping."""
    mock_client = MockOpenAIResponses.create_mock(
        responses_completion(
            '4',
            usage=ResponseUsage(
                input_tokens=2006,
                input_tokens_details=InputTokensDetails(cached_tokens=1920, cache_write_tokens=64),
                output_tokens=300,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
                total_tokens=2306,
            ),
        )
    )
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    result = await Agent(model=model).run('What is 2+2?')

    assert result.usage == RunUsage(
        requests=1,
        input_tokens=2006,
        cache_write_tokens=64,
        cache_read_tokens=1920,
        output_tokens=300,
        details={'reasoning_tokens': 10},
    )


async def test_openai_responses_stream_maps_cache_write_usage(allow_model_requests: None):
    """Synthetic stream events isolate the internal usage-field mapping."""
    base_response = resp.Response(
        id='resp_001',
        model='gpt-5.6-sol',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    response_usage = ResponseUsage(
        input_tokens=2006,
        input_tokens_details=InputTokensDetails(cached_tokens=1920, cache_write_tokens=64),
        output_tokens=300,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
        total_tokens=2306,
    )
    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base_response, type='response.created', sequence_number=0),
        resp.ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id='msg_001',
                content=[],
                role='assistant',
                status='in_progress',
                type='message',
            ),
            output_index=0,
            type='response.output_item.added',
            sequence_number=1,
        ),
        resp.ResponseTextDeltaEvent(
            item_id='msg_001',
            output_index=0,
            content_index=0,
            delta='done',
            logprobs=[],
            type='response.output_text.delta',
            sequence_number=2,
        ),
        resp.ResponseCompletedEvent(
            response=base_response.model_copy(update={'status': 'completed', 'usage': response_usage}),
            type='response.completed',
            sequence_number=3,
        ),
    ]
    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    async with Agent(model).run_stream('Solve this.') as result:
        assert await result.get_output() == 'done'

    assert result.usage == RunUsage(
        requests=1,
        input_tokens=2006,
        cache_write_tokens=64,
        cache_read_tokens=1920,
        output_tokens=300,
        details={'reasoning_tokens': 10},
    )
