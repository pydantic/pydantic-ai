"""Tests for OpenAI GPT-5.6 explicit prompt caching on the Chat Completions and Responses APIs.

Covers the `openai_prompt_cache_options` setting, `CachePoint` to `prompt_cache_breakpoint`
mapping, the `openai_supports_prompt_cache_breakpoints` profile gate, cache-write usage
mapping, and prompt cache diagnostics.

Most adapter-level tests here intentionally use mocked SDK clients rather than VCR
recordings: they pin exact SDK request kwargs, omission of unsupported fields, and
pre-request guards where no request may be sent at all. Recordings cannot reliably assert
omitted kwargs or a request that is never made, and cassette matchers are not always
sensitive to the request body. The `_e2e` tests at the end record the accept path against
the real APIs.
"""

from __future__ import annotations as _annotations

import json
import os
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import httpx2
import pytest
from cassetter import Cassette
from pydantic import BaseModel, JsonValue

from pydantic_ai import Agent, BinaryContent, CachePoint, ImageUrl, PromptedOutput, RunContext
from pydantic_ai._framing import lookalike_note_announcement
from pydantic_ai.computer_use import COMPUTER_KIND
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentInstructionSource,
    BinaryImage,
    CompactionPart,
    InstructionId,
    InstructionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from .._inline_snapshot import snapshot
from ..cassette_utils import request_json
from ..computer_fakes import FAKE_DISPLAY_SIZE, FAKE_SCREENSHOT
from ..conftest import IsStr, RequestCapture, try_import
from .mock_openai import (
    MockOpenAI,
    MockOpenAIResponses,
    completion_message,
    get_mock_chat_completion_kwargs,
    get_mock_responses_kwargs,
    response_message,
)

with try_import() as imports_successful:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, omit
    from openai.types import chat, responses as resp
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
    from openai.types.responses.response_compaction_item import ResponseCompactionItem
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.native_tools import ComputerTool
    from pydantic_ai.native_tools._tool_search import ToolSearchTool
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


class Answer(BaseModel):
    answer: str


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


@pytest.mark.parametrize('provider_name', ['openai', 'openrouter'])
async def test_openai_chat_cache_point_and_options(
    allow_model_requests: None, provider_name: Literal['openai', 'openrouter']
):
    mock_client = MockOpenAI.create_mock(chat_completion())
    if provider_name == 'openai':
        model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    else:
        model = OpenAIChatModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))
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
    """Each marker attaches to its own block; OpenAI writes the latest three (implicit mode) or four (explicit)."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(['Product docs.', CachePoint(), 'Session context.', CachePoint(), 'Question.'])

    assert 'prompt_cache_options' not in get_mock_chat_completion_kwargs(mock_client)[0]
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


async def test_openai_chat_adjacent_cache_points_collapse(allow_model_requests: None):
    """Back-to-back markers idempotently mark the same block: one breakpoint, no error."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(['Product docs.', CachePoint(), CachePoint(), 'Question.'])

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

    with pytest.raises(UserError, match='CachePoint cannot be the first content in a user message'):
        await Agent(model).run([CachePoint(), 'This should fail.'])

    assert get_mock_chat_completion_kwargs(mock_client) == []


def _history_ending_in(*tail: ModelMessage) -> list[ModelMessage]:
    """A tool call answered by a tool return, then `tail`: the shape a capability's tail reminder follows."""
    return [
        ModelRequest(parts=[UserPromptPart('Look it up.')]),
        ModelResponse(parts=[ToolCallPart('lookup', {'n': 1}, tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('lookup', 'result 1', tool_call_id='call_1')]),
        *tail,
    ]


async def test_openai_chat_leading_cache_point_attaches_to_previous_message(allow_model_requests: None):
    """A `CachePoint` opening a user message after a tool result marks the tool result instead of raising.

    That is where a tail reminder (such as `Planning`'s) puts its breakpoint once a tool loop
    is underway; the request used to fail with `CachePoint cannot be the first content in a user message`.
    """
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await model.request(
        _history_ending_in(ModelRequest(parts=[UserPromptPart([CachePoint(), 'Stay focused.'])])),
        None,
        ModelRequestParameters(),
    )

    messages = get_mock_chat_completion_kwargs(mock_client)[0]['messages']
    assert messages[2:] == snapshot(
        [
            {
                'role': 'tool',
                'tool_call_id': 'call_1',
                'content': [{'type': 'text', 'text': 'result 1', 'prompt_cache_breakpoint': {'mode': 'explicit'}}],
            },
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Stay focused.'}]},
        ]
    )


async def test_openai_chat_leading_cache_point_skips_messages_without_content(allow_model_requests: None):
    """An assistant turn holding only tool calls can't carry the marker, so the one before it does."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await model.request(
        [
            ModelRequest(parts=[UserPromptPart(['Look it up.'])]),
            ModelResponse(parts=[ToolCallPart('lookup', {'n': 1}, tool_call_id='call_1')]),
            ModelRequest(parts=[UserPromptPart([CachePoint()])]),
        ],
        None,
        ModelRequestParameters(),
    )

    messages = get_mock_chat_completion_kwargs(mock_client)[0]['messages']
    assert [message['role'] for message in messages] == ['user', 'assistant']
    assert messages[0]['content'] == snapshot(
        [{'type': 'text', 'text': 'Look it up.', 'prompt_cache_breakpoint': {'mode': 'explicit'}}]
    )


async def test_openai_chat_leading_cache_point_with_merged_system_messages(allow_model_requests: None):
    """A leading `CachePoint` after system prompts marks the merged system message, which must still be joined as text."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(
            openai_supports_prompt_cache_breakpoints=True, openai_chat_supports_multiple_system_messages=False
        ),
    )

    await model.request(
        [
            ModelRequest(
                parts=[SystemPromptPart('First.'), SystemPromptPart('Second.'), UserPromptPart([CachePoint(), 'Hi.'])],
                instructions='Be brief.',
            )
        ],
        None,
        ModelRequestParameters(),
    )

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': """\
First.

Second.

Be brief.\
""",
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': [{'text': 'Hi.', 'type': 'text'}]},
        ]
    )


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


# ===== Chat Completions: model gating =====


async def test_openai_chat_prompt_cache_options_sent_for_any_model(allow_model_requests: None):
    """Like the sibling `openai_prompt_cache_*` settings, the options are forwarded as-is for any
    model, while `CachePoint` markers stay gated by the model profile."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    assert result.output == 'response'
    request = get_mock_chat_completion_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Stable context.'}, {'type': 'text', 'text': 'Use it.'}]}
    ]


async def test_openrouter_unified_cache_adds_cache_control(allow_model_requests: None):
    """The unified `cache` setting flows through `OpenRouterModel.prepare_request` into a
    `cache_control` breakpoint on the system instructions for a supporting downstream provider."""
    c = chat.ChatCompletion.model_validate({**chat_completion().model_dump(), 'provider': 'Anthropic'})
    mock_client = AsyncOpenAI(api_key='test-key')
    create = AsyncMock(return_value=c)
    mock_client.chat.completions.create = create
    model = OpenRouterModel('anthropic/claude-sonnet-4.5', provider=OpenRouterProvider(openai_client=mock_client))

    await Agent(
        model, instructions='Stable instructions.', model_settings={'cache': '1h', 'framing_announcement': 'always'}
    ).run('Hi')

    request = create.call_args.kwargs
    # Claude Sonnet 4.5 takes no mid-conversation system messages, so with `'always'` the instructions start with the
    # announcement; it never changes, so the breakpoint after the instructions covers it too.
    announcement, system_message = request['messages'][:2]
    assert system_message == {
        'role': 'system',
        'content': [
            {'type': 'text', 'text': 'Stable instructions.', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}
        ],
    }
    assert announcement == {
        'role': 'system',
        'content': lookalike_note_announcement(system_role=False),
    }


async def test_openrouter_chat_cache_point_dropped_for_openai_models(allow_model_requests: None):
    """`OpenRouterModel` translates `CachePoint` into `cache_control`, which is dropped for
    providers without `cache_control` support; the OpenAI breakpoint mapping never applies."""
    c = chat.ChatCompletion.model_validate({**chat_completion().model_dump(), 'provider': 'OpenAI'})
    mock_client = AsyncOpenAI(api_key='test-key')
    create = AsyncMock(return_value=c)
    mock_client.chat.completions.create = create
    model = OpenRouterModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))

    await Agent(model).run(['Stable context.', CachePoint(), 'Use it.'])

    request = create.call_args.kwargs
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


async def test_openai_responses_multiple_cache_points(allow_model_requests: None):
    """Each marker attaches to its own block; OpenAI writes the latest three (implicit mode) or four (explicit)."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model).run(['Product docs.', CachePoint(), 'Session context.', CachePoint(), 'Question.'])

    assert 'prompt_cache_options' not in get_mock_responses_kwargs(mock_client)[0]
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

    with pytest.raises(UserError, match='CachePoint cannot be the first content in a user message'):
        await Agent(model).run([CachePoint(), 'This should fail.'])

    assert get_mock_responses_kwargs(mock_client) == []


async def test_openai_responses_leading_cache_point_attaches_to_previous_item(allow_model_requests: None):
    """The Responses counterpart: the breakpoint lands on the `function_call_output` before the reminder."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await model.request(
        _history_ending_in(
            ModelRequest(parts=[UserPromptPart([CachePoint(), 'Stay focused.'])]),
            ModelResponse(parts=[ToolCallPart('lookup', {'n': 2}, tool_call_id='call_2')]),
            ModelRequest(
                parts=[
                    ToolReturnPart('lookup', 'result 2', tool_call_id='call_2'),
                    UserPromptPart([CachePoint()]),
                ]
            ),
        ),
        None,
        ModelRequestParameters(),
    )

    request_input = get_mock_responses_kwargs(mock_client)[0]['input']
    assert request_input[2:] == snapshot(
        [
            {
                'type': 'function_call_output',
                'call_id': 'call_1',
                'output': [{'type': 'input_text', 'text': 'result 1', 'prompt_cache_breakpoint': {'mode': 'explicit'}}],
            },
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Stay focused.'}]},
            {'name': 'lookup', 'arguments': '{"n":2}', 'call_id': 'call_2', 'type': 'function_call'},
            {
                'type': 'function_call_output',
                'call_id': 'call_2',
                'output': [{'type': 'input_text', 'text': 'result 2', 'prompt_cache_breakpoint': {'mode': 'explicit'}}],
            },
        ]
    )


async def test_openai_responses_leading_cache_point_skips_assistant_output(allow_model_requests: None):
    """Assistant output (`output_text`) can't carry a breakpoint, so a leading `CachePoint` after it marks the user input before."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await model.request(
        [
            ModelRequest(parts=[UserPromptPart('Question.')]),
            ModelResponse(parts=[TextPart('Sent as output.', id='msg_1')], provider_name='openai'),
            ModelResponse(parts=[TextPart('Sent as a string.')]),
            ModelRequest(parts=[UserPromptPart([CachePoint(), 'Follow-up.'])]),
        ],
        None,
        ModelRequestParameters(),
    )

    assert get_mock_responses_kwargs(mock_client)[0]['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'Question.', 'prompt_cache_breakpoint': {'mode': 'explicit'}}
                ],
            },
            {
                'role': 'assistant',
                'id': 'msg_1',
                'content': [{'text': 'Sent as output.', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
            {'role': 'assistant', 'content': 'Sent as a string.'},
            {'role': 'user', 'content': [{'text': 'Follow-up.', 'type': 'input_text'}]},
        ]
    )


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


# ===== Responses API: model gating =====


async def test_openai_responses_prompt_cache_options_sent_for_any_model(allow_model_requests: None):
    """Like the sibling `openai_prompt_cache_*` settings, the options are forwarded as-is for any
    model, while `CachePoint` markers stay gated by the model profile."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})

    result = await Agent(model, model_settings=settings).run(['Stable context.', CachePoint(), 'Use it.'])

    assert result.output == 'done'
    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['input'] == [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'Stable context.'},
                {'type': 'input_text', 'text': 'Use it.'},
            ],
        }
    ]


# ===== Instruction caching =====


async def test_openai_chat_cache_instructions(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)

    result = await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    assert result.output == 'response'
    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_after_last_static(allow_model_requests: None):
    """Dynamic instructions are sorted last and stay outside the cached prefix."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Support policies.', model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'system', 'content': 'Today is 2026-08-18.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_single_breakpoint_before_appending_blocks(allow_model_requests: None):
    """An `on_change='append'` block gets no breakpoint of its own: only the one after the static instructions.

    OpenAI keeps a request's last four breakpoints, so a second one would push out the static instructions' one
    first, and that's the one a new conversation reads even when its append blocks differ.
    """
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Support policies.', model_settings=settings)

    @agent.instructions(name='memory', on_change='append')
    def memory() -> str:
        return 'The customer prefers email.'

    await agent.run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {'type': 'text', 'text': 'Support policies.', 'prompt_cache_breakpoint': {'mode': 'explicit'}}
                ],
            },
            {
                'role': 'system',
                'content': """\
<pyai:context id="agent:memory">
The customer prefers email.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_all_dynamic_falls_back_to_system_prompt(allow_model_requests: None):
    """With no static instruction to end the prefix, the boundary is the last system prompt."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, system_prompt='Support policies.', model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'system', 'content': 'Today is 2026-08-18.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_all_dynamic_without_system_prompt(allow_model_requests: None):
    """Nothing in the prefix is stable, so no breakpoint is added."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'Today is 2026-08-18.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_ignored_without_support(allow_model_requests: None):
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'Support policies.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_skipped_for_dynamic_system_prompt(allow_model_requests: None):
    """A dynamic system prompt renders in the cached prefix, so marking it would miss the cache every
    run; the breakpoint is skipped and nothing is marked."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Follow the rules.', model_settings=settings)

    @agent.system_prompt(dynamic=True)
    def current_policies() -> str:
        return 'Support policies.'

    await agent.run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {'role': 'system', 'content': 'Support policies.'},
            {'role': 'system', 'content': 'Follow the rules.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_responses_cache_instructions(allow_model_requests: None):
    """The top-level `instructions` field cannot carry a breakpoint, so instructions move into `input`."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)

    result = await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    assert result.output == 'done'
    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_responses_cache_instructions_after_last_static(allow_model_requests: None):
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Support policies.', model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'system', 'content': 'Today is 2026-08-18.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_responses_cache_instructions_all_dynamic_falls_back_to_system_prompt(
    allow_model_requests: None,
):
    """With no static instruction to end the prefix, the boundary is the last system prompt."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, system_prompt='Support policies.', model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'system', 'content': 'Today is 2026-08-18.'},
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_responses_cache_instructions_all_dynamic_without_system_prompt(allow_model_requests: None):
    """With no breakpoint to place, the instructions stay in the top-level field."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Today is 2026-08-18.'
    assert request['input'] == [{'role': 'user', 'content': 'Where is order 1234?'}]


async def test_openai_responses_cache_instructions_ignored_without_support(allow_model_requests: None):
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Support policies.'
    assert request['input'] == [{'role': 'user', 'content': 'Where is order 1234?'}]


async def test_openai_responses_cache_instructions_skipped_for_dynamic_system_prompt(allow_model_requests: None):
    """A dynamic system prompt renders in the cached prefix, so the instructions stay in the top-level
    field and no breakpoint is added."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Follow the rules.', model_settings=settings)

    @agent.system_prompt(dynamic=True)
    def current_policies() -> str:
        return 'Support policies.'

    await agent.run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Follow the rules.'
    assert request['input'] == [
        {'role': 'system', 'content': 'Support policies.'},
        {'role': 'user', 'content': 'Where is order 1234?'},
    ]


@pytest.mark.parametrize('conversation_id', ['conv_123', 'auto'])
async def test_openai_responses_cache_instructions_skipped_with_conversation_id(
    allow_model_requests: None, conversation_id: str
):
    """A conversation persists its input messages, so instructions stay in the top-level field."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True, openai_conversation_id=conversation_id)

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Support policies.'
    assert request['input'] == [{'role': 'user', 'content': 'Where is order 1234?'}]


async def test_openai_responses_cache_instructions_with_prompted_output(allow_model_requests: None):
    """Prompted output also moves instructions into `input`; they must not be sent twice.

    Its format instructions are static, so the boundary moves past them to the end of the prefix.
    """
    mock_client = MockOpenAIResponses.create_mock(responses_completion('{"answer": "shipped"}'))
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Support policies.', model_settings=settings, output_type=PromptedOutput(Answer))

    await agent.run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {'role': 'system', 'content': 'Support policies.'},
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': IsStr(regex=r'(?s).*JSON.*'),
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_without_instruction_parts(allow_model_requests: None):
    """With only a system prompt and no instructions, the boundary is the system prompt."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)

    await Agent(model, system_prompt='Support policies.', model_settings=settings).run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_responses_cache_instructions_without_instruction_parts(allow_model_requests: None):
    """The system prompt is already an input message, so nothing is relocated."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)

    await Agent(model, system_prompt='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_responses_cache_instructions_with_leading_cache_point_on_system_prompt(
    allow_model_requests: None,
):
    """A leading `CachePoint` has already put its breakpoint on the system prompt the instruction breakpoint targets."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)

    await Agent(model, system_prompt='Support policies.', model_settings=settings).run(
        [CachePoint(), 'Where is order 1234?']
    )

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': [{'text': 'Where is order 1234?', 'type': 'input_text'}]},
        ]
    )


async def test_openai_chat_cache_instructions_with_developer_role(allow_model_requests: None):
    """OpenAI's own example marks reusable instructions in a developer message."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(
            openai_system_prompt_role='developer', openai_supports_prompt_cache_breakpoints=True
        ),
    )
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'developer',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_skipped_when_system_messages_are_merged(allow_model_requests: None):
    """Merging collapses the boundary into one block, so no breakpoint can express it."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(
            openai_chat_supports_multiple_system_messages=False, openai_supports_prompt_cache_breakpoints=True
        ),
    )
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, system_prompt='Support policies.', instructions='Answer briefly.', model_settings=settings)

    await agent.run('Where is order 1234?')

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': """\
Support policies.

Answer briefly.\
""",
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_cache_instructions_skipped_for_user_system_prompt_role(allow_model_requests: None):
    """A `'user'` system prompt role can't be told apart from a real user turn, so the multi-modal
    user content must be left alone."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(openai_system_prompt_role='user', openai_supports_prompt_cache_breakpoints=True),
    )
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)
    agent = Agent(model, model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run(['Look at this', ImageUrl(url='https://example.com/image.png')])

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'Look at this', 'type': 'text'},
                    {'image_url': {'url': 'https://example.com/image.png'}, 'type': 'image_url'},
                ],
            },
            {'role': 'user', 'content': 'Today is 2026-08-18.'},
        ]
    )


async def test_openai_responses_cache_instructions_skipped_with_previous_response_id(allow_model_requests: None):
    """A stored response keeps its input, so relocated instructions would be replayed to the next
    request in the chain on top of its own. With `openai_previous_response_id` set, the instructions
    stay in the top-level field, which OpenAI doesn't replay, on every request in the chain."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True, openai_previous_response_id='auto')
    agent = Agent(model, instructions='Support policies.', model_settings=settings)

    first = await agent.run('Where is order 1234?')
    await agent.run('And order 5678?', message_history=first.all_messages())

    requests = get_mock_responses_kwargs(mock_client)
    assert 'previous_response_id' not in requests[0]
    assert requests[1]['previous_response_id'] == '123'
    assert [request['instructions'] for request in requests] == ['Support policies.', 'Support policies.']
    assert [request['input'] for request in requests] == [
        [{'role': 'user', 'content': 'Where is order 1234?'}],
        [{'role': 'user', 'content': 'And order 5678?'}],
    ]


async def test_openai_responses_cache_instructions_handoff_to_chained_agent(allow_model_requests: None):
    """A later agent that continues the chain must run under its own instructions only, whether or
    not it caches them: the earlier agent kept its instructions out of the stored input."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    refunds = Agent(
        model,
        instructions='Refunds under $50 are automatic.',
        model_settings=OpenAIResponsesModelSettings(openai_cache_instructions=True, openai_previous_response_id='auto'),
    )
    first = await refunds.run('Order 1234 costs $20. Can I get a refund?')

    collections = Agent(
        model,
        instructions='Never offer a refund. Demand payment.',
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'),
    )
    await collections.run('What should we do next?', message_history=first.all_messages())

    requests = get_mock_responses_kwargs(mock_client)
    assert requests[0]['instructions'] == 'Refunds under $50 are automatic.'
    assert requests[0]['input'] == [{'role': 'user', 'content': 'Order 1234 costs $20. Can I get a refund?'}]
    assert requests[1]['previous_response_id'] == '123'
    assert requests[1]['instructions'] == 'Never offer a refund. Demand payment.'
    assert requests[1]['input'] == [{'role': 'user', 'content': 'What should we do next?'}]


async def test_openai_responses_cache_instructions_skipped_with_explicit_previous_response_id(
    allow_model_requests: None,
):
    """An explicit `openai_previous_response_id` continues server-side state too, so the instructions
    stay in the top-level field."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True, openai_previous_response_id='resp_abc')

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['previous_response_id'] == 'resp_abc'
    assert request['instructions'] == 'Support policies.'
    assert request['input'] == [{'role': 'user', 'content': 'Where is order 1234?'}]


async def test_openai_responses_cache_instructions_not_relocated_after_compaction(allow_model_requests: None):
    """The compaction item retains the leading input messages, so relocating would send them twice."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Where is order 1234?'),
        ModelResponse(
            parts=[
                CompactionPart(
                    content='summary',
                    provider_name='openai',
                    provider_details={'encrypted_content': 'encrypted'},
                )
            ],
            provider_name='openai',
        ),
    ]

    await Agent(model, instructions='Support policies.', model_settings=settings).run(
        'And order 5678?', message_history=messages
    )

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Support policies.'
    assert request['input'] == snapshot(
        [
            {'id': None, 'encrypted_content': 'encrypted', 'type': 'compaction'},
            {'role': 'user', 'content': 'And order 5678?'},
        ]
    )


@pytest.mark.parametrize(
    'api, expected',
    [
        pytest.param(
            'chat',
            snapshot(
                [
                    {
                        'role': 'system',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Support policies.',
                                'prompt_cache_breakpoint': {'mode': 'explicit'},
                            }
                        ],
                    },
                    {'role': 'system', 'content': 'Today is 2026-08-18.'},
                    {'role': 'system', 'content': 'Refund rules.'},
                    {'role': 'user', 'content': 'Where is order 1234?'},
                ]
            ),
            id='chat',
        ),
        pytest.param(
            'responses',
            snapshot(
                [
                    {
                        'role': 'system',
                        'content': [
                            {
                                'type': 'input_text',
                                'text': 'Support policies.',
                                'prompt_cache_breakpoint': {'mode': 'explicit'},
                            }
                        ],
                    },
                    {'role': 'system', 'content': 'Today is 2026-08-18.'},
                    {'role': 'system', 'content': 'Refund rules.'},
                    {'role': 'user', 'content': 'Where is order 1234?'},
                ]
            ),
            id='responses',
        ),
    ],
)
async def test_openai_cache_instructions_unsorted_parts_keep_dynamic_out_of_prefix(
    allow_model_requests: None, api: Literal['chat', 'responses'], expected: list[dict[str, Any]]
):
    """The breakpoint goes after the leading static parts, never on a dynamic part or past one.

    Calls `model.request` directly because agent runs always sort static parts first.
    """
    instruction_parts = [
        InstructionPart(content='Support policies.'),
        InstructionPart(content='Today is 2026-08-18.', dynamic=True),
        InstructionPart(content='Refund rules.'),
    ]
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Where is order 1234?')]
    parameters = ModelRequestParameters(instruction_parts=instruction_parts)
    if api == 'chat':
        chat_client = MockOpenAI.create_mock(chat_completion())
        chat_model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=chat_client))
        await chat_model.request(messages, OpenAIChatModelSettings(openai_cache_instructions=True), parameters)
        sent = get_mock_chat_completion_kwargs(chat_client)[0]['messages']
    else:
        responses_client = MockOpenAIResponses.create_mock(responses_completion())
        responses_model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=responses_client))
        await responses_model.request(
            messages, OpenAIResponsesModelSettings(openai_cache_instructions=True), parameters
        )
        sent = get_mock_responses_kwargs(responses_client)[0]['input']

    assert sent == expected


@pytest.mark.parametrize(
    'has_instructions, expected',
    [
        pytest.param(
            True,
            snapshot(
                [
                    {
                        'role': 'developer',
                        'content': [
                            {
                                'type': 'input_text',
                                'text': 'Support policies.',
                                'prompt_cache_breakpoint': {'mode': 'explicit'},
                            }
                        ],
                    },
                    {
                        'type': 'additional_tools',
                        'role': 'developer',
                        'tools': [
                            {
                                'name': 'lookup_refund_policy',
                                'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                                'type': 'function',
                                'description': None,
                                'strict': False,
                            }
                        ],
                    },
                    {'role': 'user', 'content': 'Where is order 1234?'},
                ]
            ),
            id='with-instructions',
        ),
        pytest.param(
            False,
            snapshot(
                [
                    {
                        'type': 'additional_tools',
                        'role': 'developer',
                        'tools': [
                            {
                                'name': 'lookup_refund_policy',
                                'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                                'type': 'function',
                                'description': None,
                                'strict': False,
                            }
                        ],
                    },
                    {'role': 'user', 'content': 'Where is order 1234?'},
                ]
            ),
            id='without-instructions',
        ),
    ],
)
async def test_openai_responses_cache_instructions_with_leading_tool_reveal(
    allow_model_requests: None, has_instructions: bool, expected: list[dict[str, Any]]
):
    """An `additional_tools` item shares the `'developer'` role but is not a system prompt, so the
    instructions go ahead of it, and without instructions it's left alone instead of being marked.

    Calls `model.request` directly because the item only leads the input when a tool-availability
    delta opens the history, which an agent run doesn't produce on its first request.
    """
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(
            openai_system_prompt_role='developer', openai_supports_prompt_cache_breakpoints=True
        ),
    )
    tool = ToolDefinition(
        name='lookup_refund_policy',
        parameters_json_schema={'type': 'object', 'properties': {}},
        defer_loading=True,
        with_native=ToolSearchTool.kind,
    )

    await model.request(
        [
            ModelRequest(
                parts=[ToolAvailabilityDeltaPart(tools_added=[tool.name]), UserPromptPart('Where is order 1234?')]
            )
        ],
        OpenAIResponsesModelSettings(openai_cache_instructions=True),
        ModelRequestParameters(
            function_tools=[tool],
            native_tools=[ToolSearchTool(optional=True)],
            instruction_parts=[InstructionPart(content='Support policies.')] if has_instructions else None,
        ),
    )

    assert get_mock_responses_kwargs(mock_client)[0]['input'] == expected


async def test_openai_responses_cache_instructions_skipped_for_user_system_prompt_role(allow_model_requests: None):
    """A `'user'` system prompt role can't be told apart from a real user turn, so the instructions
    stay in the top-level field and the user content is left alone."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(openai_system_prompt_role='user', openai_supports_prompt_cache_breakpoints=True),
    )
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, model_settings=settings)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run(['Look at this', ImageUrl(url='https://example.com/image.png')])

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Today is 2026-08-18.'
    assert request['input'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'Look at this', 'type': 'input_text'},
                    {'image_url': 'https://example.com/image.png', 'type': 'input_image', 'detail': 'auto'},
                ],
            }
        ]
    )


async def test_openai_responses_cache_instructions_relocated_once_per_request_in_tool_loop(
    allow_model_requests: None,
):
    """Each request in a run rebuilds `input` from the full history, so every request must carry
    exactly one relocated copy of the instructions."""
    tool_call = resp.ResponseFunctionToolCall(
        id='fc_1',
        call_id='call_1',
        name='get_order_status',
        arguments='{"order_id": "1234"}',
        type='function_call',
        status='completed',
    )
    mock_client = MockOpenAIResponses.create_mock([response_message([tool_call]), responses_completion()])
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, instructions='Support policies.', model_settings=settings)

    @agent.tool_plain
    def get_order_status(order_id: str) -> str:
        return 'shipped'

    await agent.run('Where is order 1234?')

    requests = get_mock_responses_kwargs(mock_client)
    assert len(requests) == 2
    assert all('instructions' not in request for request in requests)
    assert [request['input'] for request in requests] == snapshot(
        [
            [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': 'Support policies.',
                            'prompt_cache_breakpoint': {'mode': 'explicit'},
                        }
                    ],
                },
                {'role': 'user', 'content': 'Where is order 1234?'},
            ],
            [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': 'Support policies.',
                            'prompt_cache_breakpoint': {'mode': 'explicit'},
                        }
                    ],
                },
                {'role': 'user', 'content': 'Where is order 1234?'},
                {
                    'name': 'get_order_status',
                    'arguments': '{"order_id":"1234"}',
                    'call_id': 'call_1',
                    'type': 'function_call',
                    'id': 'fc_1',
                },
                {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'shipped'},
            ],
        ]
    )


# ===== GPT-6.2: stable instructions as leading input messages =====


def _support_agent(model: OpenAIResponsesModel, settings: OpenAIResponsesModelSettings | None = None) -> Agent[str]:
    """Static instructions, an `on_change='append'` block and a dynamic instruction that rewrites the prefix."""
    agent = Agent(model, instructions='Support policies.', deps_type=str, model_settings=settings)

    @agent.instructions(name='queue', on_change='append')
    def queue(ctx: RunContext[str]) -> str:
        return f'Queue: {ctx.deps}.'

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    return agent


@pytest.mark.parametrize(
    'model_name, expected',
    [
        pytest.param(
            'gpt-6.2-sol',
            snapshot(
                {
                    'instructions': 'Today is 2026-08-18.',
                    'input': [
                        {'role': 'system', 'content': 'Support policies.'},
                        {
                            'role': 'system',
                            'content': """\
<pyai:context id="agent:queue">
Queue: open.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                        },
                        {'role': 'user', 'content': 'Where is order 1234?'},
                    ],
                }
            ),
            id='gpt-6.2-sol',
        ),
        pytest.param(
            'gpt-6.1-sol',
            snapshot(
                {
                    'instructions': """\
Support policies.

<pyai:context id="agent:queue">
Queue: open.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.

Today is 2026-08-18.\
""",
                    'input': [{'role': 'user', 'content': 'Where is order 1234?'}],
                }
            ),
            id='gpt-6.1-sol',
        ),
    ],
)
async def test_openai_responses_stable_instructions_as_input(
    allow_model_requests: None, model_name: str, expected: dict[str, Any]
):
    """GPT-6.2 puts the top-level `instructions` after `input` and never caches them, so the static instructions
    and the append blocks lead the input instead, and the instruction that rewrites the prefix stays in the field.
    Earlier models keep all of them in the field, ahead of the input."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=mock_client))

    await _support_agent(model).run('Where is order 1234?', deps='open')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert {'instructions': request.get('instructions'), 'input': request['input']} == expected


async def _run_rewrite_change(model: Model, stable: str = 'Support policies.') -> list[Any]:
    """The three runs where an append block changes, and then only a rewrite-mode instruction does."""
    agent = Agent(model, instructions=stable, deps_type=str)
    date = 'D1'

    @agent.instructions(name='queue', on_change='append')
    def queue(ctx: RunContext[str]) -> str:
        return f'Queue: {ctx.deps}.'

    @agent.instructions
    def current_date() -> str:
        return f'Today is {date}.'

    first = await agent.run('Which shelf holds entry 0042? Reply with just the number.', deps='A')
    second = await agent.run(
        'Which aisle holds entry 0101? Reply with just the number.', deps='B', message_history=first.all_messages()
    )
    date = 'D2'
    third = await agent.run(
        'Which volume is entry 0007? Reply with just the number.', deps='B', message_history=second.all_messages()
    )
    return [first, second, third]


@pytest.mark.parametrize('model_name', ['gpt-6.2-sol', 'gpt-6.2-astra'])
async def test_openai_responses_rewrite_change_keeps_leading_instructions(allow_model_requests: None, model_name: str):
    """On GPT-6.2 a rewrite-mode change only moves the top-level field, after the input."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=mock_client))

    await _run_rewrite_change(model)

    _, second, third = get_mock_responses_kwargs(mock_client)
    assert third['input'][: len(second['input'])] == second['input']
    assert 'Queue: A.' in third['input'][1]['content']
    assert third['instructions'] == 'Today is D2.'


@pytest.mark.parametrize('case', ['opt-out', 'gpt-6.1-sol', 'chat', 'fallback-model'])
async def test_openai_rewrite_change_opens_a_window_on_other_models(allow_model_requests: None, case: str):
    """Only a GPT-6.2 Responses model with the flag on keeps its window: the rest fold the change into the prefix."""
    if case == 'chat':
        chat_client = MockOpenAI.create_mock(chat_completion())
        await _run_rewrite_change(OpenAIChatModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=chat_client)))
        third = get_mock_chat_completion_kwargs(chat_client)[2]
    else:
        mock_client = MockOpenAIResponses.create_mock(responses_completion())
        provider = OpenAIProvider(openai_client=mock_client)
        opt_out = OpenAIModelProfile(openai_responses_stable_instructions_as_input=False)
        model: Model = {
            'opt-out': lambda: OpenAIResponsesModel('gpt-6.2-sol', provider=provider, profile=opt_out),
            'gpt-6.1-sol': lambda: OpenAIResponsesModel('gpt-6.1-sol', provider=provider),
            'fallback-model': lambda: FallbackModel(OpenAIResponsesModel('gpt-6.2-sol', provider=provider)),
        }[case]()
        await _run_rewrite_change(model)
        third = get_mock_responses_kwargs(mock_client)[2]

    sent = json.dumps(third, default=str)
    assert 'Queue: B.' in sent
    assert 'Queue: A.' not in sent


async def test_openai_responses_stable_instructions_as_input_with_cache_instructions(allow_model_requests: None):
    """With `openai_cache_instructions` on GPT-6.2, the breakpoint goes after the static instructions as usual, and
    the instruction that rewrites the prefix still stays in the top-level field, where its changes leave the cached
    conversation intact."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)

    await _support_agent(model, settings).run('Where is order 1234?', deps='open')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert {'instructions': request.get('instructions'), 'input': request['input']} == snapshot(
        {
            'instructions': 'Today is 2026-08-18.',
            'input': [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': 'Support policies.',
                            'prompt_cache_breakpoint': {'mode': 'explicit'},
                        }
                    ],
                },
                {
                    'role': 'system',
                    'content': """\
<pyai:context id="agent:queue">
Queue: open.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                },
                {'role': 'user', 'content': 'Where is order 1234?'},
            ],
        }
    )


async def test_openai_responses_stable_instructions_as_input_without_stable_instructions(allow_model_requests: None):
    """With only an instruction that rewrites the prefix, there's nothing to move, so it stays in the field."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    @agent.instructions
    def current_date() -> str:
        return 'Today is 2026-08-18.'

    await agent.run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Today is 2026-08-18.'
    assert request['input'] == [{'role': 'user', 'content': 'Where is order 1234?'}]


async def test_openai_responses_stable_instructions_as_input_append_block_without_breakpoint(
    allow_model_requests: None,
):
    """An append block moves on GPT-6.2 even when `openai_cache_instructions` has no static instruction or system
    prompt to put its breakpoint on."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_cache_instructions=True)
    agent = Agent(model, deps_type=str, model_settings=settings)

    @agent.instructions(name='queue', on_change='append')
    def queue(ctx: RunContext[str]) -> str:
        return f'Queue: {ctx.deps}.'

    await agent.run('Where is order 1234?', deps='open')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'instructions' not in request
    assert request['input'] == snapshot(
        [
            {
                'role': 'system',
                'content': """\
<pyai:context id="agent:queue">
Queue: open.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


@pytest.mark.parametrize(
    'settings',
    [
        pytest.param(OpenAIResponsesModelSettings(openai_previous_response_id='resp_seed'), id='previous_response_id'),
        pytest.param(OpenAIResponsesModelSettings(openai_conversation_id='conv_123'), id='conversation_id'),
    ],
)
async def test_openai_responses_stable_instructions_stay_top_level_with_server_side_state(
    allow_model_requests: None, settings: OpenAIResponsesModelSettings
):
    """A stored response keeps its input and a continuing request inherits it, so with an explicit response id or a
    conversation the instructions stay in the top-level field, which isn't carried over."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model, instructions='Support policies.', model_settings=settings)

    first = await agent.run('Where is order 1234?')
    await agent.run('And order 5678?', message_history=first.all_messages())

    requests = get_mock_responses_kwargs(mock_client)
    assert [request['instructions'] for request in requests] == ['Support policies.', 'Support policies.']
    assert [item for request in requests for item in request['input'] if item.get('role') == 'system'] == []


@pytest.mark.parametrize(
    'instructions, expected_second_request',
    [
        pytest.param(
            'Support policies.',
            {'continues': True, 'roles': ['user'], 'system': []},
            id='same-agent',
        ),
        pytest.param(
            'Billing policies.',
            {'continues': False, 'roles': ['system', 'user', 'assistant', 'user'], 'system': ['Billing policies.']},
            id='handoff',
        ),
    ],
)
async def test_openai_responses_auto_chain_sends_stable_instructions_once(
    allow_model_requests: None, instructions: str, expected_second_request: dict[str, Any]
):
    """An `'auto'` chain sends the static instructions as leading input when it starts, and leaves them out while it
    continues, since the stored response holds them. Different ones fail the history digest, so a handoff resends the
    history, and the model sees only the current instructions."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    first = await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    await Agent(model, instructions=instructions, model_settings=settings).run(
        'And order 5678?', message_history=first.all_messages()
    )

    first_request, second_request = get_mock_responses_kwargs(mock_client)
    assert 'instructions' not in first_request
    assert first_request['input'] == [
        {'role': 'system', 'content': 'Support policies.'},
        {'role': 'user', 'content': 'Where is order 1234?'},
    ]
    assert 'instructions' not in second_request
    assert {
        'continues': second_request.get('previous_response_id') == first.response.provider_response_id,
        'roles': [item.get('role') for item in second_request['input']],
        'system': [item['content'] for item in second_request['input'] if item.get('role') == 'system'],
    } == expected_second_request


async def test_openai_responses_auto_chain_with_cache_instructions(allow_model_requests: None):
    """With `openai_cache_instructions`, the request that starts an `'auto'` chain puts the breakpoint after the static
    instructions it sends as leading input, and a continuing request sends neither."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(
        model,
        instructions='Support policies.',
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto', openai_cache_instructions=True),
    )

    first = await agent.run('Where is order 1234?')
    await agent.run('And order 5678?', message_history=first.all_messages())

    first_request, second_request = get_mock_responses_kwargs(mock_client)
    assert first_request['input'][0] == {
        'role': 'system',
        'content': [
            {'type': 'input_text', 'text': 'Support policies.', 'prompt_cache_breakpoint': {'mode': 'explicit'}}
        ],
    }
    assert second_request['previous_response_id'] == first.response.provider_response_id
    assert 'instructions' not in second_request
    assert second_request['input'] == [{'role': 'user', 'content': 'And order 5678?'}]


async def test_openai_responses_auto_chain_opted_out_keeps_instructions_top_level(allow_model_requests: None):
    """With the profile flag off, an `'auto'` chain keeps every instruction in the top-level field and continues."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel(
        'gpt-6.2-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(openai_responses_stable_instructions_as_input=False),
    )
    agent = Agent(
        model,
        instructions='Support policies.',
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'),
    )

    first = await agent.run('Where is order 1234?')
    await agent.run('And order 5678?', message_history=first.all_messages())

    assert [
        (
            request.get('previous_response_id') == first.response.provider_response_id,
            request['instructions'],
            [item.get('role') for item in request['input']],
        )
        for request in get_mock_responses_kwargs(mock_client)
    ] == [(False, 'Support policies.', ['user']), (True, 'Support policies.', ['user'])]


async def test_openai_responses_auto_chain_continues_across_appended_change(allow_model_requests: None):
    """An appended change and a dynamic instruction leave the chain's leading instructions as they were, so the
    request continues: the change goes out after the stored conversation, and the dynamic instruction in the field."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    agent = _support_agent(model, OpenAIResponsesModelSettings(openai_previous_response_id='auto'))

    first = await agent.run('Where is order 1234?', deps='open')
    await agent.run('And order 5678?', deps='closed', message_history=first.all_messages())

    second_request = get_mock_responses_kwargs(mock_client)[1]
    assert second_request['previous_response_id'] == first.response.provider_response_id
    assert {'instructions': second_request['instructions'], 'input': second_request['input']} == snapshot(
        {
            'instructions': 'Today is 2026-08-18.',
            'input': [
                {'role': 'user', 'content': 'And order 5678?'},
                {
                    'role': 'system',
                    'content': """\
<pyai:context id="agent:queue">
Queue: closed.
</pyai:context>\
""",
                },
            ],
        }
    )


@pytest.mark.parametrize(
    'as_input, expected_roles',
    [
        pytest.param((False, True), ['system', 'user', 'assistant', 'user'], id='turned-on'),
        pytest.param((True, False), ['user', 'assistant', 'user'], id='turned-off'),
    ],
)
async def test_openai_responses_auto_chain_resends_history_when_instruction_placement_changes(
    allow_model_requests: None, as_input: tuple[bool, bool], expected_roles: list[str]
):
    """A response stored with its instructions in the top-level field doesn't hold them in its input, and one stored
    with them as leading input does, so a request that would send them the other way resends the history."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    first_model, second_model = (
        OpenAIResponsesModel(
            'gpt-6.2-sol',
            provider=OpenAIProvider(openai_client=mock_client),
            profile=OpenAIModelProfile(openai_responses_stable_instructions_as_input=flag),
        )
        for flag in as_input
    )
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    first = await Agent(first_model, instructions='Support policies.', model_settings=settings).run(
        'Where is order 1234?'
    )

    await Agent(second_model, instructions='Support policies.', model_settings=settings).run(
        'And order 5678?', message_history=first.all_messages()
    )

    second_request = get_mock_responses_kwargs(mock_client)[1]
    assert 'previous_response_id' not in second_request
    assert [item.get('role') for item in second_request['input']] == expected_roles


async def test_openai_responses_auto_chain_compaction_resends_the_window(allow_model_requests: None):
    """A compaction item keeps the window's leading system items, so compacting an `'auto'` chain whose stored input
    starts with its instructions resends the window, with them in the top-level field, instead of continuing it."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion('first'))
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    first = await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')
    compact_kwargs: dict[str, Any] = {}

    async def fake_compact(**kwargs: Any) -> resp.CompactedResponse:
        compact_kwargs.update(kwargs)
        return resp.CompactedResponse(
            id='resp-compact',
            created_at=0,
            object='response.compaction',
            output=[ResponseCompactionItem(id='comp-1', encrypted_content='encrypted', type='compaction')],
            usage=ResponseUsage.model_construct(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    model.client.responses.compact = fake_compact
    messages = [
        *first.all_messages(),
        ModelRequest(parts=[UserPromptPart('And order 5678?')], instructions='Support policies.'),
    ]
    await model.compact_messages(
        ModelRequestContext(
            model=model,
            messages=messages,
            model_settings=settings,
            model_request_parameters=ModelRequestParameters(),
        )
    )

    assert compact_kwargs['previous_response_id'] is omit
    assert compact_kwargs['instructions'] == 'Support policies.'
    assert [item.get('role') for item in compact_kwargs['input']] == ['user', 'assistant', 'user']


async def test_openai_responses_auto_chain_computer_call_fallback_resends_the_instructions(allow_model_requests: None):
    """A result for a held computer call, sent without the native computer tool, makes the request resend the whole
    history. That request starts the chain again with the static and append instructions in the input, so its digest
    records them, and the next request continues from it."""
    computer_call = response_message(
        [
            resp.ResponseComputerToolCall.model_validate(
                {
                    'id': 'cu_1',
                    'call_id': 'call_1',
                    'type': 'computer_call',
                    'status': 'completed',
                    'actions': [{'type': 'screenshot'}],
                    'pending_safety_checks': [],
                }
            )
        ]
    )
    computer_call.id = 'resp_1'
    resent = responses_completion('resent')
    resent.id = 'resp_2'
    mock_client = MockOpenAIResponses.create_mock([computer_call, resent, responses_completion('continued')])
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    computer = ToolDefinition(name='computer', parameters_json_schema={'type': 'object'}, tool_kind=COMPUTER_KIND.id)
    instruction_parts = [
        InstructionPart(content='Support policies.'),
        InstructionPart(
            content='Queue: open.', on_change='append', id=InstructionId(source=AgentInstructionSource(), name='queue')
        ),
    ]
    with_native = ModelRequestParameters(
        function_tools=[computer],
        native_tools=[ComputerTool(display_size=FAKE_DISPLAY_SIZE, optional=True)],
        allow_text_output=True,
        instruction_parts=instruction_parts,
    )
    without_native = replace(with_native, native_tools=[])

    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Look.')]
    messages.append(await model.request(messages, settings, with_native))
    messages.append(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    'computer',
                    ['Ran 1 action.', BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png')],
                    tool_call_id='call_1',
                    metadata={'performed': 1},
                    tool_kind=COMPUTER_KIND.id,
                )
            ]
        )
    )
    messages.append(await model.request(messages, settings, without_native))
    messages.append(ModelRequest.user_text_prompt('Again.'))
    await model.request(messages, settings, without_native)

    _, resend, continuation = get_mock_responses_kwargs(mock_client)
    assert 'previous_response_id' not in resend
    assert [(item.get('role'), item.get('content')) for item in resend['input'][:2]] == snapshot(
        [
            ('system', 'Support policies.'),
            (
                'system',
                """\
<pyai:context id="agent:queue">
Queue: open.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
            ),
        ]
    )
    assert continuation['previous_response_id'] == 'resp_2'
    assert continuation['input'] == [{'role': 'user', 'content': 'Again.'}]


@pytest.mark.parametrize(
    'seed_settings', [None, OpenAIResponsesModelSettings(openai_previous_response_id='auto')], ids=['default', 'auto']
)
@pytest.mark.parametrize('instructions', ['Support policies.', 'Billing policies.'], ids=['same-agent', 'handoff'])
async def test_openai_responses_explicit_id_with_history_resends_it(
    allow_model_requests: None, instructions: str, seed_settings: OpenAIResponsesModelSettings | None
):
    """A response from a full-history request stores its instructions in its input but has no history digest, and an
    `'auto'` chain's records them as leading input, which an explicit id doesn't send. Either way, an explicit id
    passed with that history resends the history, and the model sees only the current instructions."""
    mock_client = MockOpenAIResponses.create_mock([responses_completion('first'), responses_completion('second')])
    model = OpenAIResponsesModel('gpt-6.2-sol', provider=OpenAIProvider(openai_client=mock_client))
    first = await Agent(model, instructions='Support policies.', model_settings=seed_settings).run(
        'Where is order 1234?'
    )
    response_id = first.response.provider_response_id
    assert response_id is not None

    await Agent(model, instructions=instructions).run(
        'And order 5678?',
        message_history=first.all_messages(),
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id=response_id),
    )

    request = get_mock_responses_kwargs(mock_client)[1]
    assert request.get('previous_response_id') is None
    assert request['instructions'] == instructions
    assert [item.get('role') for item in request['input']] == ['user', 'assistant', 'user']


async def test_openai_chat_cache_instructions_with_cache_point(allow_model_requests: None):
    """Both breakpoints are sent; OpenAI writes the last four, so the instruction one is dropped first."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIChatModelSettings(openai_cache_instructions=True)

    await Agent(model, instructions='Support policies.', model_settings=settings).run(
        ['Reference material.', CachePoint(), 'Where is order 1234?']
    )

    assert get_mock_chat_completion_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Support policies.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Reference material.',
                        'prompt_cache_breakpoint': {'mode': 'explicit'},
                    },
                    {'type': 'text', 'text': 'Where is order 1234?'},
                ],
            },
        ]
    )


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
        cost=Decimal('0.000558'),
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
        output_reasoning_tokens=10,
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
        output_reasoning_tokens=10,
        details={'reasoning_tokens': 10},
        cost=Decimal('0.007176'),
    )


async def test_openai_chat_usage_without_cache_write_tokens(allow_model_requests: None):
    """Token details lacking `cache_write_tokens` leave the usage field at 0."""
    mock_client = MockOpenAI.create_mock(
        chat_completion(
            usage=CompletionUsage(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
            )
        )
    )
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    result = await Agent(model).run('Hello')

    assert result.usage == RunUsage(requests=1, input_tokens=2, cache_read_tokens=1, output_tokens=1)


async def test_openai_responses_usage_without_cache_write_tokens(allow_model_requests: None):
    """Token details lacking `cache_write_tokens` leave the usage field at 0.

    The SDK parses API responses without validation, so the wire shape can omit the field
    (as OpenRouter's Responses endpoint does); `model_construct` reproduces that shape.
    """
    mock_client = MockOpenAIResponses.create_mock(
        responses_completion(
            usage=ResponseUsage(
                input_tokens=20,
                input_tokens_details=InputTokensDetails.model_construct(cached_tokens=5),
                output_tokens=3,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=23,
            )
        )
    )
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    result = await Agent(model).run('Hello')

    assert result.usage == RunUsage(
        requests=1,
        input_tokens=20,
        cache_read_tokens=5,
        output_tokens=3,
        output_reasoning_tokens=0,
        details={'reasoning_tokens': 0},
    )


# ===== Recorded accept-path tests =====

# The cacheable prefix must exceed OpenAI's minimum cacheable prompt length (1024 tokens).
_STABLE_PREFIX = 'Reference catalogue for the prompt cache test corpus.\n' + '\n'.join(
    f'Entry {i:04d}: shelf {i % 23}, aisle {i % 7}, volume {i}, catalogued under subject heading {i % 11}.'
    for i in range(160)
)


def _wire_shape(body: dict[str, Any]) -> dict[str, Any]:
    """What a Responses request continues, where it puts its instructions, and its input, with `_STABLE_PREFIX` cut."""

    def cut(content: Any) -> Any:
        return content.replace(_STABLE_PREFIX, '<_STABLE_PREFIX>') if isinstance(content, str) else content

    return {
        'previous_response_id': body.get('previous_response_id'),
        'instructions': cut(body.get('instructions')),
        'input': [(item.get('role'), cut(item.get('content'))) for item in body['input']],
    }


def _request_body(cassette: Cassette, index: int) -> dict[str, Any]:
    return cast('dict[str, Any]', request_json(cassette.requests[index]))


def _assert_cache_usage(first: RunUsage, second: RunUsage) -> None:
    """The first request writes the prefix (or reads a cache left by a recent recording run
    within the TTL); the identical second request must read it back."""
    assert first.cache_write_tokens > 0 or first.cache_read_tokens > 0
    assert second.cache_read_tokens > 0


@pytest.mark.vcr
async def test_openai_chat_prompt_cache_e2e(allow_model_requests: None, openai_api_key: str, vcr: Cassette):
    """Real OpenAI Chat accepts the cache fields and reports cache write and read usage.

    If the second request misses the cache when recording, re-record: writes usually
    propagate within seconds but are not instantaneous.
    """
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIChatModelSettings(
        openai_prompt_cache_key='pydantic-ai-prompt-cache-e2e-chat',
        openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
    )
    agent = Agent(model, model_settings=settings)
    prompt = [_STABLE_PREFIX, CachePoint(), 'Reply with exactly: OK']

    first = await agent.run(prompt)
    second = await agent.run(prompt)

    assert isinstance(first.output, str)
    assert isinstance(second.output, str)
    for index in (0, 1):
        body = _request_body(vcr, index)
        assert body['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
        assert body['prompt_cache_key'] == 'pydantic-ai-prompt-cache-e2e-chat'
        assert body['messages'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_cache_usage(first.usage, second.usage)


@pytest.mark.vcr
async def test_openai_responses_prompt_cache_e2e(allow_model_requests: None, openai_api_key: str, vcr: Cassette):
    """Real OpenAI Responses accepts the cache fields and reports cache write and read usage."""
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(
        openai_prompt_cache_key='pydantic-ai-prompt-cache-e2e-responses',
        openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
    )
    agent = Agent(model, model_settings=settings)
    prompt = [_STABLE_PREFIX, CachePoint(), 'Reply with exactly: OK']

    first = await agent.run(prompt)
    second = await agent.run(prompt)

    assert isinstance(first.output, str)
    assert isinstance(second.output, str)
    for index in (0, 1):
        body = _request_body(vcr, index)
        assert body['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
        assert body['prompt_cache_key'] == 'pydantic-ai-prompt-cache-e2e-responses'
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_cache_usage(first.usage, second.usage)


@pytest.mark.vcr
@pytest.mark.parametrize(
    'model_name',
    [
        'gpt-5.6-sol',
        'gpt-5.6-terra',
        'gpt-5.6-luna',
        'gpt-6-astra',
        'gpt-6-sol',
        'gpt-6-luna',
        'gpt-6.1-sol',
        'gpt-6.1-terra',
    ],
)
async def test_openai_responses_unified_cache_12h_e2e(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette, model_name: str
):
    """`cache='12h'` sends the 12-hour TTL on Responses, which every GPT-5.6 and GPT-6 model accepts."""
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions=_STABLE_PREFIX, model_settings=ModelSettings(cache='12h'))

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: OK')

    for index in (0, 1):
        body = _request_body(vcr, index)
        assert body['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '12h'}
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_cache_usage(first.usage, second.usage)


@pytest.mark.vcr
async def test_openai_chat_unified_cache_12h_e2e(allow_model_requests: None, openai_api_key: str, vcr: Cassette):
    """Chat Completions rejects a `'12h'` TTL, so `cache='12h'` sends `'30m'` there."""
    model = OpenAIChatModel('gpt-5.6-luna', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, instructions=_STABLE_PREFIX, model_settings=ModelSettings(cache='12h'))

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: OK')

    for index in (0, 1):
        assert _request_body(vcr, index)['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    _assert_cache_usage(first.usage, second.usage)


@pytest.mark.vcr
async def test_openrouter_responses_prompt_cache_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
):
    """OpenRouter's Responses API accepts the OpenAI cache protocol for GPT-5.6.

    The downstream provider is pinned to OpenAI: OpenRouter also offers Azure routes for
    GPT-5.6, where the explicit-cache fields are not documented.
    """
    model = OpenAIResponsesModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(api_key=openrouter_api_key))
    settings = OpenAIResponsesModelSettings(
        openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
        extra_body={'provider': {'only': ['openai']}},
    )
    agent = Agent(model, model_settings=settings)
    prompt = [_STABLE_PREFIX, CachePoint(), 'Reply with exactly: OK']

    first = await agent.run(prompt)
    second = await agent.run(prompt)

    assert isinstance(first.output, str)
    assert isinstance(second.output, str)
    for index in (0, 1):
        body = _request_body(vcr, index)
        assert body['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_cache_usage(first.usage, second.usage)


_UNCACHED_SUFFIX = (
    '\n'.join(f'Margin note {i:03d}: this line follows the cache breakpoint.' for i in range(40))
    + '\nReply with exactly: OK'
)
"""Hundreds of tokens after the `CachePoint`: a write that stops short of them shows the breakpoint was honored."""


_AZURE_CACHE_MODELS = ['gpt-5.6-luna', 'gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-sol', 'gpt-6-astra']
"""The models with explicit cache breakpoints that the Gateway's Azure route serves: all but `gpt-6.1-terra`."""


def _azure_gateway_provider(gateway_api_key: str | None, request_capture: RequestCapture) -> AzureProvider:
    """Azure OpenAI's v1 API, reached through the Pydantic AI Gateway's `azure` route.

    Requests go through `request_capture`, so tests assert on the bodies the live code built rather than the
    recorded ones, which replay would match on method and URI alone.
    """
    gateway_url = os.getenv('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.info/proxy')
    return AzureProvider(
        azure_endpoint=f'{gateway_url}/azure/openai/v1',
        api_key=gateway_api_key or 'test-api-key',
        http_client=request_capture.client,
    )


async def _run_twice(
    agent: Agent[None, str], prompt: list[str | CachePoint], stream: bool
) -> tuple[RunUsage, RunUsage]:
    usages: list[RunUsage] = []
    for _ in range(2):
        if stream:
            async with agent.run_stream(prompt) as streamed:
                assert isinstance(await streamed.get_output(), str)
            usages.append(streamed.usage)
        else:
            result = await agent.run(prompt)
            assert isinstance(result.output, str)
            usages.append(result.usage)
    return usages[0], usages[1]


def _assert_azure_breakpoint_cached(first: RunUsage, second: RunUsage) -> None:
    """The first request writes the prefix up to the `CachePoint`, and the second reads exactly that back.

    Recording again within the 30-minute TTL makes the first request read instead: change the
    prompt's first line to re-record.
    """
    assert first.cache_read_tokens == 0
    assert first.cache_write_tokens >= 1024
    assert second.cache_write_tokens == 0
    assert second.cache_read_tokens == first.cache_write_tokens
    # `_UNCACHED_SUFFIX` after the breakpoint was neither written nor read.
    assert second.input_tokens - second.cache_read_tokens > 300


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize('model_name', _AZURE_CACHE_MODELS)
@pytest.mark.vcr
async def test_azure_chat_prompt_cache_e2e(
    allow_model_requests: None,
    gateway_api_key: str | None,
    request_capture: RequestCapture,
    model_name: str,
    stream: bool,
):
    """Azure OpenAI Chat honors the explicit cache fields on GPT-5.6 and later, and reports cache write and read usage.

    With `mode='explicit'` only the prefix up to the `CachePoint` is written, so a write that stops
    short of the whole prompt shows Azure honored the breakpoint rather than caching implicitly.
    """
    model = OpenAIChatModel(model_name, provider=_azure_gateway_provider(gateway_api_key, request_capture))
    settings = OpenAIChatModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})
    agent = Agent(model, model_settings=settings)
    prompt = [
        f'Azure Chat cache e2e (take 2), {model_name}, stream={stream}.\n{_STABLE_PREFIX}',
        CachePoint(),
        _UNCACHED_SUFFIX,
    ]

    first, second = await _run_twice(agent, prompt, stream)

    bodies = cast('list[dict[str, Any]]', request_capture.bodies('/chat/completions'))
    assert len(bodies) == 2
    for body in bodies:
        assert body['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
        assert body['messages'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_azure_breakpoint_cached(first, second)


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize('model_name', _AZURE_CACHE_MODELS)
@pytest.mark.vcr
async def test_azure_responses_prompt_cache_e2e(
    allow_model_requests: None,
    gateway_api_key: str | None,
    request_capture: RequestCapture,
    model_name: str,
    stream: bool,
):
    """Azure OpenAI Responses honors the explicit cache fields on GPT-5.6 and later, and reports cache write and
    read usage."""
    model = OpenAIResponsesModel(model_name, provider=_azure_gateway_provider(gateway_api_key, request_capture))
    settings = OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'})
    agent = Agent(model, model_settings=settings)
    prompt = [
        f'Azure Responses cache e2e (take 2), {model_name}, stream={stream}.\n{_STABLE_PREFIX}',
        CachePoint(),
        _UNCACHED_SUFFIX,
    ]

    first, second = await _run_twice(agent, prompt, stream)

    bodies = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert len(bodies) == 2
    for body in bodies:
        assert body['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_azure_breakpoint_cached(first, second)


@pytest.mark.parametrize('model_name', _AZURE_CACHE_MODELS)
@pytest.mark.vcr
async def test_azure_responses_cache_instructions_e2e(
    allow_model_requests: None, gateway_api_key: str | None, request_capture: RequestCapture, model_name: str
):
    """Azure OpenAI Responses honors an instruction breakpoint with the implicit-mode options, the settings
    `cache=True` maps to on the OpenAI API, and the second request reads the cache the first wrote."""
    model = OpenAIResponsesModel(model_name, provider=_azure_gateway_provider(gateway_api_key, request_capture))
    settings = OpenAIResponsesModelSettings(
        openai_prompt_cache_options={'mode': 'implicit', 'ttl': '30m'}, openai_cache_instructions=True
    )
    # Change the instructions' last line to re-record within the 30-minute TTL.
    agent = Agent(
        model,
        instructions=_STABLE_PREFIX + f'\n(Azure instruction breakpoint, take 2, {model_name})',
        model_settings=settings,
    )

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: OK')

    bodies = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert len(bodies) == 2
    for body in bodies:
        assert body['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
        assert body['input'][0]['role'] == 'system'
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    assert first.usage.cache_read_tokens == 0
    assert first.usage.cache_write_tokens >= 1024
    assert second.usage.cache_write_tokens == 0
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens


@pytest.mark.parametrize('model_name', _AZURE_CACHE_MODELS)
@pytest.mark.vcr
async def test_azure_chat_cache_instructions_e2e(
    allow_model_requests: None, gateway_api_key: str | None, request_capture: RequestCapture, model_name: str
):
    """Azure OpenAI Chat honors an instruction breakpoint on the system message with the implicit-mode options, the
    settings `cache=True` maps to on the OpenAI API, and the second request reads the cache the first wrote."""
    model = OpenAIChatModel(model_name, provider=_azure_gateway_provider(gateway_api_key, request_capture))
    settings = OpenAIChatModelSettings(
        openai_prompt_cache_options={'mode': 'implicit', 'ttl': '30m'}, openai_cache_instructions=True
    )
    # Change the instructions' last line to re-record within the 30-minute TTL.
    agent = Agent(
        model,
        instructions=_STABLE_PREFIX + f'\n(Azure Chat instruction breakpoint, {model_name})',
        model_settings=settings,
    )

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: OK')

    bodies = cast('list[dict[str, Any]]', request_capture.bodies('/chat/completions'))
    assert len(bodies) == 2
    for body in bodies:
        assert body['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
        assert body['messages'][0]['role'] == 'system'
        assert body['messages'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    assert first.usage.cache_read_tokens == 0
    assert first.usage.cache_write_tokens >= 1024
    assert second.usage.cache_write_tokens == 0
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens


def _assert_instruction_prefix_shared(first: RunUsage, second: RunUsage) -> None:
    """The second conversation reads the cached instructions written (or read) by the first.

    With `mode='explicit'` there's no implicit breakpoint, so the read comes from the instruction
    breakpoint, and everything but the second conversation's own user turn is read from the cache.
    """
    assert first.cache_write_tokens > 0 or first.cache_read_tokens > 0
    assert second.cache_read_tokens >= 1024
    assert second.input_tokens - second.cache_read_tokens < 64


@pytest.mark.vcr
async def test_openai_chat_cache_instructions_e2e(allow_model_requests: None, openai_api_key: str, vcr: Cassette):
    """Separate conversations share the instruction breakpoint on Chat Completions."""
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIChatModelSettings(
        openai_cache_instructions=True,
        openai_prompt_cache_key='pydantic-ai-cache-instructions-e2e-chat',
        openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
    )
    agent = Agent(model, instructions=_STABLE_PREFIX, model_settings=settings)

    first = await agent.run('Which shelf holds entry 0042? Reply with just the number.')
    second = await agent.run('Which aisle holds entry 0101? Reply with just the number.')

    for index in (0, 1):
        messages = _request_body(vcr, index)['messages']
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_instruction_prefix_shared(first.usage, second.usage)


@pytest.mark.vcr
@pytest.mark.parametrize('system_prompt_role', ['system', 'developer'])
async def test_openai_responses_cache_instructions_e2e(
    allow_model_requests: None, openai_api_key: str, vcr: Cassette, system_prompt_role: Literal['system', 'developer']
):
    """Separate conversations share the instructions relocated into leading input messages on Responses."""
    model = OpenAIResponsesModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=OpenAIModelProfile(openai_system_prompt_role=system_prompt_role),
    )
    settings = OpenAIResponsesModelSettings(
        openai_cache_instructions=True,
        openai_prompt_cache_key=f'pydantic-ai-cache-instructions-e2e-responses-{system_prompt_role}',
        openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
    )
    agent = Agent(model, instructions=_STABLE_PREFIX, model_settings=settings)

    first = await agent.run('Which shelf holds entry 0042? Reply with just the number.')
    second = await agent.run('Which aisle holds entry 0101? Reply with just the number.')

    for index in (0, 1):
        body = _request_body(vcr, index)
        assert 'instructions' not in body
        assert body['input'][0]['role'] == system_prompt_role
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    _assert_instruction_prefix_shared(first.usage, second.usage)


@pytest.mark.vcr
@pytest.mark.parametrize(
    'stable_instructions_as_input, expected_requests',
    [
        pytest.param(
            True,
            snapshot(
                [
                    {
                        'previous_response_id': None,
                        'instructions': 'Queue position: 17.',
                        'input': [
                            ('system', '<_STABLE_PREFIX>'),
                            ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                        ],
                    },
                    {
                        'previous_response_id': None,
                        'instructions': 'Queue position: 12.',
                        'input': [
                            ('system', '<_STABLE_PREFIX>'),
                            ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                            ('assistant', [{'text': '19', 'type': 'output_text', 'annotations': []}]),
                            ('user', 'Which aisle holds entry 0101? Reply with just the number.'),
                        ],
                    },
                ]
            ),
            id='leading-input',
        ),
        pytest.param(
            False,
            snapshot(
                [
                    {
                        'previous_response_id': None,
                        'instructions': """\
<_STABLE_PREFIX>

Queue position: 17.\
""",
                        'input': [('user', 'Which shelf holds entry 0042? Reply with just the number.')],
                    },
                    {
                        'previous_response_id': None,
                        'instructions': """\
<_STABLE_PREFIX>

Queue position: 12.\
""",
                        'input': [
                            ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                            ('assistant', [{'text': '19', 'type': 'output_text', 'annotations': []}]),
                            ('user', 'Which aisle holds entry 0101? Reply with just the number.'),
                        ],
                    },
                ]
            ),
            id='top-level',
        ),
    ],
)
async def test_openai_responses_stable_instructions_as_input_e2e(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    stable_instructions_as_input: bool,
    expected_requests: list[dict[str, Any]],
):
    """On GPT-6.2, static instructions leading the input are cached with the conversation, also while a dynamic
    instruction in the top-level field changes. In the top-level field, OpenAI puts them after the input and the next
    request reads none of them from the cache."""
    model = OpenAIResponsesModel(
        'gpt-6.2-sol',
        provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client),
        profile=OpenAIModelProfile(openai_responses_stable_instructions_as_input=stable_instructions_as_input),
    )
    agent = Agent(model, instructions=_STABLE_PREFIX)
    positions = iter(['Queue position: 17.', 'Queue position: 12.'])

    @agent.instructions
    def queue_position() -> str:
        return next(positions)

    first = await agent.run('Which shelf holds entry 0042? Reply with just the number.')
    second = await agent.run(
        'Which aisle holds entry 0101? Reply with just the number.', message_history=first.all_messages()
    )

    first_body, second_body = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert [_wire_shape(first_body), _wire_shape(second_body)] == expected_requests
    assert second_body['input'][: len(first_body['input'])] == first_body['input']
    if stable_instructions_as_input:
        # The second request reads back everything the first wrote: its whole input, ahead of the top-level field.
        assert first.usage.cache_write_tokens + first.usage.cache_read_tokens > 0
        assert second.usage.cache_read_tokens == first.usage.cache_write_tokens + first.usage.cache_read_tokens
    else:
        assert second.usage.cache_read_tokens == 0


@pytest.mark.vcr
@pytest.mark.parametrize('model_name', ['gpt-6.2-sol', 'gpt-6.2-astra'])
async def test_openai_responses_rewrite_change_keeps_cache_e2e(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture, model_name: str
):
    """After an appended change, a rewrite-mode change reads everything the previous request cached."""
    model = OpenAIResponsesModel(
        model_name, provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )

    _, second, third = await _run_rewrite_change(model, _STABLE_PREFIX)

    _, second_body, third_body = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert third_body['input'][: len(second_body['input'])] == second_body['input']
    assert third_body['instructions'] == 'Today is D2.'
    assert second.usage.cache_read_tokens > 0
    assert third.usage.cache_read_tokens == second.usage.cache_write_tokens + second.usage.cache_read_tokens


@pytest.mark.vcr
async def test_openai_responses_explicit_id_with_history_e2e(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    """An explicit id passed with the history of a default GPT-6.2 run resends that history once, then continues from
    the new response. Those requests keep the instructions in the top-level field, which isn't cached."""
    model = OpenAIResponsesModel(
        'gpt-6.2-sol', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    agent = Agent(model, instructions=_STABLE_PREFIX)

    first = await agent.run('Which shelf holds entry 0042? Reply with just the number.')
    first_id = first.response.provider_response_id
    assert first_id is not None
    second = await agent.run(
        'Which aisle holds entry 0101? Reply with just the number.',
        message_history=first.all_messages(),
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id=first_id),
    )
    second_id = second.response.provider_response_id
    assert second_id is not None
    third = await agent.run(
        'Which shelf holds entry 0101? Reply with just the number.',
        message_history=second.all_messages(),
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id=second_id),
    )

    bodies = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert [_wire_shape(body) for body in bodies] == snapshot(
        [
            {
                'previous_response_id': None,
                'instructions': None,
                'input': [
                    ('system', '<_STABLE_PREFIX>'),
                    ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                ],
            },
            {
                'previous_response_id': None,
                'instructions': '<_STABLE_PREFIX>',
                'input': [
                    ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                    ('assistant', [{'text': '19', 'type': 'output_text', 'annotations': []}]),
                    ('user', 'Which aisle holds entry 0101? Reply with just the number.'),
                ],
            },
            {
                'previous_response_id': 'resp_017d0ab63852be68006b047b02859c87d29e24628d98e4d657',
                'instructions': '<_STABLE_PREFIX>',
                'input': [('user', 'Which shelf holds entry 0101? Reply with just the number.')],
            },
        ]
    )
    assert first.usage.cache_write_tokens > 0
    assert (second.usage.cache_read_tokens, third.usage.cache_read_tokens) == (0, 0)


@pytest.mark.vcr
async def test_openai_responses_auto_chain_e2e(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    """An `'auto'` chain on GPT-6.2 sends its instructions as leading input when it starts and leaves them out while it
    continues, so the continuation reads them from the cache. A change resends the history, which the next request
    continues from in turn."""
    model = OpenAIResponsesModel(
        'gpt-6.2-sol', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    changed = _STABLE_PREFIX + '\nAnswer with just the number.'

    first = await Agent(model, instructions=_STABLE_PREFIX, model_settings=settings).run(
        'Which shelf holds entry 0042? Reply with just the number.'
    )
    second = await Agent(model, instructions=_STABLE_PREFIX, model_settings=settings).run(
        'Which aisle holds entry 0101? Reply with just the number.', message_history=first.all_messages()
    )
    third = await Agent(model, instructions=changed, model_settings=settings).run(
        'Which volume is entry 0077?', message_history=second.all_messages()
    )
    fourth = await Agent(model, instructions=changed, model_settings=settings).run(
        'Which shelf holds entry 0101?', message_history=third.all_messages()
    )

    bodies = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert [_wire_shape(body) for body in bodies] == snapshot(
        [
            {
                'previous_response_id': None,
                'instructions': None,
                'input': [
                    ('system', '<_STABLE_PREFIX>'),
                    ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                ],
            },
            {
                'previous_response_id': 'resp_04a5bbea70ed9246006b05a27282f087d28291b1a50abd0ae2',
                'instructions': None,
                'input': [('user', 'Which aisle holds entry 0101? Reply with just the number.')],
            },
            {
                'previous_response_id': None,
                'instructions': None,
                'input': [
                    (
                        'system',
                        """\
<_STABLE_PREFIX>
Answer with just the number.\
""",
                    ),
                    ('user', 'Which shelf holds entry 0042? Reply with just the number.'),
                    ('assistant', [{'text': '19', 'type': 'output_text', 'annotations': []}]),
                    ('user', 'Which aisle holds entry 0101? Reply with just the number.'),
                    ('assistant', [{'text': '3', 'type': 'output_text', 'annotations': []}]),
                    ('user', 'Which volume is entry 0077?'),
                ],
            },
            {
                'previous_response_id': 'resp_04a5bbea70ed9246006b05a275e61087d291bf29e58ba7ac94',
                'instructions': None,
                'input': [('user', 'Which shelf holds entry 0101?')],
            },
        ]
    )
    # Each continuation reads back everything the request before it cached, so the whole stored prefix stays cached.
    assert first.usage.cache_write_tokens + first.usage.cache_read_tokens > 0
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens + first.usage.cache_read_tokens
    assert fourth.usage.cache_read_tokens == third.usage.cache_write_tokens + third.usage.cache_read_tokens


# ===== Prompt cache diagnostics =====


def _last_response(messages: list[Any]) -> ModelResponse:
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    return response


def _diagnostics_model(model_name: str, openai_api_key: str, request_capture: RequestCapture) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(
        model_name, provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )


@pytest.mark.vcr
async def test_openai_responses_prompt_cache_diagnostics_e2e(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    """Each turn compares against the previous OpenAI response, including across a model switch.

    Recorded with `openai_store=False`: diagnostics don't need the compared response to be stored.
    """
    agent = Agent(
        _diagnostics_model('gpt-5.6-sol', openai_api_key, request_capture),
        model_settings=OpenAIResponsesModelSettings(openai_store=False),
    )

    first = await agent.run([_STABLE_PREFIX, 'Reply with exactly: OK'])
    second = await agent.run('Reply with exactly: AGAIN', message_history=first.all_messages())
    third = await agent.run(
        'Reply with exactly: DONE',
        message_history=second.all_messages(),
        model=_diagnostics_model('gpt-5.6-luna', openai_api_key, request_capture),
    )

    first_response = _last_response(first.all_messages())
    second_response = _last_response(second.all_messages())
    third_response = _last_response(third.all_messages())
    assert [body.get('prompt_cache_options') for body in request_capture.bodies('/v1/responses')] == [
        None,
        {'comparison_response_id': first_response.provider_response_id},
        {'comparison_response_id': second_response.provider_response_id},
    ]
    assert first_response.provider_details is not None
    assert 'prompt_cache_diagnostics' not in first_response.provider_details
    assert second_response.provider_details is not None
    assert second_response.provider_details['prompt_cache_diagnostics'] == snapshot({'type': 'cache_hit'})
    assert third_response.provider_details is not None
    assert third_response.provider_details['prompt_cache_diagnostics'] == snapshot(
        {
            'cache_missed_tokens': 4175,
            'reason': 'model_changed',
            'type': 'cache_miss',
            'comparison_reusable_tokens': 4175,
        }
    )
    # The recorded shape survives message-history serialization.
    round_tripped = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(third.all_messages()))
    round_tripped_details = _last_response(round_tripped).provider_details
    assert round_tripped_details is not None
    assert (
        round_tripped_details['prompt_cache_diagnostics'] == third_response.provider_details['prompt_cache_diagnostics']
    )


@pytest.mark.vcr
async def test_openai_responses_prompt_cache_diagnostics_stream(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    """A streamed response takes the diagnostics from the terminal event.

    Earlier status events can report `unavailable` before the comparison has finished.
    """
    agent = Agent(_diagnostics_model('gpt-5.6-sol', openai_api_key, request_capture), model_settings={'cache': False})

    first = await agent.run([_STABLE_PREFIX, 'Reply with exactly: OK'])
    async with agent.run_stream('Reply with exactly: DONE', message_history=first.all_messages()) as streamed:
        await streamed.get_output()

    assert request_capture.body('/v1/responses', index=1)['prompt_cache_options'] == {
        'comparison_response_id': _last_response(first.all_messages()).provider_response_id
    }
    streamed_response = _last_response(streamed.all_messages())
    assert streamed_response.provider_details is not None
    assert streamed_response.provider_details['prompt_cache_diagnostics'] == snapshot({'type': 'cache_hit'})


@pytest.mark.vcr
async def test_openai_responses_prompt_cache_diagnostics_not_requested_before_gpt_5_6(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    """Models before GPT-5.6 accept the field but always answer `unavailable`, so it isn't sent."""
    agent = Agent(_diagnostics_model('gpt-5.5', openai_api_key, request_capture))

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: AGAIN', message_history=first.all_messages())

    assert 'prompt_cache_options' not in request_capture.body('/v1/responses', index=1)
    second_response = _last_response(second.all_messages())
    assert second_response.provider_details is not None
    assert 'prompt_cache_diagnostics' not in second_response.provider_details


def _history(
    provider_name: str, provider_response_id: str, provider_details: dict[str, Any] | None = None
) -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest.user_text_prompt('Say hi.'),
        ModelResponse(
            parts=[TextPart('Hi!')],
            model_name='gpt-5.6-sol',
            provider_name=provider_name,
            provider_response_id=provider_response_id,
            provider_details=provider_details,
        ),
    ]


@pytest.mark.parametrize(
    ('provider', 'history', 'settings', 'expected'),
    [
        pytest.param(
            'openai', _history('openai', 'resp_1'), {}, {'comparison_response_id': 'resp_1'}, id='previous-openai'
        ),
        pytest.param(
            'openai',
            _history('openai', 'resp_1'),
            {'openai_prompt_cache_options': {'mode': 'explicit', 'ttl': '30m'}},
            {'mode': 'explicit', 'ttl': '30m', 'comparison_response_id': 'resp_1'},
            id='merged-with-options',
        ),
        pytest.param(
            'openai', _history('openai', 'resp_1'), {'openai_prompt_cache_diagnostics': False}, None, id='disabled'
        ),
        pytest.param('openai', _history('anthropic', 'msg_1'), {}, None, id='previous-other-provider'),
        # Azure also issues `resp_` IDs, but OpenAI can't look them up.
        pytest.param('openai', _history('azure', 'resp_1'), {}, None, id='previous-azure'),
        # OpenAI rejects an ID that doesn't start with `resp` with a 400.
        pytest.param('openai', _history('openai', 'chatcmpl-1'), {}, None, id='previous-non-response-id'),
        pytest.param('openai', [ModelRequest.user_text_prompt('Say hi.')], {}, None, id='no-previous-response'),
        # A `/compact` response's ID can't be used as `previous_response_id`, and the request after it starts a new prefix.
        pytest.param(
            'openai', _history('openai', 'resp_1', {'compaction': True}), {}, None, id='previous-compaction-response'
        ),
        # OpenRouter's Responses API rejects the field with a 400.
        pytest.param('openrouter', _history('openrouter', 'resp_1'), {}, None, id='openrouter'),
    ],
)
async def test_openai_responses_prompt_cache_diagnostics_request_field(
    allow_model_requests: None,
    provider: Literal['openai', 'openrouter'],
    history: list[ModelRequest | ModelResponse],
    settings: OpenAIResponsesModelSettings,
    expected: dict[str, str] | None,
):
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    if provider == 'openai':
        model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    else:
        model = OpenAIResponsesModel('openai/gpt-5.6-sol', provider=OpenRouterProvider(openai_client=mock_client))

    await Agent(model, model_settings=settings).run('Reply with exactly: OK', message_history=history)

    assert get_mock_responses_kwargs(mock_client)[0].get('prompt_cache_options') == expected


async def test_openai_chat_unified_cache(allow_model_requests: None):
    """On GPT-5.6, `cache=True` caches with the 30-minute TTL and an instruction breakpoint."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model, instructions='Support policies.', model_settings={'cache': True}).run('Where is order 1234?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    assert kwargs['messages'][0] == snapshot(
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': 'Support policies.', 'prompt_cache_breakpoint': {'mode': 'explicit'}}],
        }
    )


def _azure_wire_response(request: httpx2.Request) -> httpx2.Response:
    """A minimal Chat Completions or Responses reply, so a real `AzureProvider` client can run end to end."""
    if request.url.path.endswith('/chat/completions'):
        message = {'role': 'assistant', 'content': 'OK'}
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'gpt-5.6-sol',
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
            },
        )
    text: dict[str, JsonValue] = {'type': 'output_text', 'text': 'OK', 'annotations': []}
    return httpx2.Response(
        200,
        json={
            'id': 'resp-test',
            'object': 'response',
            'created_at': 0,
            'model': 'gpt-5.6-sol',
            'status': 'completed',
            'output': [
                {'id': 'msg-test', 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [text]}
            ],
        },
    )


_AzureSetup = Literal['v1_endpoint', 'v1_azure_client', 'api_version', 'serverless']


def _azure_provider(setup: _AzureSetup, http_client: httpx2.AsyncClient) -> AzureProvider:
    """An `AzureProvider` built the way users configure each Azure API."""
    if setup == 'v1_endpoint':
        return AzureProvider(
            azure_endpoint='https://example.openai.azure.com/openai/v1/', api_key='test', http_client=http_client
        )
    if setup == 'v1_azure_client':
        client = AsyncAzureOpenAI(
            base_url='https://example.openai.azure.com/openai/v1/',
            api_version='preview',
            api_key='test',
            http_client=http_client,
        )
        return AzureProvider(openai_client=client)
    if setup == 'api_version':
        return AzureProvider(
            azure_endpoint='https://example.openai.azure.com',
            api_version='2025-04-01-preview',
            api_key='test',
            http_client=http_client,
        )
    return AzureProvider(azure_endpoint='https://example.models.ai.azure.com', api_key='test', http_client=http_client)


@pytest.mark.parametrize(
    ('setup', 'cached'),
    [('v1_endpoint', True), ('v1_azure_client', True), ('api_version', False), ('serverless', False)],
)
@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize(
    'model_name', ['gpt-5.6-sol', 'gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-sol', 'gpt-6.1-terra', 'gpt-6-astra']
)
async def test_azure_unified_cache(
    allow_model_requests: None,
    request_capture: RequestCapture,
    setup: _AzureSetup,
    cached: bool,
    api: Literal['chat', 'responses'],
    model_name: str,
):
    """With the Azure OpenAI v1 API, through either client class, `cache=True` maps onto the prompt cache options
    of GPT-5.6 and later and an instruction breakpoint, as on OpenAI. Dated `api-version` deployments and serverless
    endpoints aren't verified to accept them, so they get neither."""
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(_azure_wire_response),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    ) as http_client:
        provider = _azure_provider(setup, http_client)
        model = (OpenAIChatModel if api == 'chat' else OpenAIResponsesModel)(model_name, provider=provider)

        await Agent(model, instructions='Support policies.', model_settings={'cache': True}).run('Where is order 1234?')

    [body] = cast(
        'list[dict[str, Any]]', request_capture.bodies('/chat/completions' if api == 'chat' else '/responses')
    )
    first = body['messages' if api == 'chat' else 'input'][0]
    if cached:
        assert body['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
        assert first['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
        assert model.resolve_cache_retention({'cache': True}) == timedelta(minutes=30)
        assert model.resolve_cache_retention({'cache': '1h'}) == timedelta(minutes=30)
    else:
        assert 'prompt_cache_options' not in body
        assert 'prompt_cache_breakpoint' not in json.dumps(body)
        assert model.resolve_cache_retention({'cache': True}) is None


async def test_azure_unified_cache_custom_deployment_name(allow_model_requests: None, request_capture: RequestCapture):
    """A deployment whose name isn't a model name opts in with a `profile=` that turns breakpoints and the unified
    setting on, since the provider's profile is derived from the name."""
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(_azure_wire_response),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    ) as http_client:
        provider = _azure_provider('v1_endpoint', http_client)
        assert OpenAIResponsesModel('prod-chat', provider=provider).resolve_cache_retention({'cache': True}) is None
        profile = OpenAIModelProfile(
            openai_supports_prompt_cache_breakpoints=True, supports_cache=True, supported_cache_retentions=('30m',)
        )
        model = OpenAIResponsesModel('prod-chat', provider=provider, profile=profile)

        await Agent(model, instructions='Support policies.', model_settings={'cache': True}).run('Where is order 1234?')

    [body] = cast('list[dict[str, Any]]', request_capture.bodies('/responses'))
    assert body['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    assert model.resolve_cache_retention({'cache': True}) == timedelta(minutes=30)


async def test_azure_unified_cache_ignored_before_gpt_5_6(allow_model_requests: None, request_capture: RequestCapture):
    """Earlier Azure deployments cache implicitly with nothing to configure, so the unified setting sends nothing."""
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(_azure_wire_response),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    ) as http_client:
        model = OpenAIResponsesModel('gpt-5.2', provider=_azure_provider('v1_endpoint', http_client))

        await Agent(model, instructions='Support policies.', model_settings={'cache': True}).run('Where is order 1234?')

    [body] = request_capture.bodies('/responses')
    assert 'prompt_cache_options' not in body
    assert body['instructions'] == 'Support policies.'


@pytest.mark.parametrize('cache', [True, '5m', '1h'])
async def test_openai_responses_unified_cache_snaps_to_30m(
    allow_model_requests: None, cache: Literal[True, '5m', '1h']
):
    """`'30m'` is the only TTL OpenAI accepts, so every requested retention snaps to it."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model, instructions='Support policies.', model_settings={'cache': cache}).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    assert 'instructions' not in request
    assert request['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    assert model.resolve_cache_retention({'cache': cache}) == timedelta(minutes=30)


async def test_openai_responses_unified_cache_false_sends_nothing(allow_model_requests: None):
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model, instructions='Support policies.', model_settings={'cache': False}).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'prompt_cache_options' not in request
    assert request['instructions'] == 'Support policies.'
    assert model.resolve_cache_retention({'cache': False}) is None


async def test_openai_responses_unified_cache_respects_chaining_gate(allow_model_requests: None):
    """With server-side state, the instructions stay top-level as `openai_cache_instructions` requires."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto', cache=True)

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    assert request['instructions'] == 'Support policies.'


@pytest.mark.parametrize(
    'settings',
    [
        OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit'}),
        OpenAIResponsesModelSettings(openai_cache_instructions=False),
    ],
)
async def test_openai_responses_explicit_cache_settings_win(
    allow_model_requests: None, settings: OpenAIResponsesModelSettings
):
    """Any explicit OpenAI cache setting takes precedence over the unified setting entirely."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['instructions'] == 'Support policies.'
    assert request.get('prompt_cache_options') == settings.get('openai_prompt_cache_options')


def test_openai_cache_settings_resolution():
    """GPT-5.6 resolves its 30-minute TTL unless `mode='explicit'` leaves only `CachePoint`s to cache, and is
    never reported as having caching not enabled, since it caches implicitly without any configuration."""
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(api_key='test'))
    assert not model._caching_not_enabled(None)  # pyright: ignore[reportPrivateUsage]
    earlier = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key='test'))
    assert not earlier._caching_not_enabled(None)  # pyright: ignore[reportPrivateUsage]
    assert model.resolve_cache_retention(OpenAIResponsesModelSettings(openai_cache_instructions=False)) == timedelta(
        minutes=30
    )
    assert (
        model.resolve_cache_retention(OpenAIResponsesModelSettings(openai_prompt_cache_options={'mode': 'explicit'}))
        is None
    )
    # A deprecated maximum on GPT-5.6, so it doesn't widen the window.
    assert model.resolve_cache_retention(OpenAIResponsesModelSettings(openai_prompt_cache_retention='24h')) is None


async def test_openai_unified_cache_ignored_before_gpt_5_6(allow_model_requests: None):
    """Earlier models cache implicitly with nothing to configure, so the unified setting sends nothing."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model, instructions='Support policies.', model_settings={'cache': '1h'}).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert 'prompt_cache_options' not in request
    assert request['instructions'] == 'Support policies.'
    assert model.resolve_cache_retention({'cache': '1h'}) is None


@pytest.mark.vcr
async def test_openai_responses_unified_cache_e2e(allow_model_requests: None, openai_api_key: str, vcr: Cassette):
    """With `cache=True`, GPT-5.6 gets the 30-minute TTL and an instruction breakpoint, and the second request
    reads the cache the first wrote."""
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(api_key=openai_api_key))
    # The instructions keep the text they were recorded with.
    agent = Agent(model, instructions=_STABLE_PREFIX + '\n(cache on by default)', model_settings={'cache': True})

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: OK')

    for index in (0, 1):
        body = _request_body(vcr, index)
        assert body['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
        assert body['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}
    assert (first.usage, second.usage) == snapshot(
        (
            RunUsage(
                details={'reasoning_tokens': 0},
                output_tokens=5,
                output_reasoning_tokens=0,
                cache_write_tokens=4026,
                input_tokens=4029,
                cost=Decimal('0.020242'),
                requests=1,
            ),
            RunUsage(
                details={'reasoning_tokens': 0},
                output_tokens=5,
                cache_read_tokens=4026,
                output_reasoning_tokens=0,
                input_tokens=4029,
                cost=Decimal('0.0017224'),
                requests=1,
            ),
        )
    )
    _assert_cache_usage(first.usage, second.usage)


async def test_openai_responses_unified_cache_stable_prefix_only(allow_model_requests: None):
    """`cache={'messages': False}` writes only the instruction breakpoint: `mode='explicit'` stops OpenAI from
    creating its implicit breakpoint at the end of the conversation."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await Agent(model, instructions='Support policies.', model_settings={'cache': {'messages': False}}).run(
        'Where is order 1234?'
    )

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['input'][0]['content'][0]['prompt_cache_breakpoint'] == {'mode': 'explicit'}


async def test_openai_responses_unified_cache_stable_prefix_only_with_chaining(allow_model_requests: None):
    """With server-side state the instructions can't carry a breakpoint, so `mode='explicit'` would cache
    nothing: the request keeps OpenAI's implicit breakpoint instead."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto', cache={'messages': False})

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    assert request['instructions'] == 'Support policies.'
    assert request['input'] == [{'role': 'user', 'content': 'Where is order 1234?'}]


async def test_openai_responses_unified_cache_stable_prefix_only_with_chaining_keeps_12h(allow_model_requests: None):
    """Falling back to the implicit breakpoint keeps the requested 12-hour TTL."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(
        openai_previous_response_id='auto', cache={'retention': '12h', 'messages': False}
    )

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '12h'}


async def test_openai_chat_unified_cache_stable_prefix_only_with_merged_system_messages(allow_model_requests: None):
    """Merged system messages can't carry the instruction breakpoint, so the request keeps OpenAI's implicit
    breakpoint rather than caching nothing."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(
            openai_chat_supports_multiple_system_messages=False, openai_supports_prompt_cache_breakpoints=True
        ),
    )
    agent = Agent(
        model,
        system_prompt='Support policies.',
        instructions='Answer briefly.',
        model_settings={'cache': {'messages': False}},
    )

    await agent.run('Where is order 1234?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['prompt_cache_options'] == {'mode': 'implicit', 'ttl': '30m'}
    assert kwargs['messages'] == snapshot(
        [
            {
                'role': 'system',
                'content': """\
Support policies.

Answer briefly.\
""",
            },
            {'role': 'user', 'content': 'Where is order 1234?'},
        ]
    )


async def test_openai_chat_unified_cache_stable_prefix_only_keeps_cache_point(allow_model_requests: None):
    """A `CachePoint` is a breakpoint `mode='explicit'` still writes, so the request stays prefix-only."""
    mock_client = MockOpenAI.create_mock(chat_completion())
    model = OpenAIChatModel(
        'gpt-5.6-sol',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(
            openai_chat_supports_multiple_system_messages=False, openai_supports_prompt_cache_breakpoints=True
        ),
    )
    agent = Agent(model, instructions='Support policies.', model_settings={'cache': {'messages': False}})

    await agent.run(['Long reference document.', CachePoint(), 'Where is order 1234?'])

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert kwargs['messages'][1]['content'][0] == snapshot(
        {'type': 'text', 'text': 'Long reference document.', 'prompt_cache_breakpoint': {'mode': 'explicit'}}
    )


async def test_openai_responses_explicit_prefix_only_options_win_over_fallback(allow_model_requests: None):
    """Options a caller sets take precedence over the unified setting, even when they match what it maps to,
    so the fallback to the implicit breakpoint leaves them alone."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))
    settings = OpenAIResponsesModelSettings(
        openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
        openai_cache_instructions=True,
        openai_previous_response_id='auto',
        cache={'messages': False},
    )

    await Agent(model, instructions='Support policies.', model_settings=settings).run('Where is order 1234?')

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['instructions'] == 'Support policies.'


async def test_openai_responses_unified_cache_stable_prefix_only_keeps_tool_result_breakpoint(
    allow_model_requests: None,
):
    """A leading `CachePoint` moves its breakpoint onto the preceding tool result, which `mode='explicit'`
    still writes, so the request stays prefix-only without an instruction breakpoint."""
    mock_client = MockOpenAIResponses.create_mock(responses_completion())
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(openai_client=mock_client))

    await model.request(
        _history_ending_in(ModelRequest(parts=[UserPromptPart([CachePoint(), 'Stay focused.'])])),
        {'cache': {'messages': False}},
        ModelRequestParameters(),
    )

    request = get_mock_responses_kwargs(mock_client)[0]
    assert request['prompt_cache_options'] == {'mode': 'explicit', 'ttl': '30m'}
    assert request['input'][2] == snapshot(
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [{'type': 'input_text', 'text': 'result 1', 'prompt_cache_breakpoint': {'mode': 'explicit'}}],
        }
    )
