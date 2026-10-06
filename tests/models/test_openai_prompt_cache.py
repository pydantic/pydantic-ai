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

from decimal import Decimal
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from cassetter import Cassette
from pydantic import BaseModel

from pydantic_ai import Agent, BinaryContent, CachePoint, ImageUrl, PromptedOutput
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    CompactionPart,
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
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from .._inline_snapshot import snapshot
from ..cassette_utils import request_json
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
    from openai import AsyncOpenAI
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
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.native_tools._tool_search import ToolSearchTool
    from pydantic_ai.profiles.openai import OpenAIModelProfile
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

    That is where a tail reminder (`SystemReminders`, `Planning`) puts its breakpoint once a tool loop
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
                    'arguments': '{"order_id": "1234"}',
                    'call_id': 'call_1',
                    'type': 'function_call',
                    'id': 'fc_1',
                },
                {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'shipped'},
            ],
        ]
    )


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
    agent = Agent(_diagnostics_model('gpt-5.6-sol', openai_api_key, request_capture))

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
