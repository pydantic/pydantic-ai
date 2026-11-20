from __future__ import annotations as _annotations

import json
import re
from collections.abc import Callable

import httpx2
import pytest
from dirty_equals import IsOneOf
from pydantic import BaseModel, JsonValue

from pydantic_ai import (
    Agent,
    BinaryImage,
    ModelRequest,
    ModelResponse,
    ModelSettingsDeltaPart,
    PartDeltaEvent,
    TextPart,
    ThinkingPart,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import FinishReason, ModelResponsePart
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsDecimal, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.zai import ZaiModel, ZaiModelSettings
    from pydantic_ai.providers.zai import ZaiProvider

    from .mock_openai import MockOpenAI, get_mock_chat_completion_kwargs


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


def zai_request_fields(capture: RequestCapture) -> list[dict[str, object]]:
    """The Z.AI-specific fields of every request the code actually built, in order.

    Everything Z.AI adds on top of the OpenAI chat schema travels in `extra_body`, which the OpenAI SDK
    merges into the top level of the payload: the `thinking` object, the `reasoning_effort` level, and
    whatever else a caller passed through. Reading them off the capture rather than the cassette is what
    makes these tests body-sensitive — the default VCR matchers ignore the body, so a payload that has
    since drifted still replays green against its recording.
    """
    return [
        {key: body[key] for key in ('thinking', 'reasoning_effort', 'tool_stream', 'user_id') if key in body}
        for body in capture.bodies('/chat/completions')
    ]


async def test_zai_thinking_across_turns(allow_model_requests: None, zai_api_key: str, request_capture: RequestCapture):
    """One glm-5.1 conversation over the whole thinking surface.

    Turn by turn: no thinking setting at all, an explicit effort level, then explicit overrides. Each
    turn pins the `extra_body.thinking` payload it produces, and turns 2 and 3 additionally show the
    prior turn's `ThinkingPart` replayed to Z.AI in the `reasoning_content` field (preserved thinking,
    which `clear_thinking=False` asks the API to honor) rather than dropped or wrapped in `<think>` tags.

    glm-5.1 doesn't accept a per-request `reasoning_effort`, so the `'high'` level on turn 2 collapses to
    plain enabled thinking — the profile-flag-off side of `test_zai_reasoning_effort`.
    """
    provider = ZaiProvider(api_key=zai_api_key, http_client=request_capture.client)
    agent = Agent(ZaiModel('glm-5.1', provider=provider))

    first = await agent.run('What is 17 * 19? Think it through.')
    second = await agent.run(
        'Now multiply that result by 2.',
        message_history=first.all_messages(),
        model_settings=ModelSettings(thinking='high'),
    )
    # Explicit overrides win over the defaults, and an unrelated `extra_body` key survives the merge:
    # `user_id` is Z.AI's own end-user identifier, which has no unified setting.
    third = await agent.run(
        'And what was my first question?',
        message_history=second.all_messages(),
        model_settings=ZaiModelSettings(
            thinking=False, zai_clear_thinking=True, extra_body={'user_id': 'pydantic-ai-test-user'}
        ),
    )
    assert third.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 17 * 19? Think it through.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                    TextPart(content=IsStr(regex=r'(?s).*\b323\b.*')),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 638},
                    input_tokens=17,
                    output_tokens=947,
                    output_reasoning_tokens=638,
                    cost=IsOneOf(None, IsDecimal(gt=0)),
                ),
                model_name='glm-5.1',
                model_id='zai:glm-5.1',
                timestamp=IsDatetime(),
                provider_name='zai',
                provider_url='https://api.z.ai/api/paas/v4',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='20260830064050fe00875d4340443e',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Now multiply that result by 2.', timestamp=IsDatetime()),
                    ModelSettingsDeltaPart(settings={'thinking': 'high'}, top_level=True),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                    TextPart(content=IsStr(regex=r'(?s).*\b646\b.*')),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 144},
                    input_tokens=974,
                    output_tokens=216,
                    output_reasoning_tokens=144,
                    cost=IsOneOf(None, IsDecimal(gt=0)),
                ),
                model_name='glm-5.1',
                model_id='zai:glm-5.1',
                timestamp=IsDatetime(),
                provider_name='zai',
                provider_url='https://api.z.ai/api/paas/v4',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='20260830064118e056bf8517e34d75',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='And what was my first question?', timestamp=IsDatetime()),
                    ModelSettingsDeltaPart(settings={'thinking': None}, top_level=True),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Your first question was: "What is 17 * 19? Think it through."')],
                usage=RequestUsage(
                    details={'reasoning_tokens': 0},
                    input_tokens=417,
                    output_tokens=19,
                    output_reasoning_tokens=0,
                    cost=IsOneOf(None, IsDecimal(gt=0)),
                ),
                model_name='glm-5.1',
                model_id='zai:glm-5.1',
                timestamp=IsDatetime(),
                provider_name='zai',
                provider_url='https://api.z.ai/api/paas/v4',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='2026083006412652767a3b187c487b',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert zai_request_fields(request_capture) == snapshot(
        [
            {'thinking': {'clear_thinking': False}},
            {'thinking': {'type': 'enabled', 'clear_thinking': False}},
            {'thinking': {'type': 'disabled', 'clear_thinking': True}, 'user_id': 'pydantic-ai-test-user'},
        ]
    )

    # Preserved thinking is only preserved if it is replayed verbatim: Z.AI requires the complete,
    # unmodified `reasoning_content` back, in the original order. Turn 2 resends turn 1's reasoning and
    # turn 3 resends both, so the replayed strings are exactly the `ThinkingPart` contents we parsed out.
    first_thinking, second_thinking = [
        part.content for result in (first, second) for part in result.response.parts if isinstance(part, ThinkingPart)
    ]
    replayed_thinking: list[str] = []
    for body in request_capture.bodies('/chat/completions'):
        messages = body['messages']
        assert isinstance(messages, list)
        for message in messages:
            assert isinstance(message, dict)
            if message.get('role') == 'assistant':
                reasoning_content = message['reasoning_content']
                assert isinstance(reasoning_content, str)
                replayed_thinking.append(reasoning_content)
    assert replayed_thinking == [first_thinking, first_thinking, second_thinking]


async def test_zai_thinking_stream(allow_model_requests: None, zai_api_key: str, request_capture: RequestCapture):
    """Streaming sends the same thinking payload, and Z.AI's `reasoning_content` deltas rebuild a `ThinkingPart`.

    The streamed path has its own parser and its own finish-reason mapping (`ZaiStreamedResponse`), so it
    gets its own recording rather than riding along on the non-streaming conversation.
    """
    provider = ZaiProvider(api_key=zai_api_key, http_client=request_capture.client)
    agent = Agent(ZaiModel('glm-5.1', provider=provider), model_settings=ModelSettings(thinking=True))

    result: AgentRunResult[str] | None = None
    async with agent.run_stream_events(user_prompt='What is 2 + 2?') as event_stream:
        async for event in event_stream:
            if isinstance(event, AgentRunResultEvent):
                result = event.result

    assert result is not None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2 + 2?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                    TextPart(content='2 + 2 = 4'),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 88},
                    output_tokens=96,
                    input_tokens=13,
                    output_reasoning_tokens=88,
                    cost=IsOneOf(None, IsDecimal(gt=0)),
                ),
                model_name='glm-5.1',
                model_id='zai:glm-5.1',
                timestamp=IsDatetime(),
                provider_name='zai',
                provider_url='https://api.z.ai/api/paas/v4',
                provider_details={'timestamp': IsDatetime(), 'finish_reason': 'stop'},
                provider_response_id='2026083006413188f010e490ed4a4a',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert zai_request_fields(request_capture) == snapshot([{'thinking': {'type': 'enabled', 'clear_thinking': False}}])


@pytest.mark.parametrize(
    'model_name,expected_exchanges',
    [
        pytest.param(
            'glm-5.2',
            snapshot(
                [
                    (
                        'minimal',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'minimal'},
                        [TextPart(content='2 + 2 = 4')],
                    ),
                    (
                        'low',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                    (
                        'medium',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'medium'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 equals 4.'),
                        ],
                    ),
                    (
                        'high',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'high'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 equals 4.'),
                        ],
                    ),
                    (
                        'xhigh',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'xhigh'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4.'),
                        ],
                    ),
                    (
                        True,
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                    (
                        False,
                        {'thinking': {'type': 'disabled', 'clear_thinking': False}},
                        [TextPart(content='2 + 2 = 4')],
                    ),
                ]
            ),
            id='glm-5.2',
        ),
        pytest.param(
            'glm-5.3',
            snapshot(
                [
                    (
                        'minimal',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [TextPart(content='2 + 2 = 4')],
                    ),
                    (
                        'low',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [TextPart(content='2 + 2 = **4**')],
                    ),
                    (
                        'medium',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'high'},
                        [TextPart(content='2 + 2 = **4**')],
                    ),
                    (
                        'high',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'high'},
                        [TextPart(content='2 + 2 = **4**')],
                    ),
                    (
                        'xhigh',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'max'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                    (
                        True,
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                    (
                        False,
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                ]
            ),
            id='glm-5.3',
        ),
        # `glm-5.3-flash` picks up the whole GLM-5.3 profile — effort support, the mapping, and
        # always-on thinking — purely by prefix match, so it gets the same recording rather than
        # a prefix assertion standing in for one.
        pytest.param(
            'glm-5.3-flash',
            snapshot(
                [
                    (
                        'minimal',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [TextPart(content='2 + 2 = 4')],
                    ),
                    (
                        'low',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [TextPart(content='2 + 2 = 4')],
                    ),
                    (
                        'medium',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'high'},
                        [TextPart(content='2 + 2 = 4')],
                    ),
                    (
                        'high',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'high'},
                        [TextPart(content=IsStr())],
                    ),
                    (
                        'xhigh',
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'max'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                    (
                        True,
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                    (
                        False,
                        {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'},
                        [
                            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                            TextPart(content='2 + 2 = 4'),
                        ],
                    ),
                ]
            ),
            id='glm-5.3-flash',
        ),
    ],
)
async def test_zai_reasoning_effort(
    allow_model_requests: None,
    zai_api_key: str,
    request_capture: RequestCapture,
    model_name: str,
    expected_exchanges: list[tuple[ThinkingLevel, dict[str, object], list[ModelResponsePart]]],
):
    """Every model that accepts a per-request `reasoning_effort`, against the real API.

    Each unified thinking value gets its own recorded request, and each snapshot row pairs the level
    that went in with the payload we sent and the parts that came back.

    GLM-5.2 accepts every unified level, so its profile carries no mapping and each level goes out
    unchanged. GLM-5.3 and GLM-5.3-flash only accept `low`/`high`/`max`, so their profile maps the
    rest onto the nearest supported one; they also always reason, so `thinking=False` sends their
    lowest effort, `low`, instead of `type: 'disabled'`. A bare `thinking=True` sends no effort on any
    of them, leaving Z.AI to apply its own default.

    The GLM-5.3 `False` rows replay a response recorded when `thinking=False` sent no `type` or effort
    (no Z.AI key was at hand to re-record them); the payload they now pin is the same one the `minimal`
    and `low` rows send, which those rows' recordings show the API accepting.

    Recording every level is what makes the API's acceptance of each value we emit part of the test
    rather than an assumption. The returned parts are the other half of it, and they are not uniform:
    on this prompt the lower efforts often come back with no reasoning at all.

    The flag-off side of `zai_supports_reasoning_effort` is `test_zai_thinking_across_turns`, where
    an effort level collapses to plain enabled thinking.
    """
    provider = ZaiProvider(api_key=zai_api_key, http_client=request_capture.client)
    agent = Agent(ZaiModel(model_name, provider=provider))

    levels: tuple[ThinkingLevel, ...] = ('minimal', 'low', 'medium', 'high', 'xhigh', True, False)
    results = [await agent.run('What is 2 + 2?', model_settings=ModelSettings(thinking=level)) for level in levels]

    assert [
        (level, fields, result.response.parts)
        for level, fields, result in zip(levels, zai_request_fields(request_capture), results, strict=True)
    ] == expected_exchanges


async def test_zai_non_thinking_model(allow_model_requests: None, zai_api_key: str, request_capture: RequestCapture):
    """`glm-4-32b-0414-128k` has no thinking support, so nothing thinking-related reaches the wire by default.

    The unified `thinking` setting is dropped by the base `prepare_request` gate, and the
    `zai_clear_thinking` default is left unset rather than defaulting to `False`, so the request carries
    no `extra_body` at all. An explicit `zai_clear_thinking` is still honored — only the default is gated
    — and the recording confirms the API accepts that bare `clear_thinking` payload on a model that never
    reasons.
    """
    provider = ZaiProvider(api_key=zai_api_key, http_client=request_capture.client)
    agent = Agent(ZaiModel('glm-4-32b-0414-128k', provider=provider))

    gated = await agent.run('What is 2 + 2?', model_settings=ModelSettings(thinking=True))
    explicit = await agent.run('What is 2 + 2?', model_settings=ZaiModelSettings(zai_clear_thinking=False))

    assert zai_request_fields(request_capture) == snapshot([{}, {'thinking': {'clear_thinking': False}}])
    assert [gated.response.parts, explicit.response.parts] == snapshot(
        [[TextPart(content='2 + 2 equals 4.')], [TextPart(content='2 + 2 equals 4.')]]
    )


async def test_zai_vision_thinking(
    allow_model_requests: None, zai_api_key: str, image_content: BinaryImage, request_capture: RequestCapture
):
    """`glm-4.6v` is a vision model that also supports thinking mode.

    Confirms the vision profile's `supports_thinking=True` end to end: with `thinking=True` and image
    input, the request carries the thinking payload and the model returns a `ThinkingPart` alongside its
    answer.
    """
    provider = ZaiProvider(api_key=zai_api_key, http_client=request_capture.client)
    agent = Agent(ZaiModel('glm-4.6v', provider=provider), model_settings=ModelSettings(thinking=True))

    result = await agent.run(['What fruit is in this image?', image_content])

    assert result.response.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(
                content="""\

The fruit in the image is a kiwi.\
"""
            ),
        ]
    )
    assert zai_request_fields(request_capture) == snapshot([{'thinking': {'type': 'enabled', 'clear_thinking': False}}])


@pytest.mark.parametrize(
    'raw_finish_reason,expected_finish_reason',
    [
        pytest.param('sensitive', 'content_filter', id='sensitive'),
        pytest.param('model_context_window_exceeded', 'length', id='model_context_window_exceeded'),
        pytest.param('network_error', 'error', id='network_error'),
    ],
)
async def test_zai_non_standard_finish_reason(
    allow_model_requests: None, raw_finish_reason: str, expected_finish_reason: FinishReason
):
    """Z.AI's non-standard `finish_reason` values are normalized instead of failing validation.

    Z.AI documents three termination causes the OpenAI schema has no room for, so
    `OpenAIChatModel._validate_completion` rejected the whole response and ended the run with
    `UnexpectedModelBehavior` (#7678). The raw string stays in `provider_details`.

    Unit tests, because none of the three can be provoked on demand: the requests that should
    trigger them come back as an HTTP 400 instead (`1261` for an over-long prompt, `1301` for
    disallowed content), and a 120k-token prompt still finished with `stop`. They arrive as a
    finish reason only when the condition develops while the response is being produced, which no
    request can stage.
    """
    message = ChatCompletionMessage(role='assistant', content='Partial answer.')
    completion = chat.ChatCompletion.model_construct(
        id='123',
        choices=[Choice.model_construct(finish_reason=raw_finish_reason, index=0, message=message)],
        created=1704067200,  # 2024-01-01
        model='glm-5.1',
        object='chat.completion',
    )
    model = ZaiModel('glm-5.1', provider=ZaiProvider(openai_client=MockOpenAI.create_mock(completion)))

    result = await Agent(model).run('Tell me something.')

    # The text produced before the non-standard finish reason is kept, not discarded.
    assert result.output == 'Partial answer.'
    assert result.response.finish_reason == expected_finish_reason
    assert (result.response.provider_details or {})['finish_reason'] == raw_finish_reason


def zai_chunk(
    delta: ChoiceDelta, finish_reason: str | None = None, *, model: str = 'glm-5.2'
) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk.model_construct(
        id='123',
        choices=[ChunkChoice.model_construct(index=0, delta=delta, finish_reason=finish_reason)],
        created=1704067200,  # 2024-01-01
        model=model,
        object='chat.completion.chunk',
    )


@pytest.mark.parametrize(
    'raw_finish_reason,expected_finish_reason',
    [
        pytest.param('sensitive', 'content_filter', id='sensitive'),
        pytest.param('model_context_window_exceeded', 'length', id='model_context_window_exceeded'),
        pytest.param('network_error', 'error', id='network_error'),
    ],
)
async def test_zai_non_standard_finish_reason_stream(
    allow_model_requests: None, raw_finish_reason: str, expected_finish_reason: FinishReason
):
    """Streamed responses map the same non-standard values, and keep the text received before them.

    A stream never raised, but every non-standard value came out as `finish_reason=None`, hiding a
    moderation hit or an interrupted generation from anything that branches on it. Unit test for the
    same reason as `test_zai_non_standard_finish_reason`.
    """

    stream = [
        zai_chunk(ChoiceDelta(content='Partial ', role='assistant'), model='glm-5.1'),
        zai_chunk(ChoiceDelta(content='answer.', role='assistant'), finish_reason=raw_finish_reason, model='glm-5.1'),
    ]
    model = ZaiModel('glm-5.1', provider=ZaiProvider(openai_client=MockOpenAI.create_mock_stream(stream)))

    async with Agent(model).run_stream('Tell me something.') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['Partial ', 'Partial answer.'])

    assert result.response.finish_reason == expected_finish_reason
    assert (result.response.provider_details or {})['finish_reason'] == raw_finish_reason


async def test_zai_sensitive_without_content_raises_content_filter_error(allow_model_requests: None):
    """A moderation hit that returns no content ends the run as a content filter, naming Z.AI's reason.

    Mapping `sensitive` onto `content_filter` is what lets the agent loop and
    [`RaiseContentFilterError`][pydantic_ai.capabilities.content_filter.RaiseContentFilterError]
    recognize it, and the raw value kept in `provider_details` is what they report.
    """
    completion = chat.ChatCompletion.model_construct(
        id='123',
        choices=[
            Choice.model_construct(
                finish_reason='sensitive', index=0, message=ChatCompletionMessage(role='assistant', content='')
            )
        ],
        created=1704067200,  # 2024-01-01
        model='glm-5.1',
        object='chat.completion',
    )
    model = ZaiModel('glm-5.1', provider=ZaiProvider(openai_client=MockOpenAI.create_mock(completion)))

    with pytest.raises(ContentFilterError, match=re.escape("Content filter triggered. Finish reason: 'sensitive'")):
        await Agent(model).run('Tell me something.')


def weather(city: str) -> str:
    return 'sunny'


class Forecast(BaseModel):
    weather: str


@pytest.mark.parametrize(
    'model_name,stream,tools,output,settings,expected_extra_body',
    [
        pytest.param(
            'glm-5.2',
            True,
            [weather],
            'Paris is sunny.',
            None,
            snapshot({'thinking': {'clear_thinking': False}, 'tool_stream': True}),
            id='streamed-with-function-tools',
        ),
        pytest.param(
            'glm-5.2',
            True,
            [],
            Forecast(weather='sunny'),
            None,
            snapshot({'thinking': {'clear_thinking': False}, 'tool_stream': True}),
            id='streamed-with-output-tool',
        ),
        pytest.param(
            'glm-5.2',
            True,
            [weather],
            'Paris is sunny.',
            ModelSettings(extra_body={'tool_stream': False}),
            snapshot({'tool_stream': False, 'thinking': {'clear_thinking': False}}),
            id='explicit-tool-stream-kept',
        ),
        pytest.param(
            'glm-5.2',
            True,
            [],
            'Paris is sunny.',
            None,
            snapshot({'thinking': {'clear_thinking': False}}),
            id='streamed-without-tools',
        ),
        pytest.param(
            'glm-5.2',
            False,
            [weather],
            'Paris is sunny.',
            None,
            snapshot({'thinking': {'clear_thinking': False}}),
            id='non-streamed',
        ),
        pytest.param(
            'glm-4.5',
            True,
            [weather],
            'Paris is sunny.',
            None,
            snapshot({'thinking': {'clear_thinking': False}}),
            id='unflagged-model',
        ),
    ],
)
async def test_zai_tool_stream_extra_body(
    allow_model_requests: None,
    model_name: str,
    stream: bool,
    tools: list[Callable[[str], str]],
    output: str | Forecast,
    settings: ModelSettings | None,
    expected_extra_body: dict[str, object],
):
    """`tool_stream` goes only on streamed requests with tools for a flagged model, and an explicit value wins.

    Function tools and output tools both count. Each case pins the whole `extra_body` handed to the SDK, so it
    also shows `tool_stream` sitting next to the `thinking` payload rather than replacing it. Unit test because
    the default cassette matchers ignore the request body, so a recording could not catch a drifted `extra_body`.
    """
    if isinstance(output, Forecast):
        tool_call = ChoiceDeltaToolCallFunction(name='final_result', arguments=output.model_dump_json())
        delta = ChoiceDelta(
            role='assistant',
            tool_calls=[ChoiceDeltaToolCall(index=0, id='call_1', type='function', function=tool_call)],
        )
    else:
        delta = ChoiceDelta(role='assistant', content=output)
    if stream:
        client = MockOpenAI.create_mock_stream([zai_chunk(delta, finish_reason='stop', model=model_name)])
    else:
        client = MockOpenAI.create_mock(
            chat.ChatCompletion.model_construct(
                id='123',
                choices=[
                    Choice.model_construct(
                        finish_reason='stop',
                        index=0,
                        message=ChatCompletionMessage(role='assistant', content=delta.content),
                    )
                ],
                created=1704067200,  # 2024-01-01
                model=model_name,
                object='chat.completion',
            )
        )
    model = ZaiModel(model_name, provider=ZaiProvider(openai_client=client))
    agent = Agent(model, output_type=type(output), tools=tools)

    if stream:
        async with agent.run_stream('What is the weather in Paris?', model_settings=settings) as result:
            assert await result.get_output() == output
    else:
        assert (await agent.run('What is the weather in Paris?', model_settings=settings)).output == output

    assert get_mock_chat_completion_kwargs(client)[-1].get('extra_body') == expected_extra_body


@pytest.mark.parametrize(
    'tools,output_type,tool_name,arguments,expected_output',
    [
        pytest.param([weather], str, 'weather', ['{"city": ', '"Paris"}'], 'It is sunny in Paris.', id='function-tool'),
        pytest.param(
            [], Forecast, 'final_result', ['{"weather": ', '"sunny"}'], Forecast(weather='sunny'), id='output-tool'
        ),
    ],
)
async def test_zai_tool_stream_streams_tool_call_arguments(
    allow_model_requests: None,
    request_capture: RequestCapture,
    tools: list[Callable[[str], str]],
    output_type: type[str] | type[Forecast],
    tool_name: str,
    arguments: list[str],
    expected_output: str | Forecast,
):
    """A streamed GLM-5.2 tool call's arguments reach the run as separate deltas, for function and output tools.

    The fake endpoint answers the way Z.AI documents `tool_stream`: it splits the arguments across chunks only
    when the request sets `tool_stream`, and sends them whole otherwise, so this fails if `ZaiModel` stops sending
    it. Unit test until a live `glm-5.2` recording exists, which needs a standard Z.AI key.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        if len(request_capture.paths) == 1:
            parts = arguments if request_capture.body('/chat/completions').get('tool_stream') else [''.join(arguments)]
            deltas: list[dict[str, JsonValue]] = [
                {
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': tool_name, 'arguments': parts[0]},
                        }
                    ]
                },
                *({'tool_calls': [{'index': 0, 'function': {'arguments': part}}]} for part in parts[1:]),
            ]
            finish_reason = 'tool_calls'
        else:
            deltas = [{'content': 'It is sunny in Paris.'}]
            finish_reason = 'stop'
        chunks = [{'delta': delta, 'finish_reason': None} for delta in deltas] + [
            {'delta': {}, 'finish_reason': finish_reason}
        ]
        content = ''.join(
            'data: '
            + json.dumps(
                {
                    'id': '123',
                    'object': 'chat.completion.chunk',
                    'created': 1704067200,  # 2024-01-01
                    'model': 'glm-5.2',
                    'choices': [{'index': 0, **chunk}],
                }
            )
            + '\n\n'
            for chunk in chunks
        )
        return httpx2.Response(
            200, content=f'{content}data: [DONE]\n\n'.encode(), headers={'content-type': 'text/event-stream'}
        )

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    ) as http_client:
        # The cassette layer ignores localhost, so the request reaches `handler`.
        provider = ZaiProvider(api_key='test', base_url='http://localhost/api/paas/v4', http_client=http_client)
        agent = Agent(ZaiModel('glm-5.2', provider=provider), output_type=output_type, tools=tools)
        async with agent.run_stream_events('Weather in Paris?') as stream:
            events = [event async for event in stream]

    assert [
        event.delta.args_delta
        for event in events
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta)
    ] == arguments[1:]
    assert {fields.get('tool_stream') for fields in zai_request_fields(request_capture)} == {True}
    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == expected_output
