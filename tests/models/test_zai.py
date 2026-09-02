from __future__ import annotations as _annotations

import re

import pytest

from pydantic_ai import Agent, BinaryImage, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import FinishReason, ModelResponsePart
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.zai import ZaiModel, ZaiModelSettings
    from pydantic_ai.providers.zai import ZaiProvider

    from .mock_openai import MockOpenAI


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
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
        {key: body[key] for key in ('thinking', 'reasoning_effort', 'user_id') if key in body}
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
                    details={'reasoning_tokens': 638}, input_tokens=17, output_tokens=947, output_reasoning_tokens=638
                ),
                model_name='glm-5.1',
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
                parts=[UserPromptPart(content='Now multiply that result by 2.', timestamp=IsDatetime())],
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
                    details={'reasoning_tokens': 144}, input_tokens=974, output_tokens=216, output_reasoning_tokens=144
                ),
                model_name='glm-5.1',
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
                parts=[UserPromptPart(content='And what was my first question?', timestamp=IsDatetime())],
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
                ),
                model_name='glm-5.1',
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
                    details={'reasoning_tokens': 88}, output_tokens=96, input_tokens=13, output_reasoning_tokens=88
                ),
                model_name='glm-5.1',
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
                        {'thinking': {'clear_thinking': False}},
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
                        {'thinking': {'clear_thinking': False}},
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
    rest onto the nearest supported one; they also always reason, so `thinking=False` is dropped
    before it reaches the wire instead of becoming `type: 'disabled'`. A bare `thinking=True` sends
    no effort on any of them, leaving Z.AI to apply its own default.

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

    def chunk(content: str, finish_reason: str | None = None) -> chat.ChatCompletionChunk:
        return chat.ChatCompletionChunk.model_construct(
            id='123',
            choices=[
                ChunkChoice.model_construct(
                    index=0, delta=ChoiceDelta(content=content, role='assistant'), finish_reason=finish_reason
                )
            ],
            created=1704067200,  # 2024-01-01
            model='glm-5.1',
            object='chat.completion.chunk',
        )

    stream = [chunk('Partial '), chunk('answer.', finish_reason=raw_finish_reason)]
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
