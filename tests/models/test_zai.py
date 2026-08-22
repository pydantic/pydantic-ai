from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from inline_snapshot import snapshot
from vcr.cassette import Cassette

from pydantic_ai import Agent, BinaryImage, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.zai import (
        ZaiModel,
        ZaiModelSettings,
        _zai_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.zai import ZaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_zai_model_simple(allow_model_requests: None, zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    agent = Agent(model=model)
    result = await agent.run('What is 2 + 2?')
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
                    TextPart(content='2 + 2 is 4.'),
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    output_tokens=437,
                    output_reasoning_tokens=427,
                    details={
                        'reasoning_tokens': 427,
                    },
                ),
                model_name='glm-4.7',
                timestamp=IsDatetime(),
                provider_name='zai',
                provider_url='https://api.z.ai/api/paas/v4',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='20260701073925df703dd30a854c37',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_zai_thinking_mode(allow_model_requests: None, zai_api_key: str, vcr: Cassette):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ModelSettings(thinking=True)
    response = await model_request(model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings)
    assert response.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content='2 + 2 is 4.'),
        ]
    )

    # The unified `thinking` setting must reach the wire as Z.AI's `extra_body.thinking` payload (merged to
    # the top level by the OpenAI SDK), and the base OpenAI `reasoning_effort` parameter must be suppressed.
    # VCR cassette matchers aren't sensitive to the request body, so assert it explicitly.
    assert len(vcr.requests) == 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert request_body['thinking'] == {'type': 'enabled', 'clear_thinking': False}
    assert 'reasoning_effort' not in request_body


async def test_zai_clear_thinking_without_thinking(allow_model_requests: None, zai_api_key: str, vcr: Cassette):
    """A bare `extra_body.thinking.clear_thinking` (no `type`) reaches the wire and the Z.AI API accepts it.

    On a thinking-capable model this no-`type` shape is now what every plain request sends, since
    preservation (`clear_thinking=False`) is the default — so the explicit `zai_clear_thinking=False` here
    coincides with it. The point of the recording is to confirm the real API accepts that standalone shape.
    Explicit-override behavior (e.g. `zai_clear_thinking=True`) and the default gating are unit-tested in
    `test_zai_settings_transformation` (VCR matchers aren't sensitive to the request body).
    """
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_clear_thinking=False)
    response = await model_request(model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings)
    assert response.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content='4'),
        ]
    )

    # No `type` key: the bare `clear_thinking` payload is what we're confirming the API accepts.
    assert len(vcr.requests) == 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert request_body['thinking'] == {'clear_thinking': False}


async def test_zai_preserved_thinking_round_trip(allow_model_requests: None, zai_api_key: str, vcr: Cassette):
    """End-to-end preserved thinking across turns: a prior-turn `ThinkingPart` is replayed to Z.AI in the
    next request's `reasoning_content` field, and the API accepts the round-trip.

    This is the headline `zai_clear_thinking=False` capability. The send-back transformation is unit-tested
    in `test_zai_sends_back_thinking_in_reasoning_content_field`, but VCR matchers aren't sensitive to the
    request body, so a regression there would still replay green; this records the real two-turn exchange to
    prove the replayed `reasoning_content` reaches the wire and Z.AI accepts it. A live probe confirmed
    `clear_thinking=False` preserves cross-turn reasoning markedly better than the server default
    (which clears it), and that neither path errors on the replay.
    """
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(thinking=True, zai_clear_thinking=False)

    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('What is 17 * 19? Think it through.')]
    first = await model_request(model, messages, model_settings=settings)
    assert first.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content=IsStr()),
        ]
    )

    messages.append(first)
    messages.append(ModelRequest.user_text_prompt('Now multiply that result by 2.'))
    second = await model_request(model, messages, model_settings=settings)
    assert second.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content=IsStr()),
        ]
    )

    # The prior-turn `ThinkingPart` must be replayed to Z.AI as `reasoning_content` on the second request,
    # alongside the `clear_thinking=False` payload. VCR matchers aren't sensitive to the body, so assert it.
    assert len(vcr.requests) == 2  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    second_body = json.loads(vcr.requests[1].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert second_body['thinking'] == {'type': 'enabled', 'clear_thinking': False}
    assistant_messages = [m for m in second_body['messages'] if m['role'] == 'assistant']
    assert assistant_messages == snapshot([{'role': 'assistant', 'reasoning_content': IsStr(), 'content': IsStr()}])


async def test_zai_vision_thinking(
    allow_model_requests: None, zai_api_key: str, image_content: BinaryImage, vcr: Cassette
):
    """`glm-4.6v` is a vision model that also supports thinking mode.

    Recorded against the real Z.AI API to confirm the vision profile's `supports_thinking=True`: with
    `thinking=True` and image input, the model returns a `ThinkingPart` alongside the answer.
    """
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.6v', provider=provider)
    request = ModelRequest(parts=[UserPromptPart(content=['What fruit is in this image?', image_content])])
    response = await model_request(model, [request], model_settings=ModelSettings(thinking=True))
    assert response.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content=IsStr(regex='(?is).*kiwi.*')),
        ]
    )

    # Pin the recorded request body: VCR matchers aren't body-sensitive, so asserting the wire shape here
    # (verified at record time) is what confirms `thinking` reaches the request for this vision model. The
    # live transform — including the vision profile's `supports_thinking` gating — is unit-tested in
    # `test_zai_settings_transformation` and `test_zai_provider_model_profile`.
    assert len(vcr.requests) == 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert request_body['thinking'] == {'type': 'enabled', 'clear_thinking': False}


async def test_zai_reasoning_effort(allow_model_requests: None, zai_api_key: str, vcr: Cassette):
    """On GLM-5.2, an explicit unified thinking effort level is forwarded as `extra_body.reasoning_effort`
    alongside the `thinking` object.

    Recorded against the real Z.AI API to confirm GLM-5.2 accepts the `reasoning_effort` parameter; the
    transformation itself is unit-tested in `test_zai_reasoning_effort_forwarded_when_supported` (VCR
    matchers aren't sensitive to the request body).
    """
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-5.2', provider=provider)
    settings = ModelSettings(thinking='high')
    response = await model_request(model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings)
    assert response.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content='2 + 2 = 4'),
        ]
    )

    assert len(vcr.requests) == 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert request_body['thinking'] == {'type': 'enabled', 'clear_thinking': False}
    assert request_body['reasoning_effort'] == 'high'


async def test_zai_thinking_stream(allow_model_requests: None, zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    agent = Agent(model=model, model_settings=ModelSettings(thinking=True))

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
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    output_tokens=564,
                    output_reasoning_tokens=561,
                    details={
                        'reasoning_tokens': 561,
                    },
                ),
                model_name='glm-4.7',
                timestamp=IsDatetime(),
                provider_name='zai',
                provider_url='https://api.z.ai/api/paas/v4',
                provider_details={
                    'timestamp': IsDatetime(),
                    'finish_reason': 'stop',
                },
                provider_response_id='202607010739425543ff9439144b2c',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'thinking,clear_thinking,supports_thinking,extra_body,expected',
    [
        # On thinking-capable models, cross-turn reasoning is preserved by default (`clear_thinking=False`),
        # independent of this turn's `type`.
        pytest.param(
            True,
            None,
            True,
            None,
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}}},
            id='enabled',
        ),
        pytest.param(
            False,
            None,
            True,
            None,
            {'extra_body': {'thinking': {'type': 'disabled', 'clear_thinking': False}}},
            id='disabled',
        ),
        # `True` and every effort level collapse to `enabled` — Z.AI has no effort granularity.
        pytest.param(
            'high',
            None,
            True,
            None,
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}}},
            id='effort-collapses',
        ),
        # No explicit `thinking`: the model thinks by default and prior reasoning is preserved.
        pytest.param(
            None, None, True, None, {'extra_body': {'thinking': {'clear_thinking': False}}}, id='model-default-thinking'
        ),
        # Non-thinking models receive no thinking payload at all.
        pytest.param(None, None, False, None, {}, id='non-thinking-model'),
        # An explicit `zai_clear_thinking` always wins over the default.
        pytest.param(
            True,
            True,
            True,
            None,
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': True}}},
            id='explicit-clear',
        ),
        pytest.param(
            True,
            False,
            True,
            None,
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}}},
            id='explicit-preserve',
        ),
        # An explicit setting is honored even on a non-thinking model; only the *default* is gated.
        pytest.param(
            None,
            False,
            False,
            None,
            {'extra_body': {'thinking': {'clear_thinking': False}}},
            id='explicit-on-non-thinking',
        ),
        pytest.param(
            True,
            None,
            True,
            {'custom_key': 'value'},
            {'extra_body': {'custom_key': 'value', 'thinking': {'type': 'enabled', 'clear_thinking': False}}},
            id='preserves-existing-extra-body',
        ),
    ],
)
def test_zai_settings_transformation(
    thinking: ThinkingLevel | None,
    clear_thinking: bool | None,
    supports_thinking: bool,
    extra_body: dict[str, Any] | None,
    expected: dict[str, Any],
):
    """`ZaiModelSettings` are translated into the `extra_body.thinking` payload the Z.AI API expects.

    A unit test (not VCR): this pins the request-body shape, which VCR cassette matchers aren't sensitive to.
    The resolved unified `thinking` setting arrives via `ModelRequestParameters.thinking` (the base
    `prepare_request` strips it from settings first); `zai_clear_thinking` stays on the settings. The
    end-to-end wire emission is covered by `test_zai_thinking_mode`.
    """
    settings = ZaiModelSettings()
    if clear_thinking is not None:
        settings['zai_clear_thinking'] = clear_thinking
    if extra_body is not None:
        settings['extra_body'] = extra_body

    # `supports_reasoning_effort=False`: effort granularity collapses to enabled (e.g. on glm-4.7).
    transformed = _zai_settings_to_openai_settings(
        settings,
        ModelRequestParameters(thinking=thinking),
        supports_thinking=supports_thinking,
        supports_reasoning_effort=False,
    )
    assert transformed == expected


def test_zai_thinking_silently_ignored_on_non_thinking_model(zai_api_key: str):
    """On a model whose profile has `supports_thinking=False`, the unified `thinking` setting is stripped.

    A unit test (not VCR): this exercises the base `prepare_request` gate (which the transformation function
    alone can't show) — `glm-4-32b-0414-128k` resolves to `supports_thinking=False`, so `thinking` never
    reaches the Z.AI translation and no `extra_body` is produced.
    """
    model = ZaiModel('glm-4-32b-0414-128k', provider=ZaiProvider(api_key=zai_api_key))
    merged_settings, _ = model.prepare_request(ZaiModelSettings(thinking=True), ModelRequestParameters())
    assert merged_settings == {}


def test_zai_sends_back_thinking_in_reasoning_content_field(zai_api_key: str):
    """Preserved thinking: a prior-turn `ThinkingPart` is sent back to Z.AI in the `reasoning_content`
    field (via `openai_chat_send_back_thinking_parts='field'`), not dropped or wrapped in `<think>` tags.

    A unit test (not VCR): the send-back goes in the request body, which VCR cassette matchers aren't
    sensitive to, so a regression here would still replay green against an existing cassette.
    """
    model = ZaiModel('glm-4.7', provider=ZaiProvider(api_key=zai_api_key))
    response = ModelResponse(
        parts=[
            ThinkingPart(content='2 plus 2 is 4', id='reasoning_content', provider_name='zai'),
            TextPart(content='4'),
        ]
    )
    assert model._map_model_response(response) == snapshot(  # pyright: ignore[reportPrivateUsage]
        {'role': 'assistant', 'reasoning_content': '2 plus 2 is 4', 'content': '4'}
    )


@pytest.mark.parametrize(
    'thinking,expected',
    [
        pytest.param(
            'minimal',
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'minimal'}},
            id='minimal',
        ),
        pytest.param(
            'low',
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'low'}},
            id='low',
        ),
        pytest.param(
            'medium',
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'medium'}},
            id='medium',
        ),
        pytest.param(
            'high',
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'high'}},
            id='high',
        ),
        pytest.param(
            'xhigh',
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}, 'reasoning_effort': 'xhigh'}},
            id='xhigh',
        ),
        # A bare `thinking=True` enables thinking but sends no effort, so Z.AI applies its own default.
        pytest.param(
            True, {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}}}, id='enabled-no-effort'
        ),
        pytest.param(False, {'extra_body': {'thinking': {'type': 'disabled', 'clear_thinking': False}}}, id='disabled'),
    ],
)
def test_zai_reasoning_effort_forwarded_when_supported(thinking: ThinkingLevel, expected: dict[str, Any]):
    """When the model supports reasoning effort, an explicit unified effort level is forwarded as
    `extra_body.reasoning_effort`, while a bare `thinking=True`/`False` adds none.

    Exercises the transform with `supports_reasoning_effort=True` (GLM-5.2, which also supports thinking, so
    cross-turn reasoning is preserved by default); the model-name -> flag mapping is covered by
    `test_zai_provider_model_profile`. Models without effort support collapse the level to thinking on/off
    (covered by `test_zai_settings_transformation`).
    """
    transformed = _zai_settings_to_openai_settings(
        ZaiModelSettings(),
        ModelRequestParameters(thinking=thinking),
        supports_thinking=True,
        supports_reasoning_effort=True,
    )
    assert transformed == expected


# --- Non-standard finish_reason handling (fixes pydantic/pydantic-ai#7678) ---
#
# Z.AI returns `sensitive` (content moderation) and `network_error` (proxy transport)
# as `finish_reason` values.  Neither is in the OpenAI SDK's strict 5-value Literal, so
# pydantic-ai's default `_validate_completion` hook (which re-runs `model_validate` with
# the SDK schema) aborts the run with `UnexpectedModelBehavior`.  The streaming path was
# lenient because the SDK's chunk constructor is permissive, but with the override in
# ZaiModel/ZaiStreamedResponse both paths now behave consistently: the widened Literal
# accepts the raw value, `_map_finish_reason` normalises it to a standard FinishReason,
# and the raw string stays in `provider_details['finish_reason']` for debugging.


@pytest.mark.skipif(not imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('raw_finish_reason', 'mapped_finish_reason'),
    [
        ('sensitive', 'content_filter'),
        ('network_error', 'error'),
    ],
)
def test_zai_nonstd_finish_reason_nonstream(raw_finish_reason: str, mapped_finish_reason: str) -> None:
    """Non-standard Z.AI finish_reasons pass the widened re-validation gate and map cleanly.

    Covers three contracts exercised by the ZaiModel overrides:
      * `_ZaiChatCompletion` widens the strict `finish_reason` Literal so that the
        non-standard values survive `ZaiModel._validate_completion`.
      * `ZaiModel._map_finish_reason` normalises the raw value onto the standard
        pydantic-ai `FinishReason` enum.
      * The existing `_map_provider_details` helper (inherited from the OpenAI base)
        stores the un-normalised raw string in `provider_details['finish_reason']`.

    We deliberately use `.model_construct()` (bypassing Pydantic field validation) when
    building the mock Choice / ChatCompletion, because the strict public constructors of
    the OpenAI SDK types would reject values like `sensitive` / `network_error` with
    `ValidationError` *before* pydantic-ai sees them.  The real on-the-wire Z.AI SDK path
    does *not* build these objects via the public constructor (it JSON-deserialises them
    through the transport layer), so `.model_construct()` is the faithful unit-test mock.

    This test is intentionally *not* async and does not go through `Agent.run()`: the
    conftest-wide `ALLOW_MODEL_REQUESTS = False` gate forbids model dispatches in unit
    tests.  Exercising the model-level overrides directly is a stricter unit test and
    avoids the dispatch gate entirely (see `test_openrouter.py` L590-603 for the same
    pattern used against OpenRouter's `finish_reason='error'` override).
    """
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.messages import FinishReason
    from pydantic_ai.models.zai import ZaiModel

    msg = ChatCompletionMessage(role='assistant', content='blocked')
    bad_choice = Choice.model_construct(finish_reason=raw_finish_reason, index=0, message=msg)
    bad_completion = chat.ChatCompletion.model_construct(
        id='123',
        choices=[bad_choice],
        created=1704067200,  # 2024-01-01
        model='glm-5.2',
        object='chat.completion',
    )

    model = ZaiModel('glm-5.2')

    # 1. Widened validation accepts the non-standard literal without raising.
    validated = model._validate_completion(bad_completion)  # type: ignore[reportPrivateUsage]
    v_choice = validated.choices[0]
    assert v_choice.finish_reason == raw_finish_reason, 'Raw finish_reason must survive widened validation'

    # 2. Normalisation maps to the standard pydantic-ai FinishReason str-Literal.
    mapped: FinishReason = model._map_finish_reason(v_choice.finish_reason)  # type: ignore[reportPrivateUsage, reportArgumentType, assignment]
    assert mapped is not None
    assert mapped == mapped_finish_reason

    # 3. The inherited `_map_provider_details` helper stores the un-normalised raw
    #    value for downstream forensics.  Call the same helper path the model uses
    #    at runtime: route through the model instance method (defined on the shared
    #    OpenAI base) so we pin the exact inheritance behaviour the dispatch path hits.
    provider_details: dict | None = model._map_provider_details(v_choice)  # type: ignore[reportPrivateUsage, arg-type]
    assert provider_details is not None
    assert provider_details.get('finish_reason') == raw_finish_reason


@pytest.mark.skipif(not imports_successful(), reason='openai not installed')
def test_zai_standard_finish_reasons_still_map_nonstream() -> None:
    """Regression guard: the widened Literal leaves standard `stop` mapping untouched.

    Unlike the non-standard cases above we intentionally use the strict public
    constructors for `Choice` / `ChatCompletion` — the standard `stop` literal is
    in the SDK's strict Literal, so it *should* survive public-ctor validation, and
    building it this way guards against the widened schema accidentally altering
    behaviour for normal values.
    """
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.messages import FinishReason
    from pydantic_ai.models.zai import ZaiModel

    msg = ChatCompletionMessage(role='assistant', content='hello')
    good_choice = Choice(finish_reason='stop', index=0, message=msg)
    good_completion = chat.ChatCompletion(
        id='789',
        choices=[good_choice],
        created=1704067200,
        model='glm-5.2',
        object='chat.completion',
    )

    model = ZaiModel('glm-5.2')
    validated = model._validate_completion(good_completion)  # type: ignore[reportPrivateUsage]
    v_choice = validated.choices[0]
    mapped: FinishReason = model._map_finish_reason(v_choice.finish_reason)  # type: ignore[reportPrivateUsage, assignment]
    assert mapped is not None
    assert mapped == 'stop'


@pytest.mark.skipif(not imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('raw_finish_reason', 'mapped_finish_reason'),
    [
        ('sensitive', 'content_filter'),
        ('network_error', 'error'),
    ],
)
async def test_zai_nonstd_finish_reason_stream(raw_finish_reason: str, mapped_finish_reason: str) -> None:
    """Streaming path: non-standard terminal finish_reason passes widened validation.

    Exercises the same three contracts as the non-stream test, but against the
    `ZaiStreamedResponse` overrides used on the stream consumer side:

      * `ZaiStreamedResponse._validate_response` re-validates every chunk through
        the widened `_ZaiChatCompletionChunk` TypedDict; non-standard values on the
        terminal chunk are accepted rather than raising `ValidationError`.
      * `ZaiStreamedResponse._map_finish_reason` normalises the raw value onto the
        standard pydantic-ai `FinishReason` enum.
      * Text deltas from intermediate chunks continue to stream through (incremental
        concatenation guard: `'hal' + 'f' == 'half'`).

    As with the non-stream test this intentionally stops at the model override layer
    rather than going through `Agent.run_stream()` — avoiding the conftest-wide
    `ALLOW_MODEL_REQUESTS = False` dispatch gate.
    """
    from collections.abc import AsyncIterator

    from openai.types import chat

    from pydantic_ai.messages import FinishReason
    from pydantic_ai.models.zai import ZaiStreamedResponse

    chunk_a = chat.ChatCompletionChunk.model_construct(
        id='c1',
        choices=[
            chat.chat_completion_chunk.Choice.model_construct(
                finish_reason=None,
                index=0,
                delta=chat.chat_completion_chunk.ChoiceDelta.model_construct(role='assistant', content='hal'),
            )
        ],
        created=1704067200,
        model='glm-5.2',
        object='chat.completion.chunk',
    )
    chunk_b = chat.ChatCompletionChunk.model_construct(
        id='c2',
        choices=[
            chat.chat_completion_chunk.Choice.model_construct(
                finish_reason=raw_finish_reason,
                index=0,
                delta=chat.chat_completion_chunk.ChoiceDelta.model_construct(content='f'),
            )
        ],
        created=1704067200,
        model='glm-5.2',
        object='chat.completion.chunk',
    )

    async def chunk_source() -> AsyncIterator[chat.ChatCompletionChunk]:
        yield chunk_a
        yield chunk_b

    # The ZaiStreamedResponse dataclass just needs `_response` wired up as an async
    # iterator for `_validate_response` to consume.  We bypass the public dataclass
    # constructor here to avoid needing to satisfy the unrelated required fields
    # inherited from `OpenAIStreamedResponse` (they're all dead code for the narrow
    # override paths we're exercising).
    resp = object.__new__(ZaiStreamedResponse)
    resp._response = chunk_source()

    text_parts: list[str] = []
    last_chunk: chat.ChatCompletionChunk | None = None
    async for validated_chunk in resp._validate_response():  # type: ignore[reportPrivateUsage]
        last_chunk = validated_chunk
        if validated_chunk.choices:
            delta = validated_chunk.choices[0].delta
            if delta is not None and getattr(delta, 'content', None) is not None:
                text_parts.append(delta.content)

    # 1. Incremental deltas continue to stream through even when the terminal chunk
    #    carries a non-standard finish_reason.
    assert ''.join(text_parts) == 'half', 'Streamed text deltas must concatenate correctly'
    assert last_chunk is not None
    terminal_choice = last_chunk.choices[0]

    # 2. Raw non-standard value survives widened chunk validation.
    assert terminal_choice.finish_reason == raw_finish_reason

    # 3. Normalisation maps to the standard pydantic-ai FinishReason str-Literal
    #    (same lookup as non-stream — the two paths share `_ZAI_FINISH_REASON_MAP`).
    mapped: FinishReason = ZaiStreamedResponse._map_finish_reason(  # type: ignore[reportPrivateUsage, reportArgumentType, assignment]
        resp,
        terminal_choice.finish_reason,  # type: ignore[reportArgumentType]
    )
    assert mapped is not None
    assert mapped == mapped_finish_reason

    # 4. The inherited provider-details helper stores the raw value on the terminal
    #    chunk's choice via the same path the live dispatch uses (defined on the
    #    shared OpenAI stream base).
    stream_provider_details: dict | None = resp._map_provider_details(last_chunk)  # type: ignore[reportPrivateUsage, arg-type]
    assert stream_provider_details is not None
    assert stream_provider_details.get('finish_reason') == raw_finish_reason
