from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from inline_snapshot import snapshot
from vcr.cassette import Cassette

from pydantic_ai import Agent, BinaryImage, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.direct import model_request
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
                    TextPart(content='4'),
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    cache_read_tokens=2,
                    output_tokens=108,
                    details={
                        'reasoning_tokens': 105,
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
                provider_response_id='20260217121043e8de4e8178114889',
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
    assert request_body['thinking'] == {'type': 'enabled'}
    assert 'reasoning_effort' not in request_body


async def test_zai_clear_thinking_without_thinking(allow_model_requests: None, zai_api_key: str, vcr: Cassette):
    """`zai_clear_thinking` set on its own (no unified `thinking`) reaches the wire as a bare
    `extra_body.thinking.clear_thinking`, with no `type`.

    `clear_thinking` tunes cross-turn thinking preservation independently of whether the current turn
    enables thinking, so it is emitted even when `thinking` is left to the model's default. This records
    a real request to confirm the Z.AI API accepts that standalone shape; the transformation itself is
    unit-tested in `test_zai_settings_transformation` (VCR matchers aren't sensitive to the request body).
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


async def test_zai_vision_thinking(allow_model_requests: None, zai_api_key: str, image_content: BinaryImage):
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


async def test_zai_reasoning_effort(allow_model_requests: None, zai_api_key: str, vcr: Cassette):
    """On GLM-5.2+, an explicit unified thinking effort level is forwarded as `extra_body.reasoning_effort`
    alongside the `thinking` object.

    Recorded against the real Z.AI API to confirm GLM-5.2 accepts the `reasoning_effort` parameter; the
    transformation itself is unit-tested in `test_zai_reasoning_effort_on_glm_5_2` (VCR matchers aren't
    sensitive to the request body).
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
    assert request_body['thinking'] == {'type': 'enabled'}
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
                    cache_read_tokens=2,
                    output_tokens=123,
                    details={
                        'reasoning_tokens': 113,
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
                provider_response_id='20260325232441b45c991535c342af',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'thinking,clear_thinking,extra_body,expected',
    [
        pytest.param(True, None, None, {'extra_body': {'thinking': {'type': 'enabled'}}}, id='enabled'),
        pytest.param(False, None, None, {'extra_body': {'thinking': {'type': 'disabled'}}}, id='disabled'),
        # `True` and every effort level collapse to `enabled` — Z.AI has no effort granularity.
        pytest.param('high', None, None, {'extra_body': {'thinking': {'type': 'enabled'}}}, id='effort-collapses'),
        pytest.param(None, None, None, {}, id='no-thinking'),
        pytest.param(
            True,
            False,
            None,
            {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}}},
            id='preserved-thinking',
        ),
        pytest.param(
            True, True, None, {'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': True}}}, id='clear'
        ),
        # `zai_clear_thinking` is independent of `type`: it tunes cross-turn thinking preservation even when the
        # current turn's thinking is left to the model's default, so it is emitted on its own.
        pytest.param(
            None, False, None, {'extra_body': {'thinking': {'clear_thinking': False}}}, id='clear-without-thinking'
        ),
        pytest.param(
            True,
            None,
            {'custom_key': 'value'},
            {'extra_body': {'custom_key': 'value', 'thinking': {'type': 'enabled'}}},
            id='preserves-existing-extra-body',
        ),
    ],
)
def test_zai_settings_transformation(
    thinking: ThinkingLevel | None,
    clear_thinking: bool | None,
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
        settings, ModelRequestParameters(thinking=thinking), supports_reasoning_effort=False
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
            {'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'minimal'}},
            id='minimal',
        ),
        pytest.param('low', {'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}}, id='low'),
        pytest.param(
            'medium', {'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'medium'}}, id='medium'
        ),
        pytest.param('high', {'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}}, id='high'),
        pytest.param(
            'xhigh', {'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'xhigh'}}, id='xhigh'
        ),
        # A bare `thinking=True` enables thinking but sends no effort, so Z.AI applies its own default.
        pytest.param(True, {'extra_body': {'thinking': {'type': 'enabled'}}}, id='enabled-no-effort'),
        pytest.param(False, {'extra_body': {'thinking': {'type': 'disabled'}}}, id='disabled'),
    ],
)
def test_zai_reasoning_effort_on_glm_5_2(thinking: ThinkingLevel, expected: dict[str, Any]):
    """On GLM-5.2+, an explicit unified thinking effort level is forwarded as `extra_body.reasoning_effort`.

    Earlier GLM models collapse effort to thinking on/off (covered by `test_zai_settings_transformation`).
    """
    transformed = _zai_settings_to_openai_settings(
        ZaiModelSettings(), ModelRequestParameters(thinking=thinking), supports_reasoning_effort=True
    )
    assert transformed == expected
