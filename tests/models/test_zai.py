from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from inline_snapshot import snapshot
from vcr.cassette import Cassette

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
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


async def test_zai_thinking_stream(allow_model_requests: None, zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    agent = Agent(model=model, model_settings=ModelSettings(thinking=True))

    result: AgentRunResult[str] | None = None
    async for event in agent.run_stream_events(user_prompt='What is 2 + 2?'):
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

    transformed = _zai_settings_to_openai_settings(settings, ModelRequestParameters(thinking=thinking))
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
