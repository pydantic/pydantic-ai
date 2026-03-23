from __future__ import annotations as _annotations

from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.direct import model_request
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.zai import ZaiModel, ZaiModelSettings
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
            ),
        ]
    )


async def test_zai_thinking_mode(allow_model_requests: None, zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=True)
    response = await model_request(model, [ModelRequest.user_text_prompt('What is 2 + 2?')], model_settings=settings)
    assert response.parts == snapshot(
        [
            ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
            TextPart(content='2 + 2 is 4.'),
        ]
    )


async def test_zai_thinking_stream(allow_model_requests: None, zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    agent = Agent(model=model, model_settings=ZaiModelSettings(zai_thinking=True))

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
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='zai'),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    cache_read_tokens=1,
                    output_tokens=59,
                    details={
                        'reasoning_tokens': 49,
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
                provider_response_id='20260323160404f91d4991ff6c48a3',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_zai_prepare_request_thinking_enabled(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=True)
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    extra_body = cast(dict[str, Any], merged_settings.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled'}


async def test_zai_prepare_request_thinking_disabled(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=False)
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    extra_body = cast(dict[str, Any], merged_settings.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'disabled'}


async def test_zai_prepare_request_preserved_thinking(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=True, zai_clear_thinking=False)
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    extra_body = cast(dict[str, Any], merged_settings.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled', 'clear_thinking': False}


async def test_zai_prepare_request_clear_thinking(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=True, zai_clear_thinking=True)
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    extra_body = cast(dict[str, Any], merged_settings.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled', 'clear_thinking': True}


async def test_zai_prepare_request_empty(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings()
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    assert merged_settings.get('extra_body') is None


async def test_zai_prepare_request_preserves_existing_extra_body(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=True, extra_body={'custom_key': 'value'})
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    extra_body = cast(dict[str, Any], merged_settings.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled'}
    assert extra_body.get('custom_key') == 'value'
