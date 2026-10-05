from datetime import timedelta
from typing import Literal

import pytest

from pydantic_ai import Agent, CachePoint, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


@pytest.mark.parametrize('cache', [None, False, True, '1h'])
@pytest.mark.parametrize('per_block', [False, True])
async def test_default_cache_breakpoints(
    allow_model_requests: None, cache: bool | Literal['1h'] | None, per_block: bool
):
    client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(type='text', text='Done')], BetaUsage(input_tokens=5, output_tokens=1))
    )
    if per_block:
        settings = AnthropicModelSettings(
            anthropic_cache=False, anthropic_cache_messages=True if cache is None else cache
        )
    else:
        settings = AnthropicModelSettings() if cache is None else AnthropicModelSettings(anthropic_cache=cache)
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=client), settings=settings)
    agent = Agent(model, instructions='Static instructions')

    @agent.tool_plain
    def lookup() -> str:
        """Look up information."""
        return 'result'

    await agent.run(
        ['Context', CachePoint(), 'Continue'],
        message_history=[
            ModelRequest(parts=[UserPromptPart('Start')]),
            ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='old')]),
            ModelRequest(parts=[ToolReturnPart('lookup', 'old result', tool_call_id='old')]),
            ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='latest')]),
            ModelRequest(parts=[ToolReturnPart('lookup', 'result', tool_call_id='latest')]),
        ],
    )
    request = get_mock_chat_completion_kwargs(client)[0]
    enabled = cache is not False
    ttl = '1h' if cache == '1h' else '5m'
    expected = {'type': 'ephemeral', 'ttl': ttl} if enabled else None
    assert request['system'][-1].get('cache_control') == expected
    assert request['tools'][-1].get('cache_control') == expected
    calls = [block for msg in request['messages'] for block in msg['content'] if block['type'] == 'tool_use']
    assert calls[0].get('cache_control') is None
    assert calls[-1].get('cache_control') == expected
    count = sum('cache_control' in block for msg in request['messages'] for block in msg['content'])
    count += sum('cache_control' in block for block in request['system'])
    count += sum('cache_control' in tool for tool in request['tools'])
    assert count <= (4 if per_block or not enabled else 3)
    retention = timedelta(hours=1) if ttl == '1h' else timedelta(minutes=5)
    assert model.resolve_cache_retention(None) == (retention if enabled else None)


@pytest.mark.parametrize(
    'settings',
    [
        AnthropicModelSettings(anthropic_cache_messages=True),
        AnthropicModelSettings(anthropic_cache_messages='1h'),
        AnthropicModelSettings(
            anthropic_cache=True, anthropic_cache_instructions=False, anthropic_cache_tool_definitions=False
        ),
    ],
)
def test_cache_defaults_preserve_overrides(settings: AnthropicModelSettings):
    client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(type='text', text='Done')], BetaUsage(input_tokens=5, output_tokens=1))
    )
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=client))
    prepared, _ = model.prepare_request(settings, ModelRequestParameters())
    assert prepared is not None
    for key, value in settings.items():
        assert prepared[key] == value
    assert settings == {key: prepared[key] for key in settings}
    if settings.get('anthropic_cache_messages'):
        assert prepared.get('anthropic_cache') is False
