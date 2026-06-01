"""Tests for mapping the unified `thinking` setting to xAI `reasoning_effort`."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, ModelResponse
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.profiles.grok import GrokModelProfile
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from ...conftest import try_import
from ..mock_xai import MockXai, create_response, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from pydantic_ai.models import xai as xai_module
    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_xai_unified_thinking(allow_model_requests: None, xai_provider: XaiProvider):
    """Test that unified thinking='high' flows through to xAI reasoning_effort."""
    m = XaiModel('grok-3-mini', provider=xai_provider)
    agent = Agent(m, model_settings={'thinking': 'high'})

    result = await agent.run('What is 2+2?')
    assert '4' in result.output
    # Verify we get thinking parts (reasoning model with high effort)
    response_messages = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    assert len(response_messages) >= 1
    # The reasoning model should produce some output
    assert result.output


async def test_xai_unified_thinking_false(allow_model_requests: None, xai_provider: XaiProvider):
    """Test that unified thinking=False on a reasoning model is silently ignored (no reasoning_effort sent)."""
    m = XaiModel('grok-3-mini', provider=xai_provider)
    agent = Agent(m, model_settings={'thinking': False})

    result = await agent.run('What is 2+2?')
    assert '4' in result.output


@pytest.mark.parametrize(
    ('thinking', 'expected_reasoning_effort'),
    [
        (False, 'none'),
        # `True` maps to the model's default: reasoning_effort is omitted so Grok 4.3 applies its own default.
        (True, None),
        ('minimal', 'low'),
        ('low', 'low'),
        ('medium', 'medium'),
        ('high', 'high'),
        ('xhigh', 'high'),
    ],
)
async def test_xai_grok_43_unified_thinking_reasoning_effort(
    allow_model_requests: None, thinking: ThinkingLevel, expected_reasoning_effort: str | None
) -> None:
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    m = XaiModel('grok-4.3', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(thinking=thinking))

    result = await agent.run('Hello')

    assert result.output == 'ok'
    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    if expected_reasoning_effort is None:
        assert 'reasoning_effort' not in kwargs
    else:
        assert kwargs['reasoning_effort'] == expected_reasoning_effort


async def test_xai_grok_43_explicit_reasoning_effort_takes_precedence(allow_model_requests: None) -> None:
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    m = XaiModel('grok-4.3', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=XaiModelSettings(thinking='high', xai_reasoning_effort='none'))

    result = await agent.run('Hello')

    assert result.output == 'ok'
    assert get_mock_chat_create_kwargs(mock_client)[0]['reasoning_effort'] == 'none'


async def test_xai_grok_3_mini_unified_medium_maps_to_high(allow_model_requests: None) -> None:
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    m = XaiModel('grok-3-mini', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(thinking='medium'))

    result = await agent.run('Hello')

    assert result.output == 'ok'
    assert get_mock_chat_create_kwargs(mock_client)[0]['reasoning_effort'] == 'high'


async def test_xai_grok_3_mini_unified_true_maps_to_high(allow_model_requests: None) -> None:
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    m = XaiModel('grok-3-mini', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(thinking=True))

    result = await agent.run('Hello')

    assert result.output == 'ok'
    assert get_mock_chat_create_kwargs(mock_client)[0]['reasoning_effort'] == 'high'


@pytest.mark.parametrize(
    ('model_name', 'thinking', 'profile'),
    [
        ('grok-3-mini', False, None),
        ('grok-3-fast', 'high', None),
        ('grok-custom', 'high', GrokModelProfile(supports_thinking=True)),
    ],
)
async def test_xai_unified_thinking_omits_reasoning_effort(
    allow_model_requests: None, model_name: str, thinking: ThinkingLevel, profile: GrokModelProfile | None
) -> None:
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(
        model_name,
        provider=XaiProvider(xai_client=mock_client),
        profile=profile,
    )
    agent = Agent(m, model_settings=ModelSettings(thinking=thinking))

    result = await agent.run('Hello')

    assert result.output == 'ok'
    assert 'reasoning_effort' not in get_mock_chat_create_kwargs(mock_client)[0]


def test_xai_effort_map_deprecated():
    """`XAI_EFFORT_MAP` stays importable but deprecated; the mapping is now profile-derived."""
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`XAI_EFFORT_MAP` is deprecated'):
        mapping = xai_module.XAI_EFFORT_MAP
    assert mapping[True] == 'high'

    with pytest.raises(AttributeError, match=r'has no attribute'):
        xai_module.DefinitelyDoesNotExist
