import importlib
from typing import get_args

import pytest

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings, ServiceTier, merge_model_settings

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@pytest.fixture(params=['openai_', 'anthropic_', 'bedrock_', 'groq_', 'gemini_', 'mistral_', 'cohere_'])
def settings(request: pytest.FixtureRequest) -> tuple[type[ModelSettings], str]:
    prefix_cls_name = request.param.replace('_', '')
    try:
        module = importlib.import_module(f'pydantic_ai.models.{prefix_cls_name}')
    except ImportError:  # pragma: lax no cover
        pytest.skip(f'{prefix_cls_name} is not installed')
    capitalized_prefix = prefix_cls_name.capitalize().replace('Openai', 'OpenAI')
    cls = getattr(module, capitalized_prefix + 'ModelSettings')
    return cls, request.param


def test_specific_prefix_settings(settings: tuple[type[ModelSettings], str]):
    settings_cls, prefix = settings
    global_settings = set(ModelSettings.__annotations__.keys())
    specific_settings = set(settings_cls.__annotations__.keys()) - global_settings
    assert all(setting.startswith(prefix) for setting in specific_settings), (
        f'{prefix} is not a prefix for {specific_settings}'
    )


def test_model_settings_has_service_tier():
    """Guard against accidental rename or removal of the top-level service_tier key."""
    assert 'service_tier' in ModelSettings.__annotations__


def test_service_tier_literal_members():
    """Lock in the cross-provider ServiceTier values so downstream mappings stay in sync."""
    assert get_args(ServiceTier) == ('auto', 'default', 'flex', 'priority')


@pytest.mark.parametrize(
    'model', ['openai', 'anthropic', 'bedrock', 'mistral', 'groq', 'cohere', 'google'], indirect=True
)
async def test_stop_settings(allow_model_requests: None, model: Model) -> None:
    agent = Agent(model=model, model_settings=ModelSettings(stop_sequences=['Paris']))
    result = await agent.run(
        'What is the capital of France? Give me an answer that contains the word "Paris", but is not the first word.'
    )

    # NOTE: Bedrock has a slightly different behavior. It will include the stop sequence in the response.
    if model.system == 'bedrock':
        assert result.output.endswith('Paris')
    else:
        assert 'Paris' not in result.output


class TestMergeModelSettingsThinking:
    """merge_model_settings with unified thinking fields."""

    def test_merge_thinking_bool_override(self):
        base: ModelSettings = {'thinking': True}
        overrides: ModelSettings = {'thinking': False}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') is False

    def test_merge_effort_override(self):
        base: ModelSettings = {'thinking': 'low'}
        overrides: ModelSettings = {'thinking': 'high'}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') == 'high'

    def test_merge_preserves_non_thinking_settings(self):
        base: ModelSettings = {'max_tokens': 1000, 'temperature': 0.5}
        overrides: ModelSettings = {'thinking': True}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('max_tokens') == 1000
        assert result.get('temperature') == 0.5
        assert result.get('thinking') is True

    def test_merge_with_none_returns_base(self):
        base: ModelSettings = {'thinking': True}
        result = merge_model_settings(base, None)
        assert result == base

    def test_merge_with_none_base_returns_overrides(self):
        overrides: ModelSettings = {'thinking': True}
        result = merge_model_settings(None, overrides)
        assert result == overrides

    def test_merge_with_both_none(self):
        result = merge_model_settings(None, None)
        assert result is None
